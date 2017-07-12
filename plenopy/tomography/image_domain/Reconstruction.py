import numpy as np
from ... import light_field
from ...image import ImageRays
from .. import ray_and_voxel
import matplotlib.pyplot as plt
from .DepthOfFieldBinning import DepthOfFieldBinning
from .transform import object_distance_2_image_distance as g2b
from .transform import image_distance_2_object_distance as b2g
from joblib import Memory
import os
from ..simulation_truth import emission_positions_of_photon_bunches
from . import transform
from ..filtered_back_projection import ramp_kernel_in_frequency_space 
from ..filtered_back_projection import frequency_filter 

cachedir = '/tmp/plenopy'
os.makedirs(cachedir, exist_ok=True)
memory = Memory(cachedir=cachedir, verbose=0)


class Reconstruction(object):

    def __init__(self, event, dof_binning=None, apply_frequency_filter=False):
        if dof_binning == None:
            focal_length = event.light_field.expected_focal_length_of_imaging_system
            self.binning = DepthOfFieldBinning(
                cx_min=1.1*event.light_field.cx_mean.min(),
                cx_max=1.1*event.light_field.cx_mean.max(),
                cy_min=1.1*event.light_field.cy_mean.min(),
                cy_max=1.1*event.light_field.cy_mean.max(),
                focal_length=focal_length,
            )

        self.event = event

        # integrate over time in photon-stream
        self._lfs_integral = light_field.sequence.integrate_around_arrival_peak(
            sequence=event.light_field.sequence,
            integration_radius=1
        )
        self.lixel_intensities = self._lfs_integral['integral']

        self.image_rays = ImageRays(event.light_field)

        self.psf = make_tomographic_system_matrix(
            supports=self.image_rays.support, 
            directions=self.image_rays.direction, 
            x_bin_edges=self.binning.x_img_bin_edges, 
            y_bin_edges=self.binning.y_img_bin_edges,
            z_bin_edges=self.binning.b_img_bin_edges,
        )

        self.rec_vol_I = np.zeros(self.binning.bin_num, dtype=np.float32)
        self.iteration = 0


        self.lixel_integral = self.psf.sum(axis=0).T # Total length of ray
        self.voxel_integral = self.psf.sum(axis=1) # Total distance of all rays in this voxel
        self.voxel_cross_psf = self.psf.dot(self.lixel_integral) # The sum of the length of all rays hiting this voxel weighted with the overlap of the ray and this voxel
        self.voxel_cross_psf = np.array(self.voxel_cross_psf).reshape((self.voxel_cross_psf.shape[0],))
        self.lixel_cross_psf = self.psf.T.dot(self.voxel_integral)
        self.lixel_cross_psf = np.array(self.lixel_cross_psf).reshape((self.lixel_cross_psf.shape[0],))

        self.voxel_within_field_of_view_mask = self.binning.voxels_within_field_of_view(
            radius=0.85
        )

        self._apply_frequency_filter = apply_frequency_filter
        if self._apply_frequency_filter:
            self._ramp_filter_kernel = ramp_kernel_in_frequency_space(
                x_num=self.binning.x_img_num,
                y_num=self.binning.y_img_num,
                z_num=self.binning.b_img_num,
            )


    def reconstructed_depth_of_field_intesities(self):
        return self.rec_vol_I.reshape(
            (
                self.binning.x_img_num, 
                self.binning.y_img_num, 
                self.binning.b_img_num
            ),
            order='C'
        )


    def one_more_iteration(self):
        rec_vol_I_n = update(
            vol_I=self.rec_vol_I.copy(),
            psf=self.psf,
            measured_lixel_I=self.lixel_intensities,
            voxel_cross_psf=self.voxel_cross_psf,
            lixel_cross_psf=self.lixel_cross_psf,
            in_fov=self.voxel_within_field_of_view_mask,
        )

        diff = np.abs(rec_vol_I_n - self.rec_vol_I).sum()

        print('Intensity difference to previous iteration '+str(diff))

        self.rec_vol_I = rec_vol_I_n.copy()

        if self._apply_frequency_filter:
            self.apply_high_pass_filter()

        self.iteration += 1


    def apply_high_pass_filter(self):
        rec_vol_I_3d = self.reconstructed_depth_of_field_intesities()
        rec_vol_I_3d = frequency_filter(
            hist=rec_vol_I_3d, 
            kernel=self._ramp_filter_kernel
        )
        self.rec_vol_I = rec_vol_I_3d.reshape(
            self.binning.bin_num,
            order='C'
        )


    def simulation_truth_depth_of_field_intesities(self):
        photon_bunches = self.event.simulation_truth.air_shower_photon_bunches
        true_emission_positions = emission_positions_of_photon_bunches(
            photon_bunches
        )

        true_emission_positions_image_domain = transform.xyz2cxcyb(
            true_emission_positions[:,0],
            true_emission_positions[:,1], 
            true_emission_positions[:,2],  
            self.binning.focal_length
        ).T
        tepid = true_emission_positions_image_domain

        # directions to positions on image screen
        tepid[:,0] = -self.binning.focal_length*np.tan(tepid[:,0])
        tepid[:,1] = -self.binning.focal_length*np.tan(tepid[:,1])

        hist = np.histogramdd(
            tepid, 
            bins=(
                self.binning.x_img_bin_edges,
                self.binning.y_img_bin_edges, 
                self.binning.b_img_bin_edges
            ),
            weights=photon_bunches.probability_to_reach_observation_level
        )

        return hist[0]


def update(vol_I, psf, measured_lixel_I,  voxel_cross_psf, lixel_cross_psf, in_fov):
    measured_I_voxel = psf.dot(measured_lixel_I)
    vox_non_zero = voxel_cross_psf > 0.0
    measured_I_voxel[vox_non_zero] /= voxel_cross_psf[vox_non_zero]

    proj_I_lixel = psf.T.dot(vol_I)
    lix_non_zero = lixel_cross_psf > 0.0
    proj_I_lixel[lix_non_zero] /= lixel_cross_psf[lix_non_zero]
    
    proj_I_voxel = psf.dot(proj_I_lixel)

    voxel_diffs = measured_I_voxel - proj_I_voxel

    vol_I[in_fov] += voxel_diffs[in_fov]

    vol_I[vol_I < 0.0] = 0.0
    return vol_I


@memory.cache
def make_tomographic_system_matrix(
    supports, 
    directions, 
    x_bin_edges, 
    y_bin_edges,
    z_bin_edges
):
    return ray_and_voxel.system_matrix(
        supports=supports,
        directions=directions, 
        x_bin_edges=x_bin_edges, 
        y_bin_edges=y_bin_edges,
        z_bin_edges=z_bin_edges,
    )