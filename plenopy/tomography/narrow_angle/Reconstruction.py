import numpy as np
from joblib import Memory
import os
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from .. import light_field
from .Rays import Rays
from . import ray_and_voxel
from .simulation_truth import histogram_photon_bunches
from . import filtered_back_projection as fbp
from .Binning import Binning
from ..plot import slices
from ..plot import xyzI

cachedir = '/tmp/plenopy'
os.makedirs(cachedir, exist_ok=True)
memory = Memory(cachedir=cachedir, verbose=0)


class Reconstruction(object):

    def __init__(self, event, binning=None, use_low_pass_filter=False, rays_in_voxel_threshold=10.0):
        self.event = event

        if binning == None:
            self.binning = Binning(
                number_z_bins=64,
                number_xy_bins=64,
            )
        else:
           self.binning = binning

        self.use_low_pass_filter = use_low_pass_filter

        # integrate over time in photon-stream
        self._lfs_integral = light_field.sequence.integrate_around_arrival_peak(
            sequence=event.light_field.sequence,
            integration_radius=3
        )
        self.lixel_intensities = self._lfs_integral['integral']

        self.rays = Rays.from_light_field_geometry(event.light_field)

        self.psf = make_tomographic_system_matrix(
            supports=-self.rays.support, 
            directions=self.rays.direction, 
            x_bin_edges=self.binning.xy_bin_edges, 
            y_bin_edges=self.binning.xy_bin_edges,
            z_bin_edges=self.binning.z_bin_edges,
        )

        self.rec_vol_I = np.zeros(self.binning.number_bins, dtype=np.float32)
        self.iteration = 0


        self.lixel_integral = self.psf.sum(axis=0).T # Total length of ray
        self.lixel_integral = np.array(self.lixel_integral).reshape((self.lixel_integral.shape[0],))

        self.voxel_integral = self.psf.sum(axis=1) # Total distance of all rays in this voxel
        self.voxel_integral = np.array(self.voxel_integral).reshape((self.voxel_integral.shape[0],))

        self.voxel_cross_psf = self.psf.dot(self.lixel_integral) # The sum of the length of all rays hiting this voxel weighted with the overlap of the ray and this voxel
        self.voxel_cross_psf = np.array(self.voxel_cross_psf).reshape((self.voxel_cross_psf.shape[0],))
        
        self.lixel_cross_psf = self.psf.T.dot(self.voxel_integral)
        self.lixel_cross_psf = np.array(self.lixel_cross_psf).reshape((self.lixel_cross_psf.shape[0],))


        self.rays_in_voxel_threshold = rays_in_voxel_threshold
        self.expected_ray_voxel_overlap = 2.0*self.binning.voxel_z_radius
        self.valid_voxel = self.voxel_integral > self.rays_in_voxel_threshold*self.expected_ray_voxel_overlap
        self.valid_voxel = np.array(self.valid_voxel).reshape((self.valid_voxel.shape[0],))

        """
        vox_integral_3D = self.voxel_integral.reshape(               
            (
                self.binning.number_xy_bins, 
                self.binning.number_xy_bins, 
                self.binning.number_z_bins
            ),
            order='C'
        )
        self.max_ray_overlap = vox_integral_3D.mean(axis=0).mean(axis=0)
        self.max_ray_overlap /= self.max_ray_overlap.mean()
        """
        self.inverse_square_law = (self.binning.z_bin_centers)**(1/3)
        self.inverse_square_law /= self.inverse_square_law.mean()

        voxel_ids = np.arange(self.psf.shape[0])
        voxel_idxs_z = np.unravel_index(voxel_ids, dims=self.binning.dims, order='C')[2]
        self.obj_dist_regularization = self.inverse_square_law[voxel_idxs_z]


    def reconstructed_volume_intesities(self, filter_sigma=1.0):
        
        rec_vol_3D = self.rec_vol_I.reshape(
            (
                self.binning.number_xy_bins, 
                self.binning.number_xy_bins, 
                self.binning.number_z_bins
            ),
            order='C'
        )

        rec_vol_3D = np.fliplr(rec_vol_3D)
        rec_vol_3D = np.flipud(rec_vol_3D)

        return self.low_pass_filter(
            vol_I=rec_vol_3D,
            filter_sigma=filter_sigma
        )


    def one_more_iteration(self):
        rec_vol_I_n = update(
            vol_I=self.rec_vol_I.copy(),
            psf=self.psf,
            measured_lixel_I=self.lixel_intensities,
            voxel_cross_psf=self.voxel_cross_psf,
            lixel_cross_psf=self.lixel_cross_psf,
            obj_dist_regularization=self.obj_dist_regularization,
            valid_voxel=self.valid_voxel,
            ray_length=self.lixel_integral,
        )

        diff = np.abs(rec_vol_I_n - self.rec_vol_I).sum()

        print('Intensity difference to previous iteration '+str(diff))

        self.rec_vol_I = rec_vol_I_n.copy()

        if self.use_low_pass_filter:
            self.rec_vol_I = self.low_pass_filter(self.rec_vol_I)

        self.iteration += 1


    def low_pass_filter(self, vol_I, filter_sigma=0.5):
        return gaussian_filter(
            input=vol_I, 
            sigma=filter_sigma, 
            order=0,
            mode='constant',
            cval=0.0,
            truncate=2*filter_sigma
        )        


    def save_imgae_slice_stack(self, out_dir='./tomography', sqrt_intensity=False):
        os.makedirs(out_dir, exist_ok=True)

        intensity_volume_2 = None
        if hasattr(self.event, 'simulation_truth'):
            if hasattr(self.event.simulation_truth, 'air_shower_photon_bunches'):
                intensity_volume_2 = self.simulation_truth_volume_intesities()

        slices.save_slice_stack(
            intensity_volume=self.reconstructed_volume_intesities(),
            event_info_repr=self.event.__repr__(), 
            xy_extent=[
                self.binning.xy_bin_edges.min(),
                self.binning.xy_bin_edges.max(),
                self.binning.xy_bin_edges.min(),
                self.binning.xy_bin_edges.max(),
            ],
            z_bin_centers=self.binning.z_bin_centers,
            output_path=out_dir, 
            image_prefix='slice_',
            intensity_volume_2=intensity_volume_2,
            xlabel='x/m',
            ylabel='y/m',
            sqrt_intensity=sqrt_intensity,
        )


    def simulation_truth_volume_intesities(
        self, 
        restrict_to_instrument_acceptance=True
    ):  
        limited_aperture_radius = None
        limited_fov_radius = None
        if restrict_to_instrument_acceptance:
            limited_aperture_radius = 1.05*self.event.light_field.expected_aperture_radius_of_imaging_system
            limited_fov_radius = 1.05*self.event.light_field.cx_mean.max()

        return histogram_photon_bunches(
            photon_bunches=self.event.simulation_truth.air_shower_photon_bunches, 
            binning=self.binning, 
            observation_level=5e3,
            limited_aperture_radius=limited_aperture_radius,
            limited_fov_radius=limited_fov_radius,
        )


    def show_xyzI(
        self, 
        rec_threshold=0.0, 
        sim_threshold=0.0, 
        alpha_max=0.2, 
        color_steps=32, 
        ball_size=100.0
    ):
        rec_xyzIs = xyzI.hist3D_to_xyzI(
            hist=self.reconstructed_volume_intesities(), 
            binning=self.binning, 
            threshold=rec_threshold
        )

        sim_xyzIs = xyzI.hist3D_to_xyzI(
            hist=self.simulation_truth_volume_intesities(),
            binning=self.binning, 
            threshold=sim_threshold,
        )

        xyzI.plot_xyzI(
            xyzIs=rec_xyzIs, 
            xyzIs2=sim_xyzIs, 
            alpha_max=alpha_max, 
            steps=color_steps,
            ball_size=ball_size,
         )
        plt.show()


def update(
    vol_I, 
    psf, 
    measured_lixel_I,  
    voxel_cross_psf, 
    lixel_cross_psf, 
    obj_dist_regularization, 
    valid_voxel,
    ray_length
):  
    measured_photon_density_along_rays = measured_lixel_I/ray_length
    measured_ph_I_voxel = (psf.dot(measured_photon_density_along_rays))
    measured_ph_I_voxel /= obj_dist_regularization

    proj_ph_dist_int_lixel = psf.T.dot(vol_I)
    proj_ph_lixel = proj_ph_dist_int_lixel/ray_length
    proj_ph_dst_lixel = proj_ph_lixel/ray_length
    
    proj_ph_I_voxel = psf.dot(proj_ph_dst_lixel)

    voxel_diffs = measured_ph_I_voxel - proj_ph_I_voxel

    vol_I[valid_voxel] += voxel_diffs[valid_voxel]

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