import numpy as np
from ... import light_field
from ...image import ImageRays
from .. import ray_and_voxel
import matplotlib.pyplot as plt
from .DepthOfFieldBinning import DepthOfFieldBinning
from .transform import object_distance_2_image_distance as g2b
from .transform import image_distance_2_object_distance as b2g

class Reconstruction(object):

    def __init__(self, event, dof_binning=None):
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

        self.psf = ray_and_voxel.point_spread_function(
            supports=self.image_rays.support, 
            directions=self.image_rays.direction, 
            x_bin_edges=self.binning.x_img_bin_edges, 
            y_bin_edges=self.binning.y_img_bin_edges,
            z_bin_edges=self.binning.b_img_bin_edges,
        )

        self.vol_I = np.zeros(self.binning.bin_num, dtype=np.float32)


    def save_vol_I_plot(self, path='.'):
        pass


def update(vol_I, psf, measured_I):
    measured_I_of_voxel = psf.dot(measured_I)
    proj_I_of_voxel = psf.dot(psf.T.dot(vol_I))

    voxel_diffs = measured_I_of_voxel - proj_I_of_voxel
    voxel_diffs /= number_of_voxels_in_psf_per_voxel

    vol_I += voxel_diffs
    vol_I[vol_I < 0.0] = 0.0

    return vol_I
