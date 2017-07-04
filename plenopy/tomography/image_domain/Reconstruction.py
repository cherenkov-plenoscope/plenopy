import numpy as np
from ... import light_field
from ...image import ImageRays
from .. import ray_and_voxel
import matplotlib.pyplot as plt


class Reconstruction(object):

    def __init__(self, event, binning):
        self.event = event
        self.binning = binning

        # integrate over time in photon-stream
        self.lixel_intensities = lixel_intensities_at_arrival_peak(
            light_field_sequece=event.light_field.sequence,
            integration_radius=1
        )
        self._focal_length = self.event.light_field.expected_focal_length_of_imaging_system
        
        self.img_x_bin_edges = self._focal_length*np.tan(self.binning.cx_bin_edges)
        self.img_y_bin_edges = self._focal_length*np.tan(self.binning.cy_bin_edges)
        # 1/f = 1/g + 1/b
        self.img_b_bin_edges = 1.0/(
            1.0/self._focal_length + 
            1.0/self.binning.obj_bin_edges
        )

        self.image_rays = ImageRays(event.light_field)

        self.psf = ray_and_voxel.point_spread_function(
            supports=self.image_rays.support, 
            directions=self.image_rays.direction, 
            x_bin_edges=self.img_x_bin_edges, 
            y_bin_edges=self.img_y_bin_edges,
            z_bin_edges=self.img_b_bin_edges,
        )

        self.vol_I = np.zeros(self.binning.bins_num, dtype=np.float32)


    def save_vol_I_plot(self):

        image_stack = self.vol_I.reshape(
            shape=(
                self.binning.cx_num,
                self.binning.cy_num,
                self.binning.obj_num
            )
        )

        for obj_idx, obj_dist in enumerate(self.binning.obj_bin_centers):
            img = image_stack[:,:,obj_idx]

            ax.set_xlabel('x/m')
            ax.set_ylabel('y/m')



def lixel_intensities_at_arrival_peak(
    light_field_sequece, 
    integration_radius=1
):
    t_peak = light_field.sequence.time_slice_with_max_intensity(
        sequence=light_field_sequece
    )
    t_start = np.max([t_peak - integration_radius, 0])
    t_stop = np.min([t_peak + integration_radius + 1, light_field_sequece.shape[0]-1])

    return np.sum(light_field_sequece[t_start:t_stop,:], axis=0)


def update(vol_I, psf, measured_I):
    measured_I_of_voxel = psf.dot(measured_I)
    proj_I_of_voxel = psf.dot(psf.T.dot(vol_I))

    voxel_diffs = measured_I_of_voxel - proj_I_of_voxel
    voxel_diffs /= number_of_voxels_in_psf_per_voxel

    vol_I += voxel_diffs
    vol_I[vol_I < 0.0] = 0.0

    return vol_I
