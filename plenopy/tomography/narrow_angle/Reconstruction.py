"""
This 'narrow angle tomography' or '3D deconvolution' is inspired by:
@article{levoy2006light,
    title={Light field microscopy},
    author={Levoy, Marc and Ng, Ren and Adams, Andrew and Footer, Matthew and Horowitz, Mark},
    journal={ACM Transactions on Graphics (TOG)},
    volume={25},
    number={3},
    pages={924--934},
    year={2006},
    publisher={ACM}
}
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from ... import light_field
from ..Rays import Rays
from ..simulation_truth import histogram_photon_bunches
from ..Binning import Binning
from ...plot import slices
from ...plot import xyzI
from .deconvolution import update
from .deconvolution import make_cached_tomographic_system_matrix
from ...photon_stream import cython_reader as phscr


class Reconstruction(object):
    def __init__(
        self,
        event,
        binning=None,
        lixel_intensities=None,
        lixel_supports=None,
        lixel_directions=None,
        use_low_pass_filter=False,
        rays_in_voxel_threshold=3.0
    ):
        self.event = event
        f = event.light_field_geometry.expected_focal_length_of_imaging_system
        D = 2.0*event.light_field_geometry.expected_aperture_radius_of_imaging_system

        if binning is None:
            self.binning = Binning(
                number_z_bins=96,
                number_xy_bins=96,
                z_min=15.0*f,
                z_max=165.0*f,
                xy_diameter=4.5*D,
            )
        else:
            self.binning = binning

        self.use_low_pass_filter = use_low_pass_filter

        if lixel_intensities is None:
            # integrate over time in photon-stream
            raw = event.raw_sensor_response
            lixel_sequence = np.zeros(
                shape=(
                    raw.number_time_slices,
                    raw.number_lixel),
                dtype=np.uint16)

            phscr.stream2sequence(
                photon_stream=raw.photon_stream,
                time_slice_duration=raw.time_slice_duration,
                NEXT_READOUT_CHANNEL_MARKER=raw.NEXT_READOUT_CHANNEL_MARKER,
                sequence=lixel_sequence,
                time_delay_mean=event.light_field_geometry.time_delay_mean)


            self._lfs_integral = light_field.sequence.integrate_around_arrival_peak(
                sequence=lixel_sequence,
                integration_radius=3
            )
            self.lixel_intensities = self._lfs_integral['integral']
        else:
            self.lixel_intensities = lixel_intensities

        if lixel_supports is None or lixel_directions is None:
            rays = Rays.from_light_field_geometry(event.light_field_geometry)
            self.lixel_supports = rays.support
            self.direction = rays.direction
        else:
            self.lixel_supports = lixel_supports
            self.direction = lixel_directions

        self.system_matrix = make_cached_tomographic_system_matrix(
            supports=-self.lixel_supports,
            directions=self.direction,
            x_bin_edges=self.binning.xy_bin_edges,
            y_bin_edges=self.binning.xy_bin_edges,
            z_bin_edges=self.binning.z_bin_edges,
        )

        self.rec_vol_I = np.zeros(self.binning.number_bins, dtype=np.float32)
        self.iteration = 0

        # Total length of ray
        self.lixel_integral = self.system_matrix.sum(axis=0).T
        self.lixel_integral = np.array(self.lixel_integral).reshape(
            (self.lixel_integral.shape[0],)
        )

        # Total distance of all rays in this voxel
        self.voxel_integral = self.system_matrix.sum(axis=1)
        self.voxel_integral = np.array(self.voxel_integral).reshape(
            (self.voxel_integral.shape[0],)
        )

        self.rays_in_voxel_threshold = rays_in_voxel_threshold
        self.expected_ray_voxel_overlap = 2.0*self.binning.voxel_z_radius
        self.valid_voxel = (
            self.voxel_integral >
            self.rays_in_voxel_threshold*self.expected_ray_voxel_overlap
        )
        self.valid_voxel = np.array(self.valid_voxel).reshape(
            (self.valid_voxel.shape[0],)
        )

    def one_more_iteration(self):
        rec_vol_I_n = update(
            vol_I=self.rec_vol_I.copy(),
            system_matrix=self.system_matrix,
            measured_lixel_I=self.lixel_intensities,
            valid_voxel=self.valid_voxel,
            ray_length=self.lixel_integral,
        )
        diff = np.abs(rec_vol_I_n - self.rec_vol_I).sum()
        print('Intensity difference to previous iteration '+str(diff))
        self.rec_vol_I = rec_vol_I_n.copy()

        if self.use_low_pass_filter:
            self.rec_vol_I = self.low_pass_filter(self.rec_vol_I)

        self.iteration += 1

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

    def low_pass_filter(self, vol_I, filter_sigma=0.5):
        return gaussian_filter(
            input=vol_I,
            sigma=filter_sigma,
            order=0,
            mode='constant',
            cval=0.0,
            truncate=2*filter_sigma
        )

    def save_imgae_slice_stack(
        self,
        out_dir='./tomography',
        sqrt_intensity=False
    ):
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

    def _reconstructed_intesity_vs_obj_dist(self):
        rec_vol_I = self.reconstructed_volume_intesities()
        return rec_vol_I.sum(axis=0).sum(axis=0)

    def _simulated_intesity_vs_obj_dist(self):
        sim_vol_I = self.simulation_truth_volume_intesities()
        return sim_vol_I.sum(axis=0).sum(axis=0)

    def _show_rec_vs_sim_intesity_vs_obj_dist(self):
        rec_vol_I = self._reconstructed_intesity_vs_obj_dist()
        rec_vol_I /= rec_vol_I.mean()
        sim_vol_I = self._simulated_intesity_vs_obj_dist()
        sim_vol_I /= sim_vol_I.mean()
        obj_dist = self.binning.z_bin_centers
        plt.plot(obj_dist, sim_vol_I, 'r', label='simulation truth')
        plt.plot(obj_dist, rec_vol_I, 'b', label='reconstruction')
        plt.xlabel('object distance/m')
        plt.ylabel('photon intensity normalized area under curve/1')
        plt.legend(
            bbox_to_anchor=(0., 1.02, 1., .102),
            loc=3,
            ncol=2,
            mode="expand",
            borderaxespad=0.
        )
        plt.show()
