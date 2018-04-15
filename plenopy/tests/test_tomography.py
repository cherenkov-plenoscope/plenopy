import pytest
import numpy as np
import plenopy as pl
import pkg_resources


def test_binning_flat_voxel_center_positions():
    binning = pl.tomography.Binning(
        z_min=0.0,
        z_max=1e3,
        number_z_bins=101,
        xy_diameter=18.0,
        number_xy_bins=33)

    xyz_flat = binning.flat_xyz_voxel_positions()

    i = 0
    for x in range(binning.number_xy_bins):
        for y in range(binning.number_xy_bins):
            for z in range(binning.number_z_bins):
                assert xyz_flat[i, 0] == binning.xy_bin_centers[x]
                assert xyz_flat[i, 1] == binning.xy_bin_centers[y]
                assert xyz_flat[i, 2] == binning.z_bin_centers[z]
                i += 1

"""
The events for testing are 'recorded' on a MAGIC 17m aperture diameter
Plenoscope. They are small in size and thus fine to test the interfaces of the
reconstruction but they can not give physical results, 17m is just too small of
a baseline.
"""
run_path = pkg_resources.resource_filename(
    'plenopy',
    'tests/resources/run.acp'
)


def test_narrow_angle_deconvolution():
    run = pl.Run(run_path)
    event = run[0]

    rec = pl.tomography.narrow_angle.Reconstruction(
        event=event,
        binning=pl.tomography.Binning(number_z_bins=32, number_xy_bins=16)
    )

    for i in range(10):
        rec.one_more_iteration()

    vol = rec.reconstructed_volume_intesities()
    assert vol.shape[0] == 16
    assert vol.shape[1] == 16
    assert vol.shape[2] == 32
    assert (vol < 0.0).sum() == 0


def test_narrow_filtered_back_projection():
    run = pl.Run(run_path)
    event = run[0]

    raw = event.raw_sensor_response
    lixel_sequence = np.zeros(
        shape=(
            raw.number_time_slices,
            raw.number_lixel),
        dtype=np.uint16)

    pl.photon_stream.cython_reader.stream2sequence(
        photon_stream=raw.photon_stream,
        time_slice_duration=raw.time_slice_duration,
        NEXT_READOUT_CHANNEL_MARKER=raw.NEXT_READOUT_CHANNEL_MARKER,
        sequence=lixel_sequence,
        time_delay_mean=event.light_field_geometry.time_delay_mean)

    lixel_intensities = pl.sequence.integrate_around_arrival_peak(
        sequence=lixel_sequence,
        integration_radius=5
    )['integral']

    rec = pl.tomography.filtered_back_projection.Reconstruction(
        rays=pl.tomography.Rays.from_light_field_geometry(event.light_field_geometry),
        intensities=lixel_intensities,
        binning=pl.tomography.Binning(number_z_bins=32, number_xy_bins=16))

    vol = rec.intensity_volume
    assert vol.shape[0] == 16
    assert vol.shape[1] == 16
    assert vol.shape[2] == 32
    assert (vol < 0.0).sum() == 0


def test_image_domain():
    run = pl.Run(run_path)
    event = run[0]

    rec = pl.tomography.image_domain.Reconstruction(
        event=event,
    )

    for i in range(10):
        rec.one_more_iteration()

    vol = rec.reconstructed_depth_of_field_intesities()
    assert (vol < 0.0).sum() == 0
