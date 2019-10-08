import pytest
import numpy as np
import plenopy as pl
import pkg_resources


"""
The events for testing are 'recorded' on a MAGIC 17m aperture diameter
Plenoscope. They are small in size and thus fine to test the interfaces of the
reconstruction but they can not give physical results, 17m is just too small of
a baseline.
"""
run_path = pkg_resources.resource_filename(
    'plenopy',
    'tests/resources/run.acp')


def test_image_domain():
    run = pl.Run(run_path)
    event = run[0]

    trigger_preparation = pl.trigger.prepare_refocus_sum_trigger(
        light_field_geometry=run.light_field_geometry,
        object_distances=[10e3])

    trig = pl.trigger.apply_refocus_sum_trigger(
        event=event,
        trigger_preparation=trigger_preparation)

    roi = pl.trigger.region_of_interest_from_trigger_response(
        trigger_response=trig,
        time_slice_duration=event.raw_sensor_response.time_slice_duration,
        pixel_pos_cx=run.light_field_geometry.pixel_pos_cx,
        pixel_pos_cy=run.light_field_geometry.pixel_pos_cy,)

    photons = pl.classify.RawPhotons.from_event(event)

    cherenkov_photons = pl.classify.cherenkov_photons_in_roi_in_image(
        photons=photons,
        roi=roi)

    binning = pl.tomography.image_domain.init_binning_for_depth_of_field(
        focal_length=run.light_field_geometry.sensor_plane2imaging_system.expected_imaging_system_focal_length)

    photon_arrival_times, photon_lixel_ids = event.photon_arrival_times_and_lixel_ids()

    rec = pl.tomography.image_domain.init_reconstruction(
        light_field_geometry=event.light_field_geometry,
        photon_lixel_ids=photon_lixel_ids,
        binning=binning,)

    for i in range(10):
        rec = pl.tomography.image_domain.one_more_iteration(rec)

    vol = rec['reconstructed_volume_intensity']
    assert (vol < 0.0).sum() == 0
