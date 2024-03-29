import numpy as np
import plenopy as pl
import os
import ray_voxel_overlap as rvo

"""
The events for testing are 'recorded' on a MAGIC 17m aperture diameter
Plenoscope. They are small in size and thus fine to test the interfaces of the
reconstruction but they can not give physical results, 17m is just too small of
a baseline.
"""


run_path = os.path.join(pl.testing.pkg_dir(), "tests", "resources", "run.acp")


def test_image_domain():
    run = pl.Run(run_path)
    event = run[0]

    trigger_image_geometry = pl.trigger.geometry.init_trigger_image_geometry(
        image_outer_radius_rad=np.deg2rad(1.5),
        pixel_spacing_rad=np.deg2rad(0.15),
        pixel_radius_rad=2.2 * np.deg2rad(0.15),
        max_number_nearest_lixel_in_pixel=7,
    )

    trigger_geometry = pl.trigger.geometry.init_trigger_geometry(
        light_field_geometry=run.light_field_geometry,
        trigger_image_geometry=trigger_image_geometry,
        object_distances=[10e3],
    )

    trigger_response, _ = pl.trigger.estimate.first_stage(
        raw_sensor_response=event.raw_sensor_response,
        light_field_geometry=run.light_field_geometry,
        trigger_geometry=trigger_geometry,
        integration_time_slices=10,
    )

    roi = pl.trigger.region_of_interest.from_trigger_response(
        trigger_response=trigger_response,
        trigger_geometry=trigger_geometry,
        time_slice_duration=event.raw_sensor_response["time_slice_duration"],
    )

    photons = pl.classify.RawPhotons.from_event(event)

    cherenkov_photons = pl.classify.cherenkov_photons_in_roi_in_image(
        photons=photons, roi=roi
    )

    binning = pl.Tomography.Image_Domain.Binning.init(
        focal_length=run.light_field_geometry.sensor_plane2imaging_system.expected_imaging_system_focal_length
    )

    (
        photon_arrival_times,
        photon_lixel_ids,
    ) = pl.light_field_sequence.photon_arrival_times_and_lixel_ids(
        event.raw_sensor_response
    )

    ssm_jobs = pl.Tomography.System_Matrix.make_jobs(
        light_field_geometry=run.light_field_geometry,
        sen_x_bin_edges=binning["sen_x_bin_edges"],
        sen_y_bin_edges=binning["sen_y_bin_edges"],
        sen_z_bin_edges=binning["sen_z_bin_edges"],
        num_lixels_in_job=1000,
        num_samples_per_lixel=3,
        random_seed=0,
    )
    ssm_results = []
    for job in ssm_jobs:
        result = pl.Tomography.System_Matrix.run_job(job)
        ssm_results.append(result)

    sparse_system_matrix = pl.Tomography.System_Matrix.reduce_results(
        results=ssm_results
    )

    psf = pl.Tomography.Image_Domain.Point_Spread_Function.init(
        sparse_system_matrix=sparse_system_matrix,
    )

    rec = pl.Tomography.Image_Domain.Reconstruction.init(
        light_field_geometry=event.light_field_geometry,
        photon_lixel_ids=photon_lixel_ids,
        binning=binning,
    )

    for i in range(10):
        rec = pl.Tomography.Image_Domain.Reconstruction.iterate(
            reconstruction=rec,
            point_spread_function=psf,
        )

    vol = rec["reconstructed_volume_intensity"]
    assert (vol < 0.0).sum() == 0
