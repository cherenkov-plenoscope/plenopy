import numpy as np
import plenopy as pl
import pkg_resources
import tempfile
import os

light_field_geometry_path = pkg_resources.resource_filename(
    'plenopy',
    os.path.join('tests', 'resources', 'run.acp', 'input', 'plenoscope'))
light_field_geometry = pl.LightFieldGeometry(light_field_geometry_path)


def test_num_lixel_jobs():
    jobs = pl.tomography.system_matrix.make_jobs(
        light_field_geometry=light_field_geometry,
        sen_x_bin_edges=None,
        sen_y_bin_edges=None,
        sen_z_bin_edges=None,
        num_lixels_in_job=100,
        random_seed=0)

    actual_num_lixel = np.sum([len(job["cx_mean"]) for job in jobs])
    assert actual_num_lixel == light_field_geometry.number_lixel

    random_seeds = [job["random_seed"] for job in jobs]
    num_different_random_seeds = len(set(random_seeds))
    assert len(jobs) == num_different_random_seeds


def test_run_job():
    NUM_LIXELS_IN_JOB = 100
    focal_length = light_field_geometry.expected_focal_length_of_imaging_system
    fov_radius = np.deg2rad(2.5)

    binning = pl.tomography.image_domain.init_binning_for_depth_of_field(
        focal_length=focal_length,
        cx_min=-fov_radius,
        cx_max=fov_radius,
        number_cx_bins=32,
        cy_min=-fov_radius,
        cy_max=fov_radius,
        number_cy_bins=32,
        obj_min=2.5e3,
        obj_max=25e3,
        number_obj_bins=16)

    jobs = pl.tomography.system_matrix.make_jobs(
        light_field_geometry=light_field_geometry,
        sen_x_bin_edges=binning["sen_x_bin_edges"],
        sen_y_bin_edges=binning["sen_y_bin_edges"],
        sen_z_bin_edges=binning["sen_z_bin_edges"],
        num_lixels_in_job=NUM_LIXELS_IN_JOB,
        num_samples_per_lixel=3,
        random_seed=0)

    result = pl.tomography.system_matrix.run_job(jobs[0])
    assert len(result) == NUM_LIXELS_IN_JOB

    expected_sen_z_range = binning["sen_z_max"] - binning["sen_z_min"]
    for lix in range(len(result)):
        lixel_total_overlap = np.sum(result[lix]["lixel_voxel_overlaps"])
        np.testing.assert_approx_equal(
            actual=lixel_total_overlap,
            desired=expected_sen_z_range,
            significant=1.9)

    # In production, this for-loop can be processed in parallel
    results = []
    for job in jobs:
        result = pl.tomography.system_matrix.run_job(job)
        results.append(result)

    sparse_sys_mat = pl.tomography.system_matrix.reduce_results(
        results=results)

    mat = pl.tomography.system_matrix.__make_matrix(
        sparse_system_matrix=sparse_sys_mat,
        light_field_geometry=light_field_geometry,
        binning=binning)

    assert mat.shape[0] == binning["number_bins"]
    assert mat.shape[1] == light_field_geometry.number_lixel

    with tempfile.TemporaryDirectory(prefix="test_plenopy") as tmp:
        path = os.path.join(tmp, "sysmat")
        pl.tomography.system_matrix.write(sparse_sys_mat, path)
        sparse_sys_mat_back = pl.tomography.system_matrix.read(path)

    for key in sparse_sys_mat:
        assert key in sparse_sys_mat_back
