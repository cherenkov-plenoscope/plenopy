import ray_voxel_overlap as rvo
import numpy as np
import scipy
import os
import array
import json


def make_jobs(
    light_field_geometry,
    sen_x_bin_edges,
    sen_y_bin_edges,
    sen_z_bin_edges,
    random_seed,
    num_lixels_in_job=1000,
    num_samples_per_lixel=100,
):
    """
    Returns a list of jobs to steer the compute of a 'system-matrix' for
    tomography.
    The jobs can then be computed in parallel. The results of the jobs
    need to be reduced later on to obtain the 'system-matrix'.
    The work is split along the beams observed by the photo-sensors in the
    instrument. So this can only profit from parallel computing when there are
    many photo-sensors.

    light_field_geometry : class
        The geometry of the light-field observed by our instrument
    sen_x_bin_edges : array 1d
        Bin-edges in (sen)sor-frame along x-axis, perpendicular to
        optical-axis.
    sen_x_bin_edges : array 1d
        Like sen_x_bin_edges, but on y-axis.
    sen_z_bin_edges : array 1d
        like sen_x_bin_edges but parallel to optical-axis.
    random_seed : int
        The random-seed used for the estimate of the 'system-matrix'.
    num_lixels_in_job : int
        This many beams/lixels will be worked on by a single job.
    num_samples_per_lixel : int
        This many rays are casted to approximate a beam observed by a
        photo-sensor. The spread of the rays is based on the spread stored in
        light_field_geometry.
    """
    focal_length = light_field_geometry.expected_focal_length_of_imaging_system

    jobs = []
    idx_start = 0
    while idx_start < light_field_geometry.number_lixel:
        idx_stop = idx_start + num_lixels_in_job
        if idx_stop > light_field_geometry.number_lixel:
            idx_stop = light_field_geometry.number_lixel
        job = {}
        job["random_seed"] = int(random_seed + len(jobs))
        job["num_samples_per_lixel"] = int(num_samples_per_lixel)
        job["sen_x_bin_edges"] = sen_x_bin_edges
        job["sen_y_bin_edges"] = sen_y_bin_edges
        job["sen_z_bin_edges"] = sen_z_bin_edges
        job["focal_length"] = focal_length
        job["cx_mean"] = light_field_geometry.cx_mean[idx_start:idx_stop]
        job["cx_std"] = light_field_geometry.cx_std[idx_start:idx_stop]
        job["cy_mean"] = light_field_geometry.cy_mean[idx_start:idx_stop]
        job["cy_std"] = light_field_geometry.cy_std[idx_start:idx_stop]
        job["x_mean"] = light_field_geometry.x_mean[idx_start:idx_stop]
        job["x_std"] = light_field_geometry.x_std[idx_start:idx_stop]
        job["y_mean"] = light_field_geometry.y_mean[idx_start:idx_stop]
        job["y_std"] = light_field_geometry.y_std[idx_start:idx_stop]
        job["lixel_idx"] = np.arange(idx_start, idx_stop)
        job["number_lixel"] = light_field_geometry.number_lixel
        job["number_voxel"] = (
            (len(sen_x_bin_edges) - 1)
            * (len(sen_y_bin_edges) - 1)
            * (len(sen_z_bin_edges) - 1)
        )
        jobs.append(job)
        idx_start = idx_stop
    return jobs


def run_job(job):
    """
    Computes and returns the result of a single job for the
    estimate of a system-matrix in tomography.

    job : dict
        The confi. for the computation of a part of the system-matrix.
    """
    prng = np.random.Generator(np.random.MT19937(seed=job["random_seed"]))

    results = []
    num_lixel = job["cx_mean"].shape[0]
    for lix in range(num_lixel):
        (image_ray_supports, image_ray_directions) = _make_image_ray_bundle(
            prng=prng,
            cx_mean=job["cx_mean"][lix],
            cx_std=job["cx_std"][lix],
            cy_mean=job["cy_mean"][lix],
            cy_std=job["cy_std"][lix],
            x_mean=job["x_mean"][lix],
            x_std=job["x_std"][lix],
            y_mean=job["y_mean"][lix],
            y_std=job["y_std"][lix],
            focal_length=job["focal_length"],
            num_samples=job["num_samples_per_lixel"],
        )

        (
            lixel_voxel_overlaps,
            voxel_indicies,
        ) = rvo.estimate_overlap_of_ray_bundle_with_voxels(
            supports=image_ray_supports,
            directions=image_ray_directions,
            x_bin_edges=job["sen_x_bin_edges"],
            y_bin_edges=job["sen_y_bin_edges"],
            z_bin_edges=job["sen_z_bin_edges"],
            order="C",
        )

        result = {}
        result["lixel_idx"] = np.uint32(job["lixel_idx"][lix])
        result["voxel_indicies"] = voxel_indicies.astype(np.uint32)
        result["lixel_voxel_overlaps"] = lixel_voxel_overlaps.astype(
            np.float32
        )
        result["number_lixel"] = job["number_lixel"]
        result["number_voxel"] = job["number_voxel"]
        results.append(result)
    return results


def _make_image_ray_bundle(
    prng,
    cx_mean,
    cx_std,
    cy_mean,
    cy_std,
    x_mean,
    x_std,
    y_mean,
    y_std,
    focal_length,
    num_samples,
):
    """
    Returns the supports and directions of multiple rays which all approximate
    the geometry of a beam observed by a single photo-sensor.
    """
    cx = prng.normal(loc=cx_mean, scale=cx_std, size=num_samples)
    cy = prng.normal(loc=cy_mean, scale=cy_std, size=num_samples)
    x = prng.normal(loc=x_mean, scale=x_std / 2, size=num_samples)
    y = prng.normal(loc=y_mean, scale=y_std / 2, size=num_samples)

    sensor_plane_intersections = np.array(
        [
            -focal_length * np.tan(cx),
            -focal_length * np.tan(cy),
            focal_length * np.ones(num_samples),
        ]
    ).T
    image_ray_supports = np.array([x, y, np.zeros(num_samples)]).T

    image_ray_directions = sensor_plane_intersections - image_ray_supports
    no = np.linalg.norm(image_ray_directions, axis=1)
    image_ray_directions[:, 0] /= no
    image_ray_directions[:, 1] /= no
    image_ray_directions[:, 2] /= no

    return image_ray_supports, image_ray_directions


SYSTEM_MATRIX_DTYPE = {
    "lixel_voxel_overlaps": "float32",
    "lixel_indicies": "uint32",
    "voxel_indicies": "uint32",
}


def reduce_results(results):
    """
    Returns a dict representing the sparse system-mstrix to be used in
    tomography.

    results : list of results.
        Each result in the list is a result of a 'job'.
    """
    lixel_voxel_overlaps = array.array("f")
    voxel_indicies = array.array("L")
    lixel_indicies = array.array("L")

    sys_mat = {}
    sys_mat["number_lixel"] = results[0][0]["number_lixel"]
    sys_mat["number_voxel"] = results[0][0]["number_voxel"]

    for result in results:
        assert sys_mat["number_lixel"] == result[0]["number_lixel"]
        assert sys_mat["number_voxel"] == result[0]["number_voxel"]

        for lix in range(len(result)):
            lixel_result = result[lix]
            num_overlaps = lixel_result["voxel_indicies"].shape[0]
            __lixel_indicies = lixel_result["lixel_idx"] * np.ones(
                num_overlaps, dtype=np.uint32
            )
            lixel_voxel_overlaps.extend(lixel_result["lixel_voxel_overlaps"])
            voxel_indicies.extend(lixel_result["voxel_indicies"])
            lixel_indicies.extend(__lixel_indicies)

    sys_mat["lixel_indicies"] = np.array(lixel_indicies, dtype=np.uint32)
    sys_mat["voxel_indicies"] = np.array(voxel_indicies, dtype=np.uint32)
    sys_mat["lixel_voxel_overlaps"] = np.array(
        lixel_voxel_overlaps, dtype=np.float32
    )
    return sys_mat


def write(sparse_system_matrix, path):
    ssm = sparse_system_matrix
    os.makedirs(path)
    for key in SYSTEM_MATRIX_DTYPE:
        key_dtype_str = SYSTEM_MATRIX_DTYPE[key]
        with open(os.path.join(path, key + "." + key_dtype_str), "wb") as f:
            f.write(ssm[key].tobytes())
    with open(os.path.join(path, "number.json"), "wb") as f:
        f.write(
            json.dumps(
                {
                    "number_lixel": ssm["number_lixel"],
                    "number_voxel": ssm["number_voxel"],
                }
            )
        )


def read(path):
    ssm = {}
    with open(os.path.join(path, "number.json"), "rb") as f:
        ssm = json.loads(f.read())
    for key in SYSTEM_MATRIX_DTYPE:
        key_dtype_str = SYSTEM_MATRIX_DTYPE[key]
        with open(os.path.join(path, key + "." + key_dtype_str), "rb") as f:
            ssm[key] = np.frombuffer(
                f.read(), dtype=np.dtype(SYSTEM_MATRIX_DTYPE[key])
            )
    return ssm


def to_numpy_csr_matrix(sparse_system_matrix):
    """
    Returns a numpy CSR-matrix for efficient application of the system-matrix
    in iterative tomography.

    sparse_system_matrix : dict
        Represents the system-matrix with explicit arrays of indices for
        beams and volume-cells in a dict.
    """
    ssm = sparse_system_matrix
    s = scipy.sparse.coo_matrix(
        (
            ssm["lixel_voxel_overlaps"],
            (ssm["voxel_indicies"], ssm["lixel_indicies"]),
        ),
        shape=(ssm["number_voxel"], ssm["number_lixel"]),
        dtype=np.float32,
    )

    return s.tocsr()
