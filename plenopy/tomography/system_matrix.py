import ray_voxel_overlap as rvo
import numpy as np
import scipy
import os
import array


def __make_jobs(
    light_field_geometry,
    sen_x_bin_edges,
    sen_y_bin_edges,
    sen_z_bin_edges,
    random_seed,
    num_lixels_in_job=1000,
    num_samples_per_lixel=100,
):
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
        job["cx_mean"] = light_field_geometry.cx_mean[idx_start: idx_stop]
        job["cx_std"] = light_field_geometry.cx_std[idx_start: idx_stop]
        job["cy_mean"] = light_field_geometry.cy_mean[idx_start: idx_stop]
        job["cy_std"] = light_field_geometry.cy_std[idx_start: idx_stop]
        job["x_mean"] = light_field_geometry.x_mean[idx_start: idx_stop]
        job["x_std"] = light_field_geometry.x_std[idx_start: idx_stop]
        job["y_mean"] = light_field_geometry.y_mean[idx_start: idx_stop]
        job["y_std"] = light_field_geometry.y_std[idx_start: idx_stop]
        job["lixel_idx"] = np.arange(idx_start, idx_stop)
        jobs.append(job)
        idx_start = idx_stop
    return jobs


def __run_job(job):
    np.random.seed(job["random_seed"])

    results = []
    num_lixel = job["cx_mean"].shape[0]
    for lix in range(num_lixel):
        (image_ray_supports,
            image_ray_directions) = __make_image_ray_bundle(
            cx_mean=job["cx_mean"][lix],
            cx_std=job["cx_std"][lix],
            cy_mean=job["cy_mean"][lix],
            cy_std=job["cy_std"][lix],
            x_mean=job["x_mean"][lix],
            x_std=job["x_std"][lix],
            y_mean=job["y_mean"][lix],
            y_std=job["y_std"][lix],
            focal_length=job["focal_length"],
            num_samples=job["num_samples_per_lixel"])

        (lixel_voxel_overlaps,
            voxel_indicies) = rvo.estimate_overlap_of_ray_bundle_with_voxels(
            supports=image_ray_supports,
            directions=image_ray_directions,
            x_bin_edges=job["sen_x_bin_edges"],
            y_bin_edges=job["sen_y_bin_edges"],
            z_bin_edges=job["sen_z_bin_edges"],
            order="C")

        result = {}
        result["lixel_idx"] = np.uint32(job["lixel_idx"][lix])
        result["voxel_indicies"] = voxel_indicies.astype(np.uint32)
        result["lixel_voxel_overlaps"] = lixel_voxel_overlaps.astype(
            np.float32)
        results.append(result)
    return results


def __make_image_ray_bundle(
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
    cx = np.random.normal(loc=cx_mean, scale=cx_std, size=num_samples)
    cy = np.random.normal(loc=cy_mean, scale=cy_std, size=num_samples)
    x = np.random.normal(loc=x_mean, scale=x_std/2, size=num_samples)
    y = np.random.normal(loc=y_mean, scale=y_std/2, size=num_samples)

    sensor_plane_intersections = np.array([
        -focal_length*np.tan(cx),
        -focal_length*np.tan(cy),
        focal_length*np.ones(num_samples)]).T
    image_ray_supports = np.array([x, y, np.zeros(num_samples)]).T

    image_ray_directions = sensor_plane_intersections - image_ray_supports
    no = np.linalg.norm(image_ray_directions, axis=1)
    image_ray_directions[:, 0] /= no
    image_ray_directions[:, 1] /= no
    image_ray_directions[:, 2] /= no

    return image_ray_supports, image_ray_directions


__KEY_DTYPE = {
    "lixel_voxel_overlaps": "float32",
    "lixel_indicies": "uint32",
    "voxel_indicies": "uint32"}


def __reduce_results(results):
    lixel_voxel_overlaps = array.array('f')
    voxel_indicies = array.array('L')
    lixel_indicies = array.array('L')

    for result in results:
        for lix in range(len(result)):
            lixel_result = result[lix]
            num_overlaps = lixel_result["voxel_indicies"].shape[0]
            __lixel_indicies = lixel_result["lixel_idx"]*np.ones(
                num_overlaps,
                dtype=np.uint32)
            lixel_voxel_overlaps.extend(lixel_result["lixel_voxel_overlaps"])
            voxel_indicies.extend(lixel_result["voxel_indicies"])
            lixel_indicies.extend(__lixel_indicies)

    sys_mat = {}
    sys_mat["lixel_indicies"] = np.array(lixel_indicies, dtype=np.uint32)
    sys_mat["voxel_indicies"] = np.array(voxel_indicies, dtype=np.uint32)
    sys_mat["lixel_voxel_overlaps"] = np.array(
            lixel_voxel_overlaps,
            dtype=np.float32)
    return sys_mat


def write_sparse(sparse_system_matrix, path):
    os.makedirs(path)
    for key in __KEY_DTYPE:
        with open(os.path.join(path, key + "." + __KEY_DTYPE[key]), "wb") as f:
            f.write(sparse_system_matrix[key].tobytes())


def read_sparse(path):
    sparse_sys_mat = {}
    for key in __KEY_DTYPE:
        with open(os.path.join(path, key + "." + __KEY_DTYPE[key]), "rb") as f:
            sparse_sys_mat[key] = np.frombuffer(
                f.read(),
                dtype=np.dtype(__KEY_DTYPE[key]))
    return sparse_sys_mat


def __make_matrix(sparse_system_matrix, light_field_geometry, binning):
    sm = sparse_system_matrix
    s = scipy.sparse.coo_matrix(
        (
            sm["lixel_voxel_overlaps"],
            (
                sm["voxel_indicies"],
                sm["lixel_indicies"]
            )
        ),
        shape=(binning["number_bins"], light_field_geometry.number_lixel),
        dtype=np.float32)

    return s.tocsr()
