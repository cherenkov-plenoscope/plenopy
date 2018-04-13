import numpy as np


def estimate_3d_plane_model(xyz_point_cloud):
    X = xyz_point_cloud
    C = np.average(X, axis=0)
    CX = X - C
    U, S, V = np.linalg.svd(CX)
    N = V[-1]  # normal of plane
    d = -N.dot(C)  # distance to origin
    if d < 0:
        d *= -1
        N *= -1
    return np.array([N[0], N[1], N[2], d])


def distance_to_plane(plane_model, xyz_point_cloud):
    xyz1 = np.ones((xyz_point_cloud.shape[0], 4))
    xyz1[:, :3] = xyz_point_cloud
    return np.abs(plane_model.dot(xyz1.T))


def draw_sample(xyz_point_cloud, sample_size):
    sample = np.random.choice(
        np.arange(xyz_point_cloud.shape[0]),
        size=sample_size,
        replace=False)
    return xyz_point_cloud[sample]


def fit(
    xyz_point_cloud,
    max_number_itarations,
    min_number_points_for_plane_fit,
    max_orthogonal_distance_of_inlier,
):
    best_number_inliers = 0
    best_model = None
    best_inliers = []

    for i in range(max_number_itarations):
        xyz_sample = draw_sample(
            xyz_point_cloud=xyz_point_cloud,
            sample_size=min_number_points_for_plane_fit)

        model_based_on_sample = estimate_3d_plane_model(xyz_sample)

        distances_to_plane = distance_to_plane(
            plane_model=model_based_on_sample,
            xyz_point_cloud=xyz_point_cloud)

        inlier = distances_to_plane <= max_orthogonal_distance_of_inlier
        number_inlier = np.sum(inlier)

        if number_inlier > best_number_inliers:
            best_number_inliers = number_inlier
            best_model = model_based_on_sample
            best_inliers = inlier

    return best_model, best_inliers
