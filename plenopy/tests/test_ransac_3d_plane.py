import numpy as np
import plenopy as pl


def test_estimate_3d_plane_model_1():
    xyz = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])

    m = pl.tools.ransac_3d_plane.estimate_3d_plane_model(xyz_point_cloud=xyz)

    nx, ny, nz, d = m
    assert nx == 0
    assert ny == 0
    assert nz == 1
    assert d == 0


def test_estimate_3d_plane_model_2():
    xyz = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    m = pl.tools.ransac_3d_plane.estimate_3d_plane_model(xyz_point_cloud=xyz)

    nx, ny, nz, d = m
    n = np.array([1, 1, 1])
    len_n = np.linalg.norm(n)
    assert np.isclose(nx, -1 / len_n)
    assert np.isclose(ny, -1 / len_n)
    assert np.isclose(nz, -1 / len_n)
    assert np.isclose(d, 1 / len_n)


def test_estimate_3d_plane_model_3():
    prng = np.random.Generator(np.random.MT19937(seed=0))

    z0 = 3
    xx, yy = np.mgrid[-10:10, -10:10]
    x = xx.flatten()
    y = yy.flatten()
    z = prng.normal(loc=z0, size=400)
    xyz = np.c_[x, y, z]

    m = pl.tools.ransac_3d_plane.estimate_3d_plane_model(xyz_point_cloud=xyz)

    nx, ny, nz, d = m
    assert np.isclose(nx, 0.0, atol=3e-2)
    assert np.isclose(ny, 0.0, atol=3e-2)
    assert np.isclose(nz, -1.0, atol=5e-2)
    assert np.isclose(d, z0, atol=6e-2)


def test_distance_to_plane():
    m = np.array([0, 0, 1, 0])
    d2p = pl.tools.ransac_3d_plane.distance_to_plane

    assert d2p(m, np.array([[0, 0, 0], [0, 0, 1]]))[0] == 0
    assert d2p(m, np.array([[0, 0, 0], [0, 0, 1]]))[1] == 1

    for z in range(100):
        assert d2p(m, np.array([0, 0, z]))[0] == z

    for x in np.linspace(-1, 1, 100):
        for y in np.linspace(-1, 1, 100):
            assert d2p(m, np.array([x, y, 1]))[0] == 1


def test_draw_sample():
    prng = np.random.Generator(np.random.MT19937(seed=0))

    for i in np.arange(1, 1000, 10):
        s = pl.tools.ransac_3d_plane.draw_sample(
            xyz_point_cloud=np.arange(1000), sample_size=i, prng=prng
        )
        assert len(set(s)) == i

    s = pl.tools.ransac_3d_plane.draw_sample(
        xyz_point_cloud=np.arange(100), sample_size=100, prng=prng
    )
    assert len(set(s)) == 100


def test_ransac_full():
    prng = np.random.Generator(np.random.MT19937(seed=0))

    for nx in np.linspace(-1, 1, 4):
        for ny in np.linspace(-1, 1, 4):
            for nz in np.linspace(-1, 1, 4):
                for d in np.linspace(1, 2, 4):
                    n = np.array([nx, ny, nz])
                    n = n / np.linalg.norm(n)

                    if d < 0:
                        d *= -1
                        n *= -1

                    # true plane points, signal

                    n_points_plane = 70
                    x = prng.uniform(low=-10, high=10, size=n_points_plane)
                    y = prng.uniform(low=-10, high=10, size=n_points_plane)
                    z = -d / n[2] - (n[0] * x) / n[2] - (n[1] * y) / n[2]

                    valid = (z <= 10) * (z >= -10)
                    x = x[valid]
                    y = y[valid]
                    z = z[valid]
                    n_points_plane = z.shape[0]

                    # noise points
                    signal_over_noise = 70 / 30
                    n_points_noise = int(
                        np.round(signal_over_noise * n_points_plane)
                    )
                    noise_x = prng.uniform(
                        low=-10, high=10, size=n_points_noise
                    )
                    noise_y = prng.uniform(
                        low=-10, high=10, size=n_points_noise
                    )
                    noise_z = prng.uniform(
                        low=-10, high=10, size=n_points_noise
                    )

                    xyz = np.c_[
                        np.hstack([x, noise_x]),
                        np.hstack([y, noise_y]),
                        np.hstack([z, noise_z]),
                    ]
                    assert xyz.shape[0] == n_points_noise + n_points_plane
                    assert xyz.shape[1] == 3

                    model, inliers = pl.tools.ransac_3d_plane.fit(
                        xyz_point_cloud=xyz,
                        max_number_itarations=200,
                        min_number_points_for_plane_fit=3,
                        max_orthogonal_distance_of_inlier=0.3,
                        prng=prng,
                    )

                    (nx_r, ny_r, nz_r, d_r) = model

                    n_r = np.array([nx_r, ny_r, nz_r])

                    assert np.isclose(d_r, d, atol=2e-1)
                    assert np.isclose(n_r[0], n[0], atol=5e-2)
                    assert np.isclose(n_r[1], n[1], atol=5e-2)
                    assert np.isclose(n_r[2], n[2], atol=5e-2)
                    assert (
                        np.sum(inliers[0:n_points_plane])
                        >= 0.9 * n_points_plane
                    )
                    assert (
                        np.sum(inliers[n_points_plane:]) < 0.2 * n_points_noise
                    )
