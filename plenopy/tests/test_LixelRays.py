import pytest
import numpy as np
import plenopy as pl
import pkg_resources


def test_init():
    prng = np.random.Generator(np.random.MT19937(seed=0))

    N = 100
    support = np.zeros(shape=(N, 3))
    support[:, 0] = prng.uniform(size=N)
    support[:, 1] = prng.uniform(size=N)
    support[:, 2] = np.zeros(N)

    direction = np.zeros(shape=(N, 3))
    cx = 0.1 * prng.uniform(size=N)
    cy = 0.1 * prng.uniform(size=N)
    direction[:, 0] = cx
    direction[:, 1] = cy
    direction[:, 2] = np.sqrt(1.0 - cx ** 2 - cy ** 2)

    rays = pl.Tomography.Rays(x=support[:, 0], y=support[:, 1], cx=cx, cy=cy)

    assert np.allclose(rays.support, support)
    assert np.allclose(rays.direction, direction)


def test_ray_xy_intersection():

    support = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 2.0, 100.0])
    direction /= np.linalg.norm(direction)

    rays = pl.Tomography.Rays(
        x=np.array([support[0]]),
        y=np.array([support[1]]),
        cx=np.array([direction[0]]),
        cy=np.array([direction[1]]),
    )

    xy = rays.xy_intersections_in_object_distance(1000.0)
    assert abs(xy[0, 0] - (1000 / direction[2] * -direction[0])) < 1e-9
    assert abs(xy[0, 1] - (1000 / direction[2] * -direction[1])) < 1e-9


def test_init_from_light_fielg_geometry():
    path = pkg_resources.resource_filename(
        "plenopy", "tests/resources/run.acp/input/plenoscope"
    )
    lfg = pl.LightFieldGeometry(path)
    rays = pl.Tomography.Rays.from_light_field_geometry(lfg)

    assert rays.support.shape[0] == 19741
    assert rays.support.shape[1] == 3

    assert (
        np.linalg.norm([rays.support[:, 0], rays.support[:, 1]], axis=0).max()
        < 8.5
    )
