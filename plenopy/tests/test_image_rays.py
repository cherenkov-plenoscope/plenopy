import numpy as np
import plenopy as pl
import pkg_resources
import pytest


@pytest.fixture(scope='session')
def light_field_geometry():
    path = pkg_resources.resource_filename(
        'plenopy',
        'tests/resources/run.acp/input/plenoscope')
    lfg = pl.LightFieldGeometry(path)
    assert lfg.number_lixel == 19741
    assert lfg.number_pixel == 1039
    assert lfg.number_paxel == 19
    return lfg


def test_number_rays(light_field_geometry):
    image_rays = pl.image.ImageRays(light_field_geometry)
    assert image_rays.support.shape[0] == light_field_geometry.number_lixel
    assert image_rays.direction.shape[0] == light_field_geometry.number_lixel


def test_image_rays_cx_cy_for_obj(light_field_geometry):
    image_rays = pl.image.ImageRays(light_field_geometry)
    cx, cy = image_rays.pixel_ids_of_lixels_in_object_distance(10e3)
    assert cx.shape[0] == light_field_geometry.number_lixel
    assert cy.shape[0] == light_field_geometry.number_lixel


def test_image_rays_inside_fov(light_field_geometry):
    num_lix = light_field_geometry.number_lixel
    image_rays = pl.image.ImageRays(light_field_geometry)

    objs_mins_maxs = [
        [25.e3, 0.98, 1.00],
        [5.0e3, 0.97, 0.98],
        [2.5e3, 0.96, 0.97],
        [1.2e3, 0.91, 0.93],
        [0.6e3, 0.82, 0.83],
        [0.3e3, 0.61, 0.62],
    ]

    for obj_min_max in objs_mins_maxs:
        object_distance = obj_min_max[0]
        min_fraction = obj_min_max[1]
        max_fraction = obj_min_max[2]
        pixel_ids, in_fov = image_rays.pixel_ids_of_lixels_in_object_distance(
            object_distance)
        assert (
            in_fov.sum() >= min_fraction*num_lix and
            in_fov.sum() < max_fraction*num_lix)
