import numpy as np
import array
import scipy.spatial

from .. import tools
from ..image import ImageRays


def generate_trigger_image(
    image_outer_radius_rad,
    pixel_spacing_rad,
    pixel_radius_rad,
    max_number_nearest_lixel_in_pixel,
):
    assert image_outer_radius_rad > 0.
    assert pixel_spacing_rad > 0.
    assert pixel_radius_rad > 0.
    grid_cx_cy = tools.hexagonal_grid.make_hexagonal_grid(
        outer_radius=image_outer_radius_rad,
        spacing=pixel_spacing_rad,
        inner_radius=0.0
    )
    trg_img = {}
    trg_img["pixel_cx_rad"] = grid_cx_cy[:, 0]
    trg_img["pixel_cy_rad"] = grid_cx_cy[:, 1]
    trg_img["pixel_radius_rad"] = pixel_radius_rad
    trg_img["number_pixel"] = grid_cx_cy.shape[0]
    trg_img["max_number_nearest_lixel_in_pixel"] = (
        max_number_nearest_lixel_in_pixel
    )
    return trg_img


def prepare_trigger_geometry(
    light_field_geometry,
    trigger_image,
    object_distances=[7.5e3, 15e3, 22.5e3],
):
    tg = {}
    tg['image'] = trigger_image
    tg['number_foci'] = len(object_distances)
    tg['number_lixel'] = np.uint32(light_field_geometry.number_lixel)
    tg['foci'] = []

    for object_distance in object_distances:

        lixel_to_pixel = estimate_projection_of_light_field_to_image(
            light_field_geometry=light_field_geometry,
            object_distance=object_distance,
            image_pixel_cx_rad=tg['image']['pixel_cx_rad'],
            image_pixel_cy_rad=tg['image']['pixel_cy_rad'],
            image_pixel_radius_rad=tg['image']['pixel_radius_rad'],
            max_number_nearest_lixel_in_pixel=tg['image'][
                'max_number_nearest_lixel_in_pixel'],
        )

        focus = list_of_lists_to_arrays(list_of_lists=lixel_to_pixel)
        focus["object_distance_m"] = object_distance
        tg['foci'].append(focus)
    return tg


def estimate_projection_of_light_field_to_image(
    light_field_geometry,
    object_distance,
    image_pixel_cx_rad,
    image_pixel_cy_rad,
    image_pixel_radius_rad,
    max_number_nearest_lixel_in_pixel=7,
):
    '''
    Returns a list over lixels of lists of pixels.
    For each lixel there is a list of pixels where the lixel has to be added
    to.
    [
        [pixel A, pixel B, pixel C, ... ],    lixel 0
        [pixel B, pixel C, pixel D, ... ],    lixel 1
        [pixel D, pixel E, pixel F, ... ],    lixel 2
        .
        .
        .
        [...],    lixel N
    ]

    Parameters
    ----------
    light_field_geometry

    object_distance             The object-distance the image is focused on.
                                float

    image_pixel_cx_rad          Array of floats

    image_pixel_cy_rad          Array of floats

    image_pixel_radius_rad      Array of floats
    '''
    image_rays = ImageRays(light_field_geometry)
    lixel_cx, lixel_cy = image_rays.cx_cy_in_object_distance(object_distance)
    trigger_pixel_tree = scipy.spatial.cKDTree(
        np.array([
            image_pixel_cx_rad,
            image_pixel_cy_rad
        ]).T
    )
    search = trigger_pixel_tree.query(
        x=np.vstack((lixel_cx, lixel_cy)).T,
        k=max_number_nearest_lixel_in_pixel,
    )
    projection_lixel_to_pixel = []
    lixel_to_pixel_distances_rad = search[0]
    lixel_to_pixel_ids = search[1]
    for lix in range(light_field_geometry.number_lixel):
        lixel_to_pixel = []
        for pix in range(max_number_nearest_lixel_in_pixel):
            dd = lixel_to_pixel_distances_rad[lix, pix]
            if dd <= image_pixel_radius_rad:
                lixel_to_pixel.append(lixel_to_pixel_ids[lix, pix])
        projection_lixel_to_pixel.append(lixel_to_pixel)
    return projection_lixel_to_pixel


def list_of_lists_to_arrays(list_of_lists):
    starts = array.array('l')
    lengths = array.array('l')
    stream = array.array('l')
    i = 0
    for _list in list_of_lists:
        starts.append(i)
        length = 0
        for symbol in _list:
            stream.append(symbol)
            i += 1
            length += 1
        lengths.append(length)
    return {
        "starts": np.array(starts, dtype=np.uint32),
        "lengths": np.array(lengths, dtype=np.uint32),
        "links": np.array(stream, dtype=np.uint32),
    }


def arrays_to_list_of_lists(starts, lengths, links):
    number_lixel = starts.shape[0]
    lol = [[] for p in range(number_lixel)]
    for lixel in range(number_lixel):
        for l in range(lengths[lixel]):
            idx = starts[lixel] + l
            lol[lixel].append(links[idx])
    return lol


def invert_projection_matrix(lixel_to_pixel, number_pixel, number_lixel):
    assert number_lixel == len(lixel_to_pixel)
    pixel_to_lixel = [[] for p in range(number_pixel)]
    for lixel in range(number_lixel):
        for pixel in lixel_to_pixel[lixel]:
            pixel_to_lixel[pixel].append(lixel)
    return pixel_to_lixel
