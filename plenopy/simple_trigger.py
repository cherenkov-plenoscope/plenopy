import numpy as np
from .image import ImageRays
import scipy.spatial
import array


def make_hexagonal_grid(
    outer_radius,
    spacing,
    inner_radius=0.0
):
    hex_a = np.array([np.sqrt(3)/2, 0.5])*spacing
    hex_b = np.array([0, 1])*spacing

    grid = []
    sample_radius = 2.0*np.floor(outer_radius/spacing);
    for a in np.arange(-sample_radius, sample_radius+1):
        for b in np.arange(-sample_radius, sample_radius+1):
            cell_ab = hex_a*a + hex_b*b
            cell_norm = np.linalg.norm(cell_ab)
            if cell_norm <= outer_radius and cell_norm >= inner_radius:
                grid.append(cell_ab);
    return np.array(grid)


def estimate_projection_of_light_field_to_image(
    light_field_geometry,
    object_distance,
    image_pixel_cx_rad,
    image_pixel_cy_rad,
    image_pixel_radius_rad,
    number_nearest_neighbors=100,
):
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
        k=number_nearest_neighbors,
    )
    projection = []
    lixel_to_pixel_distances_rad = search[0]
    lixel_to_pixel_ids = search[1]
    for lix in range(light_field_geometry.number_lixel):
        lixel_to_pixel = []
        for pix in range(number_nearest_neighbors):
            dd = lixel_to_pixel_distances_rad[lix, pix]
            if dd <= image_pixel_radius_rad:
                lixel_to_pixel.append(lixel_to_pixel_ids[lix, pix])
        projection.append(lixel_to_pixel)
    return projection


def list_of_lists_to_arrays(list_of_lists):
    starts = array.array('l')
    lengths = array.array('l')
    stream = array.array('l')
    i = 0
    for _list in list_of_lists:
        starts.append(i)
        l = 0
        for symbol in _list:
            stream.append(symbol)
            i += 1
            l += 1
        lengths.append(l)
    return {
        "starts": np.array(starts, dtype=np.uint32),
        "lengths": np.array(lengths, dtype=np.uint32),
        "links": np.array(stream, dtype=np.uint32),
    }
