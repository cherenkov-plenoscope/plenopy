import numpy as np
from .image import ImageRays
import scipy.spatial
import array
import json
import os


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


def generate_trigger_image_from_physical_layout(light_field_geometry):
    r"""
    A trigger-patch of pixels for pixel 'c':
                     ___
                 ___/   \___
                /   \___/   \
                \___/ c \___/
                /   \___/   \
                \___/   \___/
                    \___/

    """
    lfg = light_field_geometry
    trg_img = {}
    trg_img["pixel_cx_rad"] = lfg.pixel_pos_cx
    trg_img["pixel_cy_rad"] = lfg.pixel_pos_cy
    trg_img["pixel_radius_rad"] = (
        2.2*lfg.sensor_plane2imaging_system.pixel_FoV_hex_flat2flat)
    trg_img["number_pixel"] = lfg.pixel_pos_cy.shape[0]
    return trg_img


def generate_trigger_image(
    image_outer_radius_rad,
    pixel_spacing_rad,
    pixel_radius_rad,
):
    assert image_outer_radius_rad > 0.
    assert pixel_spacing_rad > 0.
    assert pixel_radius_rad > 0.
    grid_cx_cy = make_hexagonal_grid(
        outer_radius=image_outer_radius_rad,
        spacing=pixel_spacing_rad,
        inner_radius=0.0
    )
    trg_img = {}
    trg_img["pixel_cx_rad"] = grid_cx_cy[:, 0]
    trg_img["pixel_cy_rad"] = grid_cx_cy[:, 1]
    trg_img["pixel_radius_rad"] = pixel_radius_rad
    trg_img["number_pixel"] = grid_cx_cy[:, 0]
    return trg_img


def prepare_trigger_geometry(
    light_field_geometry,
    trigger_image,
    object_distances=[7.5e3, 15e3, 22.5e3],
    max_number_nearest_image_pixels=100,
):
    tg = {}
    tg['image'] = trigger_image
    tg['number_foci'] = len(object_distances)
    tg['number_lixel'] = np.uint32(light_field_geometry.number_lixel)
    tg['foci'] = []

    for object_distance in object_distances:

        projection_lol = estimate_projection_of_light_field_to_image(
            light_field_geometry=light_field_geometry,
            object_distance=object_distance,
            image_pixel_cx_rad=tg['image']['pixel_cx_rad'],
            image_pixel_cy_rad=tg['image']['pixel_cy_rad'],
            image_pixel_radius_rad=tg['image']['pixel_radius_rad'],
            max_number_nearest_image_pixels=max_number_nearest_image_pixels,
        )
        focus = list_of_lists_to_arrays(list_of_lists=projection_lol)
        focus["object_distance_m"] = object_distance
        tg['foci'].append(focus)
    return tg


def estimate_projection_of_light_field_to_image(
    light_field_geometry,
    object_distance,
    image_pixel_cx_rad,
    image_pixel_cy_rad,
    image_pixel_radius_rad,
    max_number_nearest_image_pixels=100,
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
        k=max_number_nearest_image_pixels,
    )
    projection_lixel_to_pixel = []
    lixel_to_pixel_distances_rad = search[0]
    lixel_to_pixel_ids = search[1]
    for lix in range(light_field_geometry.number_lixel):
        lixel_to_pixel = []
        for pix in range(max_number_nearest_image_pixels):
            dd = lixel_to_pixel_distances_rad[lix, pix]
            if dd <= image_pixel_radius_rad:
                lixel_to_pixel.append(lixel_to_pixel_ids[lix, pix])
        projection_lixel_to_pixel.append(lixel_to_pixel)
    return projection_lixel_to_pixel


def _write(path, value, dtype):
    with open(path+'.'+dtype, 'wb') as f:
        f.write(np.float32(value).astype(dtype).tobytes())


def _read(path, dtype):
    with open(path+'.'+dtype, 'rb') as f:
        return np.frombuffer(f.read(), dtype=dtype)


def write_trigger_geometry_to_path(trigger_geometry, path):
    assert_trigger_geometry_consistent(trigger_geometry=trigger_geometry)
    os.makedirs(path, exist_ok=True)
    tg = trigger_geometry

    join = os.path.join
    _write(join(path, "number_lixel"), tg['number_lixel'], 'u4')

    _write(join(path, "image.number_pixel"), tg['image']['number_pixel'], 'u4')
    _write(join(path, "image.pixel_cx_rad"), tg['image']['pixel_cx_rad'], 'f4')
    _write(join(path, "image.pixel_cy_rad"), tg['image']['pixel_cy_rad'], 'f4')
    _write(
        join(path, "image.pixel_radius_rad"),
        tg['image']['pixel_radius_rad'],
        'f4'
    )

    _write(join(path, "number_foci"), tg['number_foci'], 'u4')
    for focus in range(tg['number_foci']):
        name = "foci.{:06d}".format(focus)
        _write(
            join(path, name+".object_distance_m"),
            tg['foci'][focus]['object_distance_m'],
            'f4')
        _write(join(path, name+".starts"), tg['foci'][focus]['starts'], 'u4')
        _write(join(path, name+".lengths"), tg['foci'][focus]['lengths'], 'u4')
        _write(join(path, name+".links"), tg['foci'][focus]['links'], 'u4')


def read_trigger_geometry_from_path(path):
    join = os.path.join
    tg = {}
    tg['number_lixel'] = _read(join(path, "number_lixel"), 'u4')[0]

    tg['image'] = {}
    tg['image']['number_pixel'] = _read(
        join(path, "image.number_pixel"),
        'u4'
    )[0]
    tg['image']['pixel_cx_rad'] = _read(join(path, "image.pixel_cx_rad"), 'f4')
    tg['image']['pixel_cy_rad'] = _read(join(path, "image.pixel_cy_rad"), 'f4')
    tg['image']['pixel_radius_rad'] = _read(
        join(path, "image.pixel_radius_rad"),
        'f4'
    )[0]

    tg['number_foci'] = _read(join(path, "number_foci"), 'u4')[0]
    tg['foci'] = [{} for i in range(tg['number_foci'])]
    for focus in range(tg['number_foci']):
        name = "foci.{:06d}".format(focus)
        tg['foci'][focus]['object_distance_m'] = _read(
            join(path, name+".object_distance_m"),
            'f4'
        )[0]
        tg['foci'][focus]['starts'] = _read(join(path, name+".starts"), 'u4')
        tg['foci'][focus]['lengths'] = _read(join(path, name+".lengths"), 'u4')
        tg['foci'][focus]['links'] = _read(join(path, name+".links"), 'u4')

    assert_trigger_geometry_consistent(trigger_geometry=tg)
    return tg


def assert_trigger_geometry_consistent(trigger_geometry):
    tg = trigger_geometry
    assert tg['image']['number_pixel'] == tg['image']['pixel_cx_rad'].shape[0]
    assert tg['image']['number_pixel'] == tg['image']['pixel_cy_rad'].shape[0]
    assert tg['image']['pixel_radius_rad'] >= 0.0

    for focus in range(tg['number_foci']):
        assert tg['number_lixel'] == tg['foci'][focus]['starts'].shape[0]
        assert tg['number_lixel'] == tg['foci'][focus]['lengths'].shape[0]


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
