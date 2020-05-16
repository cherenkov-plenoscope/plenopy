import numpy as np
import os
import json


def write_trigger_geometry_to_path(trigger_geometry, path):
    assert_trigger_geometry_consistent(trigger_geometry=trigger_geometry)
    os.makedirs(path, exist_ok=True)
    tg = trigger_geometry

    join = os.path.join
    _write(join(path, "number_lixel"), tg['number_lixel'], 'u4')

    _write(join(path, "image.number_pixel"), tg['image']['number_pixel'], 'u4')
    _write(
        join(path, "image.max_number_nearest_lixel_in_pixel"),
        tg['image']['max_number_nearest_lixel_in_pixel'],
        'u4')
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
    tg['image']['max_number_nearest_lixel_in_pixel'] = _read(
        join(path, "image.max_number_nearest_lixel_in_pixel"),
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


def _write(path, value, dtype):
    with open(path+'.'+dtype, 'wb') as f:
        f.write(np.float32(value).astype(dtype).tobytes())


def _read(path, dtype):
    with open(path+'.'+dtype, 'rb') as f:
        return np.frombuffer(f.read(), dtype=dtype)


def assert_trigger_geometry_consistent(trigger_geometry):
    tg = trigger_geometry
    assert tg['image']['number_pixel'] == tg['image']['pixel_cx_rad'].shape[0]
    assert tg['image']['number_pixel'] == tg['image']['pixel_cy_rad'].shape[0]
    assert tg['image']['pixel_radius_rad'] >= 0.0

    for focus in range(tg['number_foci']):
        assert tg['number_lixel'] == tg['foci'][focus]['starts'].shape[0]
        assert tg['number_lixel'] == tg['foci'][focus]['lengths'].shape[0]


def read_trigger_response_from_path(path):
    with open(path, 'rt') as fin:
        trigger_response = json.loads(fin.read())
    return trigger_response


def write_trigger_response_to_path(trigger_response ,path):
    with open(path, 'wt') as fout:
        fout.write(json.dumps(trigger_response, indent=4))
