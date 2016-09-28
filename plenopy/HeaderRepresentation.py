import numpy as np
import struct


class HeaderBadShape(Exception):
    pass


class WrongMarker(Exception):
    pass


def assert_shape_is_valid(h):
    if h.shape[0] != 273:
        raise HeaderBadShape


def str2float(four_byte_string):
    return struct.unpack('f', four_byte_string.encode())[0]


def assert_marker_of_header_is(h, expected_marker_string):
    if h[0] != str2float(expected_marker_string):
        raise WrongMarker


def mct_sensor_plane2imaging_system(h):
    assert_shape_is_valid(h)

    out = 'MCT plenoscope sensor plane 2 imaging_system header\n'

    if int(h[2 - 1]) == 1:
        out += '  2 ' + 'Type: Monte Carlo\n'
    elif int(h[2 - 1]) == 0:
        out += '  2 ' + 'Type: Observation\n'

    SensorPlane2ImagingSystem()

    return out


def read_float32_header(path):
    raw = np.fromfile(path, dtype=np.float32)
    return raw  