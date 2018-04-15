import numpy as np
import struct


def assert_shape_is_valid(h):
    assert h.shape[0] == 273


def str2float(four_byte_string):
    return struct.unpack('f', four_byte_string.encode())[0]


def assert_marker_of_header_is(h, expected_marker_string):
    assert h[0] == str2float(expected_marker_string)


def read_float32_header(path):
    raw = np.fromfile(path, dtype=np.float32)
    return raw
