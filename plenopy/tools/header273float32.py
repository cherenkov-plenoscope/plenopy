import numpy as np
import struct


def assert_shape_is_valid(h):
    assert h.shape[0] == 273


def str2float(four_byte_string):
    return struct.unpack("f", four_byte_string.encode())[0]


def str2int32(four_byte_string):
    return struct.unpack("i", four_byte_string.encode())[0]


def assert_marker_of_header_is(h, expected_marker_string):
    assert h[0] == str2float(expected_marker_string)


def interpret_bytes_from_float32_as_int32(float32):
    ff = np.float32(float32)
    return struct.unpack("i", struct.pack("f", ff))


def read_float32_header(path):
    raw = np.fromfile(path, dtype=np.float32)
    return raw
