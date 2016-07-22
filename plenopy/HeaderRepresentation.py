#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, print_function, division
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


def corsika_event_header_repr(h):
    assert_shape_is_valid(h)
    assert_marker_of_header_is(h, 'EVTH')

    out = 'CORSIKA event header\n'
    out += '  2 ' + 'event number ' + str(int(h[2 - 1])) + '\n'
    out += '  3 ' + 'particle id ' + str(int(h[3 - 1])) + '\n'
    out += '  4 ' + 'total energy ' + str(h[4 - 1]) + ' GeV\n'
    out += '  5 ' + 'starting altitude ' + str(h[5 - 1]) + ' g/cm^2\n'
    #out+= '  6 '+'number of first target if fixed'+str(h[6-1])+'\n'
    out += '  7 ' + \
        'z coordinate (height) of first interaction ' + str(h[7 - 1]) + ' cm\n'
    out += ' 11 ' + 'zenith angle Theta ' + \
        str(np.rad2deg(h[11 - 1])) + ' deg\n'
    out += ' 12 ' + 'azimuth angle Phi ' + \
        str(np.rad2deg(h[12 - 1])) + ' deg\n'
    out += ' 98 ' + 'number of uses ' + str(int(h[98 - 1])) + '\n'
    for i in range(int(h[98 - 1])):
        out += '    ' + 'reuse ' + str(i + 1) + ': core position x=' + str(
            h[98 - 1 + i + 1]) + ' cm, y=' + str(h[118 - 1 + i + 1]) + ' cm\n'
    return out


def corsika_run_header_repr(h):
    assert_shape_is_valid(h)
    assert_marker_of_header_is(h, 'RUNH')

    out = 'CORSIKA run header\n'
    out += '  2 ' + 'run number ' + str(int(h[2 - 1])) + '\n'
    out += '  4 ' + 'date of begin run (yymmdd) ' + str(int(h[3 - 1])) + '\n'
    out += '  4 ' + 'version of program ' + str(h[4 - 1]) + '\n'
    out += ' 16 ' + 'slope of energy spectrum ' + str(h[16 - 1]) + '\n'
    out += ' 17 ' + 'lower limit of energy range ' + str(h[17 - 1]) + ' GeV\n'
    out += ' 18 ' + 'upper limit of energy range ' + str(h[18 - 1]) + ' GeV\n'
    out += '248 ' + 'XSCATT scatter range in x direction for Cherenkov ' + \
        str(h[248 - 1]) + ' cm\n'
    out += '249 ' + 'YSCATT scatter range in x direction for Cherenkov ' + \
        str(h[249 - 1]) + ' cm\n'
    return out


def mct_sensor_plane2imaging_system(h):
    assert_shape_is_valid(h)

    out = 'MCT plenoscope sensor plane 2 imaging_system header\n'

    if int(h[2 - 1]) == 1:
        out += '  2 ' + 'Type: Monte Carlo\n'
    elif int(h[2 - 1]) == 0:
        out += '  2 ' + 'Type: Observation\n'

    SensorPlane2ImagingSystem()

    return out
