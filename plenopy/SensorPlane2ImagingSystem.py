#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, print_function, division
import numpy as np
import glob
import os

__all__ = ['SensorPlane2ImagingSystem']


class SensorPlane2ImagingSystem(object):

    def __init__(self, path):
        self.homogeneous_transformation = self.__read(path)
        self.light_filed_sensor_distance = self.homogeneous_transformation[
            2, 3]

    def __read(self, path):
        gh = np.fromfile(path, dtype=np.float32)
        return np.array([
            [gh[11 - 1], gh[14 - 1], gh[17 - 1], gh[20 - 1]],
            [gh[12 - 1], gh[15 - 1], gh[18 - 1], gh[21 - 1]],
            [gh[13 - 1], gh[16 - 1], gh[19 - 1], gh[22 - 1]],
            [0.0,       0.0,       0.0,       1.0],
        ])
