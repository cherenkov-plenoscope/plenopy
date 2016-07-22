#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, print_function, division
import numpy as np

class RawLighFieldSensorResponse(object):
    """
    The raw response of simple TDCs and QDCs of all the number_lixel read out 
    channels.

    arrival_time    [number_lixel]
                    The relative arrival times of the pulses found in each lixel
                    sensor the light field sensor [s].

    intensity       [number_lixel]
                    The amount of photo equivalent detected in each lixel 
                    sensor. [p.e.]
    """
    def __init__(self, path):
        raw = np.fromfile(path, dtype=np.float32)
        raw = raw.reshape([raw.shape[0]/2 ,2])
        self.arrival_time = raw[:,0]
        self.intensity = raw[:,1]

    def __repr__(self):
        out = 'RawLighFieldSensorResponse('
        out+= str(self.arrival_time.shape[0])+' lixel, '
        out+= 'Sum_Intensity '+str(round(self.intensity.sum()))+' p.e.)\n'
        return out