#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, print_function, division
import numpy as np

class LixelRays(object):
    """
    support     [number_lixel x 3]
                Support vectors of all the lixel rays on the principal aperture
                plane. [x_mean, y_mean, 0]

    direction   [number_lixel x 3]
                Direction vectors of all the lixel rays.
                [cx_mean, cy_mean, sqrt(1 - cx_mean^2 - cy_mean^2)]
    """
    def __init__(self, x, y, cx, cy):
        number_lixel = x.shape[0]
        self.support = np.array([x, y, np.zeros(number_lixel)]).T
        dir_z = np.sqrt(1.0 - cx**2.0 - cy**2.0)
        # (cos_x, cos_y, sqrt(1 - cos_x^2 - cos_y^2))^T
        self.direction  = np.array([cx, cy, dir_z]).T

    def slice_intersections_in_object_distance(self, object_distance):
        """
        Returns the x,y intersections of the lixel rays with the x,y plane 
        at z=object_distance.

        Parameters
        ----------
        object_distance     The distance to the principal aperture plane.
        """
        scale_factors = object_distance/self.direction[:,2]
        pos3D = self.support - (scale_factors*self.direction.T).T
        return pos3D[:,0:2]

    def __str__(self):
        out = 'LixelRays('
        out+= str(self.support.shape[0])+' lixels'
        out+= ')\n'
        return out

    def __repr__(self):
        return self.__str__()