#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, print_function, division
import numpy as np


class CherenkovProductionVolume(object):

    def __init__(self, light_field, obj_dist_min=1e3, obj_dist_max=25e3):

        n_z_bins = int(np.sqrt(light_field.number_pixel))

        object_distances = np.logspace(
            np.log10(obj_dist_min),
            np.log10(obj_dist_max),
            n_z_bins
        )

        n_lixel = light_field.valid_lixel.sum()
        xs = np.zeros([n_z_bins, n_lixel])
        ys = np.zeros([n_z_bins, n_lixel])
        lixel_intensity = light_field.intensity.flatten(
        )[light_field.valid_lixel.flatten()]

        for i, object_distance in enumerate(object_distances):
            pos_xy = light_field.rays.slice_intersections_in_object_distance(
                object_distance)
            xs[i, :] = pos_xy[:, 0][light_field.valid_lixel.flatten()]
            ys[i, :] = pos_xy[:, 1][light_field.valid_lixel.flatten()]

        r = 5.0 * light_field.expected_aperture_radius_of_imaging_system
        xmax = r
        xmin = -r
        ymax = r
        ymin = -r

        n_xy_bins = n_z_bins

        self.intensity = np.zeros([n_z_bins, n_xy_bins, n_xy_bins])
        self.xedges = np.zeros(n_xy_bins + 1)
        self.yedges = np.zeros(n_xy_bins + 1)
        for i in range(n_z_bins):
            self.intensity[i, :, :], self.xedges, self.yedges = np.histogram2d(
                x=xs[i],
                y=ys[i],
                weights=lixel_intensity,
                bins=[n_xy_bins, n_xy_bins],
                range=[[xmin, xmax], [ymin, ymax]])

    def __repr__(self):
        out = 'CherenkovProductionVolume()\n'
        return out
