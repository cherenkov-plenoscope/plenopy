#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, print_function, division
import numpy as np
import scipy.spatial
import os
from .LixelRays import LixelRays
from .SensorPlane2ImagingSystem import SensorPlane2ImagingSystem
from .HeaderRepresentation import assert_marker_of_header_is


class LixelStatistics(object):
    """
    number_lixel    The number count of light field cells (lixel)
                    This is also the number of read out channels.
                    In this plenoscope design, each lixel is both
                    participating to a picture cell (pixel) and 
                    a principal aperture cell (paxel).
                    number_lixel = number_pixel x number_paxel

    number_pixel    The number of directional clusters forming
                    the picture cells. Each pixel is composed of
                    number_paxel paxel.

    number_paxel    The number of positional clusters forming 
                    the principal aperture cells (paxel) on the 
                    principal aperture plane

    efficiency      [number_pixel x number_paxel]
                    The average efficiency of a lixel cell. During 
                    the calibration simulation, photons thrown
                    into the plenoscope. The photons are evenly 
                    spread over the aperture and the field of view.
                    The more photons reached the lixel sensor, the 
                    higher is the efficiency. This takes losses into
                    account except for the photo-electric sensor 
                    efficiency. [1]

    x_mean, x_std, 
    y_mean, y_std   [number_pixel x number_paxel]
                    The average x,y position and its spread of the spatial 
                    lixel bin on the principal aperture plane. [m]

    cx_mean, cx_std, 
    cy_mean, cy_std [number_pixel x number_paxel]
                    The average cos_x direction and its spread of the 
                    directional lixel bin in the field of view [rad]

                    The cx or cy is short for cos_x or cos_y. cx and cy are 
                    the x and y components of the normalized incomming 
                    direction vector vec{d} on the principal aperture plane.

                    vec{d} = (cx, cy, sqrt(1 - cx^2 - cy^2))

    time_delay_mean, 
    time_delay_std  [number_pixel x number_paxel]
                    The average arrival time delay and its spread for 
                    a photon to travel from the principal aperture 
                    plane to the lixel sensor. [s]

    rays            The rays corresponding to all the lixels. Every lixel 
                    defines a ray with its support x_mean, y_mean on the 
                    principal aperture plane and its direction cx_mean, cy_mean 
                    in the field of view.

    pixel_pos_cx, 
    pixel_pos_cy    [number_pixel]
                    The average direction accross all the lixels cx_mean and 
                    cy_mean in pixel [rad]

    paxel_pos_x, 
    paxel_pos_y     [number_paxel]
                    The average position accross all the lixels x_mean and 
                    y_mean in paxel on the principal aperture plane. [m]

    expected_focal_length_of_imaging_system     The focal length of the
                                                imaging system, the light 
                                                field sensor was designed 
                                                for [m]

    expected_aperture_radius_of_imaging_system  The radius of the imaging
                                                system's aperture the light 
                                                field sensor was designed
                                                for [m]

    pixel_pos_tree  Helps to locate neighbor pixels.
                    A 2D quad tree structure storing all the pixel_pos_cx and
                    pixel_pos_cy pairs. Neighboring pixles can be found 
                    efficiently with a query on the tree.

    paxel_pos_tree  Helps to locate neighbor paxels.
                    A 2D quad tree structure storing all the paxel_pos_x and
                    paxel_pos_y pairs. Neighboring paxles can be found 
                    efficiently with a query on the tree.

    lixel_outer_radius  The outer radius of the hexagonal lixel sensor area 
                        in the light field sensor. [m]

    lixel_z_orientation The orientation angel of the hexagonal lixel sensor area 
                        in the light field sensor. [rad]

    lixel_positions_x, 
    lixel_positions_y   The hexagonal lixel sensor center x,y positions in the
                        light field sensor. With respect to the light field 
                        sensor plane frame. [m] 
    ------ 
    """

    def __init__(self, path):
        path = os.path.abspath(path)

        self.__read_light_field_sensor_geometry_header(
            os.path.join(path, 'light_field_sensor_geometry.header.bin'))
        self.__read_lixel_positions(os.path.join(path, 'lixel_positions.bin'))
        self.__read_lixel_statistics(
            os.path.join(path, 'lixel_statistics.bin'))

        self.__calc_pixel_and_paxel_average_positions()
        self.__init_lixel_polygons()
        self.__init_lixel_rays()

        #self.valid_efficiency = self.efficiency > 0.10
        self.valid_efficiency = self.most_efficient_lixels(0.95)

    def __calc_pixel_and_paxel_average_positions(self):
        self.paxel_pos_x = np.nanmean(self.x_mean, axis=0)
        self.paxel_pos_y = np.nanmean(self.y_mean, axis=0)

        self.pixel_pos_cx = np.nanmean(self.cx_mean, axis=1)
        self.pixel_pos_cy = np.nanmean(self.cy_mean, axis=1)

        self.pixel_pos_tree = scipy.spatial.cKDTree(
            np.array([self.pixel_pos_cx, self.pixel_pos_cy]).T)
        self.paxel_pos_tree = scipy.spatial.cKDTree(
            np.array([self.paxel_pos_x, self.paxel_pos_y]).T)

        self.paxel_efficiency_along_pixel = np.nanmean(self.efficiency, axis=0)
        self.pixel_efficiency_along_paxel = np.nanmean(self.efficiency, axis=1)

    def __read_lixel_statistics(self, path):
        ls = np.fromfile(path, dtype=np.float32)
        ls = ls.reshape([ls.shape[0] / 12, 12])

        for i, attribute_name in enumerate([
                'efficiency', 'efficiency_std',
                'cx_mean', 'cx_std',
                'cy_mean', 'cy_std',
                'x_mean', 'x_std',
                'y_mean', 'y_std',
                'time_delay_mean', 'time_delay_std'
        ]):
            setattr(
                self,
                attribute_name,
                ls[:, i].reshape(self.number_pixel, self.number_paxel)
            )

    def __read_light_field_sensor_geometry_header(self, path):
        gh = np.fromfile(path, dtype=np.float32)
        assert_marker_of_header_is(gh, 'PLGH')
        self.number_pixel = int(gh[101 - 1])
        self.number_paxel = int(gh[102 - 1])
        self.number_lixel = self.number_pixel * self.number_paxel

        self.lixel_outer_radius = gh[103 - 1]
        self.lixel_z_orientation = gh[105 - 1]

        self.expected_focal_length_of_imaging_system = gh[23 - 1]
        self.expected_aperture_radius_of_imaging_system = gh[24 - 1]

        self.sensor_plane2imaging_system = SensorPlane2ImagingSystem(path)

    def __read_lixel_positions(self, path):
        lp = np.fromfile(path, dtype=np.float32)
        lp = lp.reshape([lp.shape[0] / 2, 2])
        self.lixel_positions_x = lp[:, 0]
        self.lixel_positions_y = lp[:, 1]

    def __init_lixel_polygons(self):
        s32 = np.sqrt(3) / 2.

        poly_template = np.array([
            [0, 1],
            [-s32, 0.5],
            [-s32, -0.5],
            [0, -1],
            [s32, -0.5],
            [s32, 0.5],
        ])
        poly_template *= self.lixel_outer_radius

        lixel_centers_xy = np.array([
            self.lixel_positions_x,
            self.lixel_positions_y
        ])

        self.lixel_polygons = [xy + poly_template for xy in lixel_centers_xy.T]

    def __init_lixel_rays(self):
        self.rays = LixelRays(
            x=self.x_mean.flatten(),
            y=self.y_mean.flatten(),
            cx=self.cx_mean.flatten(),
            cy=self.cy_mean.flatten())

    def most_efficient_lixels(self, fraction):
        """
        Returns a boolean mask (shape=[numnper_pixel, number_paxel]) of
        the most efficient lixels. 

        Parameters
        ----------
        fraction    float 0.0-1.0 the fraction of most efficient lixels
                    to be masked. fraction=1.0 will mask all the lixels
                    and fraction=0.5 will mask the 50 percent most
                    efficient lixels.
        """
        number_valid_lixels = int(np.floor(self.number_lixel * fraction))
        flat_idxs = np.argsort(self.efficiency.flatten()
                               )[-number_valid_lixels:]
        flat_mask = np.zeros(self.number_lixel, dtype=bool)
        flat_mask[flat_idxs] = True
        return flat_mask.reshape([self.number_pixel, self.number_paxel])

    def __repr__(self):
        out = 'LixelStatistics( '
        out += str(self.number_lixel) + ' lixel = '
        out += str(self.number_pixel) + ' pixel x '
        out += str(self.number_paxel) + ' paxel)\n'
        return out
