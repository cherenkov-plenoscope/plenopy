#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, print_function, division
import numpy as np
from .Image import Image
from . import Masks


class LightField(object):

    def __init__(self, raw_plenoscope_response, lixel_statistics, sensor_plane2imaging_system):
        self.__dict__ = lixel_statistics.__dict__.copy()
        self.__doc__ = """
        intensity       [number_pixel x number_paxel]
                        The photon equivalent (p.e.) count in each lixel. [p.e.]

        arrival_time    [number_pixel x number_paxel]
                        The relative arrival time of the puls in each lixel on 
                        the principal aperture plane [s]

        valid_lixel     [number_pixel x number_paxel]
                        A boolean mask which marks all the lixels both having
                        sufficien efficiency (valid_efficiency) AND have 
                        reasonable relative arrival times to form one event.
        """
        self.__doc__ += lixel_statistics.__doc__

        self.__sensor_plane2imaging_system = sensor_plane2imaging_system

        self.__init_intensity(raw_plenoscope_response)
        self.__init_arrival_times(raw_plenoscope_response)
        self.__init_valid_lixel_mask()

    def __init_intensity(self, raw_plenoscope_response):
        self.intensity = raw_plenoscope_response.intensity.copy()
        self.intensity = self.intensity.reshape(
            self.number_pixel,
            self.number_paxel)
        # correct for efficiency of lixels
        mean_efficiency_where_sensitive = self.efficiency[
            self.valid_efficiency].mean()
        self.intensity[np.invert(self.valid_efficiency)] = 0.0
        self.intensity[
            self.valid_efficiency] /= self.efficiency[self.valid_efficiency]
        self.intensity[
            self.valid_efficiency] *= mean_efficiency_where_sensitive

    def __init_arrival_times(self, raw_plenoscope_response):
        self.arrival_time = raw_plenoscope_response.arrival_time.copy()
        self.arrival_time = self.arrival_time.reshape(
            self.number_pixel,
            self.number_paxel)
        # correct for time delay of lixels
        self.arrival_time -= self.time_delay_mean
        self.arrival_time -= self.arrival_time[self.valid_efficiency].min()
        self.arrival_time[np.invert(self.valid_efficiency)] = 0.0

    def __init_valid_lixel_mask(self):
        # too low efficiency
        valid_effi = self.valid_efficiency
        # too large time delay
        speed_of_light = 3e8
        max_arrival_time_without_multiple_reflections_on_imaging_system = self.expected_focal_length_of_imaging_system / speed_of_light
        valid_time = self.arrival_time < max_arrival_time_without_multiple_reflections_on_imaging_system
        self.valid_lixel = np.logical_and(valid_time, valid_effi)

    def __refocus_alpha(self, wanted_object_distance):
        focal_length = self.expected_focal_length_of_imaging_system
        image_sensor_distance = self.__sensor_plane2imaging_system.light_filed_sensor_distance

        alpha = 1 / image_sensor_distance * 1 / \
            ((1 / focal_length) - (1 / wanted_object_distance))
        return alpha

    def refocus(self, wanted_object_distance):
        """
        Return directional image Intensity(cx, cy) as seen by a classical 
        Imaging Atmospheric Cherenkov Telescope when focusing on the 
        wanted_object_distance

        Parameters
        ----------
        wanted_object_distance  The wanted object distance to focus on

        Comments
        --------
        The image returned is the intesity distribution of the light rays on a 
        classic image sensor when this image sensor would have been positioned
        in the image distance b, corresponding to the wanted_object_distance g, 
        given the focal length f on the imaging system.

        1/f = 1/g + 1/b

        """
        focal_length = self.expected_focal_length_of_imaging_system
        alpha = self.__refocus_alpha(wanted_object_distance)

        n_pixel = self.number_pixel
        n_paxel = self.number_paxel

        dx = np.tan(self.pixel_pos_cx) * focal_length
        dy = np.tan(self.pixel_pos_cy) * focal_length

        px = self.paxel_pos_x
        py = self.paxel_pos_y

        dx_tick = px[:, np.newaxis] + \
            (dx[np.newaxis, :] - px[:, np.newaxis]) / alpha
        dy_tick = py[:, np.newaxis] + \
            (dy[np.newaxis, :] - py[:, np.newaxis]) / alpha

        dx_des = np.arctan(dx_tick / focal_length)
        dy_des = np.arctan(dy_tick / focal_length)

        # (127*8400, 2)
        desired_x_and_y = np.zeros((np.prod(dx_des.shape), 2))
        desired_x_and_y[:, 0] = dx_des.flatten()
        desired_x_and_y[:, 1] = dy_des.flatten()

        nearest_pixel_distances, nearest_pixel_indices = self.pixel_pos_tree.query(
            desired_x_and_y)

        summanden = self.intensity[
            nearest_pixel_indices, np.arange(n_paxel).repeat(n_pixel)]
        summanden = summanden.reshape(n_paxel, n_pixel)

        return Image(
            summanden.sum(axis=0),
            self.pixel_pos_cx,
            self.pixel_pos_cy)

    def pixel_sum(self, paxel_weights=None):
        """
        Return Image    The directional intesity distribution Intensity(cx,cy) 
                        over the field of view.

        Parameters
        ----------
        paxel_weights   [optional]
                        1D array with the weights of the paxel to be used to
                        form the image. All paxel weights =1.0 is the full
                        aperture of the plenoscope. You can provide 
                        paxel_weights which correspond to sub apertures to 
                        get images seen by only this sub aperture part.

        Comment
        -------
            This corresponds to the classical image as seen by an Imaging 
            Atmospheric Cherenkov Telescope
        """
        if paxel_weights is None:
            paxel_weights = np.ones(self.number_paxel)
        return Image(
            np.dot(self.intensity, paxel_weights),
            self.pixel_pos_cx,
            self.pixel_pos_cy)

    def paxel_sum(self, pixel_weights=None, interpolate_central_paxel=False):
        """
        Return Image    The positional intesity distribution on the principal 
                        aperture plane, i.e. Intensity(x,y).

        Parameters
        ----------
        pixel_weights   [optional]
                        1D array with the weights of the pixel to be used to
                        form the image. All pixel weights =1.0 is the full
                        field of view of the plenoscope. You can provide 
                        pixel_weights which correspond to sub field of views 
                        to get intensity distributions on the principal
                        aperture with only specific incoming directions 
                        cx,cy
        """

        if pixel_weights is None:
            pixel_weights = np.ones(self.number_pixel)
        paxel_intensity = np.dot(self.intensity.T, pixel_weights)

        if interpolate_central_paxel:
            central_paxel_and_neighbours_idxs = self.paxel_pos_tree.query([0, 0], 7)[
                1]
            central_paxel_idx = central_paxel_and_neighbours_idxs[0]
            neighbours_idxs = central_paxel_and_neighbours_idxs[1:]

            paxel_intensity[central_paxel_idx] = 0.0

            for n in neighbours_idxs:
                paxel_intensity[central_paxel_idx] += paxel_intensity[n]

            paxel_intensity[central_paxel_idx] /= neighbours_idxs.shape[0]

        return Image(
            paxel_intensity,
            self.paxel_pos_x,
            self.paxel_pos_y)

    def pixel_sum_sub_aperture(self, x, y, r, smear=0.0):
        """
        Return Image    An image as seen by a sub aperture at sub_x, sub_y with 
                        radius sub_r.

        Parameters
        ----------
        x, y    x and y position of sub aperture on principal aperture plane

        r       radius of sub aperture

        smear   [optional] same units as r. If smear != 0.0, paxel will
                be taken into account not abrupt but smootly according to 
                their distance to the circular sub aperture's center

                    /\ paxel weight
                     |
                1.0 -|-------\ 
                     |       .\ 
                     |       . \ 
                     |       .  \ 
                     |       .   \ 
                     |       .    \ 
                     |       .     \ 
                     |       .      \ 
                     |       .       \ 
                0.0 -|-------|--------|--------------->
                             r       r+smear       distance of paxel
                                                   to sub aperture center
        """
        return self.pixel_sum(
            Masks.circular(
                self.paxel_pos_x,
                self.paxel_pos_y,
                x=x,
                y=y,
                r=r,
                smear=smear
            )
        )

    def __repr__(self):
        out = 'LightField('
        out += str(self.number_lixel) + ' lixel = '
        out += str(self.number_pixel) + ' pixel x '
        out += str(self.number_paxel) + ' paxel, '
        out += 'Sum_Intensity = ' + \
            str(round(self.intensity.sum())) + ' p.e.)\n'
        return out
