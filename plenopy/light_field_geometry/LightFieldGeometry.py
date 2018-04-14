import numpy as np
import scipy.spatial
from scipy.constants import speed_of_light
import os
from .PlenoscopeGeometry import PlenoscopeGeometry
from ..tools.HeaderRepresentation import assert_marker_of_header_is
from ..tools.HeaderRepresentation import read_float32_header
from . import isochor_image

class LightFieldGeometry(object):
    """
    number_lixel    The number of light field cells (lixel)
                    This is also the number of read out channels.

    number_pixel    The number of directional bins forming the picture cells.
                    Each pixel is composed of number_paxel paxel.

    number_paxel    The number of positional bins forming the principal
                    aperture cells (paxel) on the principal aperture plane.

    efficiency      [number_lixel]
                    The average efficiency of a lixel. During
                    the calibration, photons are thrown
                    into the plenoscope. The photons are evenly
                    spread over the aperture and the field of view.
                    The more photons reached the lixel sensor, the
                    higher is its efficiency. This takes losses into
                    account except for the photo-electric sensor
                    efficiency. [1]

    x_mean, x_std,
    y_mean, y_std   [number_lixel]
                    The average x,y position and its spread of positional bins
                    on the principal aperture plane. [m]

    cx_mean, cx_std,
    cy_mean, cy_std [number_lixel]
                    The average cos_x direction and its spread of the
                    directional bins in the field of view [rad]

                    The cx or cy is short for cos_x or cos_y. cx and cy are
                    the x and y components of the normalized incomming
                    direction vector vec{d} on the principal aperture plane.

                    vec{d} = (cx, cy, sqrt(1 - cx^2 - cy^2))

    time_delay_mean,
    time_delay_std  [number_lixel]
                    The average arrival time delay and its spread for
                    a photon to travel from the principal aperture
                    plane to the lixel sensor. [s]

    time_delay_image_mean,
    time_delay_image_std    [number_lixel]
                            The time-delay which needs to be added to each
                            lixel so that the photons of a light-front
                            coming from direction cx, cy arrive isochor in
                            pixel-cx-cy.

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

        self._read_light_field_sensor_geometry_header(
            os.path.join(path, 'light_field_sensor_geometry.header.bin'))

        self._read_lixel_positions(
            os.path.join(path, 'lixel_positions.bin'))

        self._read_lixel_statistics(
            os.path.join(path, 'lixel_statistics.bin'))

        self._calc_pixel_and_paxel_average_positions()
        self._init_lixel_polygons()

        # self.valid_efficiency = self.efficiency > 0.10
        self.valid_efficiency = self.most_efficient_lixels(0.95)
        self._init_time_delay_image()

    def _calc_pixel_and_paxel_average_positions(self):
        npix = self.number_pixel
        npax = self.number_paxel
        self.paxel_pos_x = np.nanmean(
            self.x_mean.reshape(npix, npax),
            axis=0)
        self.paxel_pos_y = np.nanmean(
            self.y_mean.reshape(npix, npax),
            axis=0)

        self.pixel_pos_cx = np.nanmean(
            self.cx_mean.reshape(npix, npax),
            axis=1)
        self.pixel_pos_cy = np.nanmean(
            self.cy_mean.reshape(npix, npax),
            axis=1)

        self.pixel_pos_tree = scipy.spatial.cKDTree(
            np.array([self.pixel_pos_cx, self.pixel_pos_cy]).T)
        self.paxel_pos_tree = scipy.spatial.cKDTree(
            np.array([self.paxel_pos_x, self.paxel_pos_y]).T)

        self.paxel_efficiency_along_pixel = np.nanmean(
            self.efficiency.reshape(npix, npax),
            axis=0)
        self.pixel_efficiency_along_paxel = np.nanmean(
            self.efficiency.reshape(npix, npax),
            axis=1)

    def _read_lixel_statistics(self, path):
        ls = np.fromfile(path, dtype=np.float32)
        ls = ls.reshape([ls.shape[0] // 12, 12])

        self.efficiency = ls[:, 0].copy()
        self.efficiency_std = ls[:, 1].copy()

        self.cx_mean = ls[:, 2].copy()
        self.cx_std = ls[:, 3].copy()
        self.cy_mean = ls[:, 4].copy()
        self.cy_std = ls[:, 5].copy()

        self.x_mean = ls[:, 6].copy()
        self.x_std = ls[:, 7].copy()
        self.y_mean = ls[:, 8].copy()
        self.y_std = ls[:, 9].copy()

        self.time_delay_mean = ls[:, 10].copy()
        self.time_delay_std = ls[:, 11].copy()

    def _read_light_field_sensor_geometry_header(self, path):
        gh = read_float32_header(path)
        assert_marker_of_header_is(gh, 'PLGH')
        self.sensor_plane2imaging_system = PlenoscopeGeometry(gh)

        self.number_pixel = int(gh[101 - 1])
        self.number_paxel = int(gh[102 - 1])
        self.number_lixel = self.number_pixel * self.number_paxel

        self.lixel_outer_radius = gh[103 - 1]
        self.lixel_z_orientation = gh[105 - 1]

        self.expected_focal_length_of_imaging_system = gh[23 - 1]
        self.expected_aperture_radius_of_imaging_system = gh[24 - 1]

    def _read_lixel_positions(self, path):
        lp = np.fromfile(path, dtype=np.float32)
        lp = lp.reshape([lp.shape[0] // 2, 2])
        self.lixel_positions_x = lp[:, 0]
        self.lixel_positions_y = lp[:, 1]

    def _init_lixel_polygons(self):
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
        idxs = np.argsort(self.efficiency)[-number_valid_lixels:]
        mask = np.zeros(self.number_lixel, dtype=bool)
        mask[idxs] = True
        return mask

    def _init_time_delay_image(self):
        # distances d from pap to plane of isochor light-front
        d_mean, d_std  = isochor_image.relative_path_length_for_isochor_image(
            cx_mean=self.cx_mean, cx_std=self.cx_std,
            cy_mean=self.cy_mean, cy_std=self.cy_std,
            x_mean=self.x_mean, x_std=self.x_std,
            y_mean=self.y_mean, y_std=self.y_std)

        t_mean = d_mean/speed_of_light
        t_std = d_std/speed_of_light

        self.time_delay_image_mean = -self.time_delay_mean + t_mean
        self.time_delay_image_mean -= np.min(self.time_delay_image_mean)
        self.time_delay_image_std = np.sqrt(
            self.time_delay_std**2 + t_std**2)

    def __repr__(self):
        out = self.__class__.__name__
        out += '('
        out += str(self.number_lixel) + ' lixel, '
        out += str(self.number_pixel) + ' pixel, '
        out += str(self.number_paxel) + ' paxel'
        out += ')'
        return out
