import numpy as np


class ImageRays(object):
    """
    support     [number_lixel x 3]
                Support vectors of all the lixel rays on the principal aperture
                plane. [x_mean, y_mean, 0]

    direction   [number_lixel x 3]
                Direction vectors of all the image rays after passing the
                imaging system.
    """

    def __init__(self, light_field_geometry):
        """
        Parameters
        ----------

        light_field_geometry
        """
        lfg = light_field_geometry
        # All in principal aperture frame
        x = lfg.x_mean
        y = lfg.y_mean
        cx = lfg.cx_mean
        cy = lfg.cy_mean
        self._f = lfg.expected_focal_length_of_imaging_system
        bs = lfg.sensor_plane2imaging_system.sensor_plane_distance
        # 3d intersection with image sensor plane
        sensor_plane_intersections = np.array([
            -bs*np.tan(cx),
            -bs*np.tan(cy),
            bs*np.ones(lfg.number_lixel)]).T
        self.support = np.array([
            x,
            y,
            np.zeros(lfg.number_lixel)]).T
        self.direction = sensor_plane_intersections - self.support
        no = np.linalg.norm(self.direction, axis=1)
        self.direction[:, 0] /= no
        self.direction[:, 1] /= no
        self.direction[:, 2] /= no
        self.pixel_pos_tree = lfg.pixel_pos_tree

    def cx_cy_in_object_distance(self, object_distance):
        """
        Returns the cx and cy pixel intersections of the image-rays when
        refocussing to the desired object-distance.

        Parameters
        ----------
        object_distance     The distance from the aperture to focus on.
        """
        image_distance = 1/(1/self._f - 1/object_distance)
        scale_factors = image_distance/self.direction[:, 2]
        pos = self.support + (scale_factors * self.direction.T).T
        ix = pos[:, 0]
        iy = pos[:, 1]
        cx = -np.arctan(ix/image_distance)
        cy = -np.arctan(iy/image_distance)
        return cx, cy

    def pixel_ids_of_lixels_in_object_distance(self, object_distance):
        cx, cy = self.cx_cy_in_object_distance(object_distance)
        cxy = np.vstack((cx, cy)).T
        number_nearest_neighbors = 1
        distances, pixel_indicies = self.pixel_pos_tree.query(
            x=cxy,
            k=number_nearest_neighbors)
        return pixel_indicies

    def __repr__(self):
        out = self.__class__.__name__
        out += '('
        out += str(self.support.shape[0]) + ' image rays'
        out += ')'
        return out
