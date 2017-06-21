import numpy as np


class Rays(object):
    """
    support     [number_lixel x 3]
                Support vectors of all the lixel rays on the principal aperture
                plane. [x_mean, y_mean, 0]

    direction   [number_lixel x 3]
                Direction vectors of all the lixel rays.
                [cx_mean, cy_mean, sqrt(1 - cx_mean^2 - cy_mean^2)]
    """

    def __init__(self, x, y, cx, cy):
        """
        Parameters
        ----------

        x, y        The x and y intersections of the rays on the principal
                    aperture plane.

        cx, cy      The cos x and cos y direction components of the rays
                    relative to the principal aperture plane.
        """
        number_lixel = x.shape[0]
        self.support = np.array([x, y, np.zeros(number_lixel)]).T
        dir_z = np.sqrt(1.0 - cx**2.0 - cy**2.0)
        # (cos_x, cos_y, sqrt(1 - cos_x^2 - cos_y^2))^T
        self.direction = np.array([cx, cy, dir_z]).T

    @classmethod
    def from_light_field_geometry(cls, light_field_geometry):
        return cls(
            x=light_field_geometry.x_mean,
            y=light_field_geometry.y_mean,
            cx=light_field_geometry.cx_mean,
            cy=light_field_geometry.cy_mean)

    def xy_intersections_in_object_distance(self, object_distance):
        """
        Returns the x,y intersections of the lixel rays with the x,y plane
        at z=object_distance.

        Parameters
        ----------
        object_distance     The distance to the principal aperture plane.
                            scalar or 1D array.

        Returns
        --------

        intersections:

        if `object_distance` is a scalar returns a 2D array (M, 2)
            M: number of rays
        else it returns a 2D array (M, N, 2)
            M: number of rays
            N: number of object distances
        """
        object_distance = np.atleast_3d(object_distance)
        scale_factors = object_distance / self.direction[:, 2]
        pos3D = self.support[:, None, :] - (
            scale_factors * self.direction.T[:, None, :]).T
        if object_distance.shape[1] == 1:
            return pos3D[:, 0, 0:2]
        else:
            return pos3D[..., 0:2]

    def intersections_with_xy_plane(self, object_distance):
        """
        Returns the x,y,z intersections of the lixel rays with the x,y plane
        at z=object_distance.

        Parameters
        ----------
        object_distance     The distance to the principal aperture plane.
                            scalar or 1D array.

        Returns
        --------

        intersections a 2D array (M, N, 3)
            M: number of rays
            N: number of object distances
        """
        object_distance = np.atleast_3d(object_distance)
        scale_factors = object_distance / self.direction[:, 2]
        pos3D = self.support[:, None, :] - (
            scale_factors * self.direction.T[:, None, :]).T
        pos3D[..., 2] *= -1
        return pos3D

    def __repr__(self):
        out = 'Rays('
        out += str(self.support.shape[0]) + ' lixels'
        out += ')'
        return out
