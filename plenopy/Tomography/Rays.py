import numpy as np

class Rays(object):
    """
    supports    [number_lixel x 3]
                Support vectors of all the lixel rays on the principal aperture
                plane. [x_mean, y_mean, 0]

    directions  [number_lixel x 3]
                Direction vectors of all the lixel rays.
                [cx_mean, cy_mean, sqrt(1 - cx_mean^2 - cy_mean^2)]
    """

    def __init__(self, xs, ys, z, cxs, cys):
        number_rays = xs.shape[0]
        self.supports = np.array([xs, ys, z*np.ones(number_rays)]).T
        dir_zs = np.sqrt(1.0 - cxs**2.0 - cys**2.0)
        # (cos_x, cos_y, sqrt(1 - cos_x^2 - cos_y^2))^T
        self.directions = np.array([cxs, cys, dir_zs]).T

    def xy_intersections_in_z(self, z):
        """
        Returns the x,y intersections of the rays with the x,y plane in z

        Parameters
        ----------
        z               The distance to the principal aperture plane.
        """
        scale_factors = z/self.directions[:, 2]
        pos3D = self.supports-(scale_factors*self.directions.T).T
        return pos3D[:, 0:2]

    def __repr__(self):
        out = 'Rays('
        out += str(self.supports.shape[0]) + ' rays'
        out += ')\n'
        return out
