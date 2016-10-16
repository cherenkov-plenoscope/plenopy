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

    def __init__(self, light_field):

        # All in principal aperture frame
        x = light_field.x_mean.flatten()
        y = light_field.y_mean.flatten()

        cx = light_field.cx_mean.flatten()
        cy = light_field.cy_mean.flatten()

        self._f = light_field.expected_focal_length_of_imaging_system
        bs = light_field.sensor_plane2imaging_system.sensor_plane_distance

        number_lixel = x.shape[0]

        # 3d intersection with image sensor plane
        img = np.array([
            bs*np.tan(cx), 
            bs*np.tan(cy), 
            bs*np.ones(number_lixel)]).T

        self.support = np.array([x, y, np.zeros(number_lixel)]).T
        
        self.direction = img - self.support
        no = np.linalg.norm(self.direction, axis=1)
        self.direction[:,0] /= no
        self.direction[:,1] /= no
        self.direction[:,2] /= no


    def cx_cy_in_object_distance(self, object_distance):
        """
        Returns the cx and cy pixel directions when refocussing to the desired
        object_distance.

        Parameters
        ----------
        object_distance     Objects in this distance are mapped sharp inthe image.
        """

        image_distance = 1/(1/self._f - 1/object_distance)

        scale_factors = image_distance/self.direction[:, 2]
        pos = self.support + (scale_factors * self.direction.T).T
        ix = pos[:,0]
        iy = pos[:,1]

        cx = np.arctan(ix/image_distance)
        cy = np.arctan(iy/image_distance)
        return cx, cy

    def __repr__(self):
        out = 'ImageRays('
        out += str(self.support.shape[0]) + ' image rays'
        out += ')\n'
        return out
