import numpy as np


class Image:
    """
    A 2D Images to display the classic IACT images and the intensity
    distributions on the principal aperture plane.

    Parameters
    ----------
    intensity   The photon equivalent intensity in each channel [p.e.]

    pixel_pos_x, pixel_pos_y    The x and y position of the channels
                                [either m or rad]
    """

    def __init__(self, intensity, positions_x, positions_y):
        self.intensity = intensity
        self.pixel_pos_x = positions_x
        self.pixel_pos_y = positions_y

    def __repr__(self):
        out = self.__class__.__name__
        out += "("
        out += str(self.intensity.shape[0]) + " channels, "
        out += "Sum_intensity " + str(round(self.intensity.sum())) + " p.e."
        out += ")"
        return out
