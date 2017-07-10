import numpy as np
import matplotlib.pyplot as plt
from .plot import add_pixel_image_to_ax


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

    def plot(self):
        fig, ax = plt.subplots()
        add_pixel_image_to_ax(self, ax)
        plt.show()

    def __repr__(self):
        out = 'Image('
        out += str(self.intensity.shape[0]) + ' channels, '
        out += 'Sum_intensity ' + str(round(self.intensity.sum())) + ' p.e.'
        out += ')'
        return out
