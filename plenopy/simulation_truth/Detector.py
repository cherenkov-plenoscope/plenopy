import numpy as np
from . import mctracer_pulse_origins as po


class Detector(object):
    """
    The full simulation-truth on the origin of the pulses in a read-out-channel
    of the light-field-sensor.
    """
    def __init__(self, path):
        """
        Parameters
        ----------
        path        The input path to the additional detector
                    pulse origin binary file.
        """
        self.pulse_origins = np.fromfile(path, dtype=np.int32)

    def number_air_shower_pulses(self):
        return (self.pulse_origins >= 0).sum()

    def number_night_sky_background_pulses(self):
        return (self.pulse_origins == po.NIGHT_SKY_BACKGROUND).sum()

    def __repr__(self):
        out = self.__class__.__name__
        out += '('
        out += str(self.number_air_shower_pulses())+' air-shower pulses'
        out += ')'
        return out
