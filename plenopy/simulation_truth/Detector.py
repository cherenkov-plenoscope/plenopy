import numpy as np
import os


class Detector(object):
    MCTRACER_DEFAULT = -1
    NIGHT_SKY_BACKGROUND = -100
    PHOTO_ELECTRIC_CONVERTER_ACCIDENTAL = - 201
    PHOTO_ELECTRIC_CONVERTER_CROSSTALK = - 202
    """
    The full simulation truth on the origin of the pulses in a read out channel
    of the light field sensor.
    """
    def __init__(self, light_field, detector_pulse_origin_path):
        """
        Parameters
        ----------
        light_field

        detector_pulse_origin_path  The input path to the additional detector
                                    pulse origin binary file.
        """
        self.pulse_origins = np.fromfile(
            detector_pulse_origin_path, dtype=np.int32)

    def lixel_wise_pulse_origins():
        stream = []
        i = 0
        for lixel in range(light_field.number_lixel):
            pulses_in_lixel = []
            for pulse in range(light_field.sequence[:, lixel].sum()):
                pulses_in_lixel.append(self.pulse_origins[i])
                i += 1
            stream.append(np.array(pulses_in_lixel, dtype=np.int32))
        return stream

    def number_air_shower_pulses(self):
        return (self.pulse_origins >= 0).sum()

    def __repr__(self):
        out = self.__class__.__name__
        out += '('
        out += str(self.number_air_shower_pulses())+' air-shower pulses'
        out += ')'
        return out
