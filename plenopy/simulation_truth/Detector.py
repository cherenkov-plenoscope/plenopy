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

        pulse_IDs = np.fromfile(detector_pulse_origin_path, dtype=np.int32)
        self.stream = []

        i = 0
        for lixel in range(light_field.number_lixel):
            pulses_in_lixel = []
            for pulse in range(light_field.sequence[:,lixel].sum()):
                pulses_in_lixel.append(pulse_IDs[i])
                i += 1
            self.stream.append(
                np.array(pulses_in_lixel, dtype=np.int32))


    def number_air_shower_pulses(self):
        counter = 0
        for lixel in self.stream:
            counter += (lixel>=0).sum()
        return counter


    def __repr__(self):
        out = 'SimulationTruthDetector('
        out += str(self.number_air_shower_pulses())+' air-shower pulses'
        out += ')'
        return out