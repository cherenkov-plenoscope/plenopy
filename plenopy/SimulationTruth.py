#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, print_function, division
import numpy as np
import os
from . import Corsika

class Intensity(object):
    MCTRACER_DEFAULT = -1
    NIGHT_SKY_BACKGROUND = -100;
    PHOTO_ELECTRIC_CONVERTER_ACCIDENTAL = - 201;
    PHOTO_ELECTRIC_CONVERTER_CROSSTALK = - 202;
    """
    The full simulation truth on the origin of the pulses in a read out channel
    of the light field sensor.
    """
    def __init__(self, path):
        """
        Parameters
        ----------
        path    The input path to a the simulation truth intensity text file.
        """
        self._read_raw(path)
        self._init_channels()

    def _read_raw(self, path):
        self.lixel_intensities = []
        with open(path, 'r') as handle:
            for line in handle:
                if line != "\n": 
                    self.lixel_intensities.append(
                        np.fromstring(line, sep=' ', dtype=int))
                else:
                    self.lixel_intensities.append(
                        np.zeros(0, dtype=int))

        self.number_lixel = len(self.lixel_intensities)

    def _init_channels(self):

        self.air_shower = np.zeros(self.number_lixel)
        self.night_sky_background = np.zeros(self.number_lixel)
        self.photo_electric_accidental = np.zeros(self.number_lixel)
        self.photo_electric_crosstalk = np.zeros(self.number_lixel)

        for i, lixel_intensity in enumerate(self.lixel_intensities):
            self.air_shower[i] = np.count_nonzero(
                lixel_intensity > self.MCTRACER_DEFAULT)
            self.night_sky_background[i] = np.count_nonzero(
                lixel_intensity == self.NIGHT_SKY_BACKGROUND)
            self.photo_electric_accidental[i] = np.count_nonzero(
                lixel_intensity == self.PHOTO_ELECTRIC_CONVERTER_ACCIDENTAL)
            self.photo_electric_crosstalk[i] = np.count_nonzero(
                lixel_intensity == self.PHOTO_ELECTRIC_CONVERTER_CROSSTALK)

    def __repr__(self):
        out = 'Intensity( '
        out += str(self.number_lixel)+' lixels'
        out += ' )\n'
        return out


class SimulationTruth(object):
    """
    Additional truth known from the simulation itself

    CORSIKA run header      [float 273] The raw run header

    CORSIKA event header    [float 273] The raw event header

    intensity               [only if present in event] 
                            The detailed simulation truth of the pulses which
                            contribute to the intensity found in a read out 
                            channel (lixel).
    """

    def __init__(self, path):
        self.corsika_event_header = self._read_273_float_header(
            os.path.join(path, 'corsika_event_header.bin'))
        self.corsika_run_header = self._read_273_float_header(
            os.path.join(path, 'corsika_run_header.bin'))

        self._read_optional_air_shower_photon_bunches(
            os.path.join(path, 'air_shower_photons.bin'))
        self._read_optional_intensity_truth(
            os.path.join(path, 'intensity_truth.txt'))

    def _read_optional_air_shower_photon_bunches(self, path):
        try:
            self.air_shower_photon_bunches = Corsika.AirShowerPhotonBunches(path)
        except(FileNotFoundError):
            pass

    def _read_optional_intensity_truth(self, path):
        try:
            self.intensity = Intensity(path)
            self._init_intensity_truth_with_air_shower_truth()
        except(FileNotFoundError):
            pass

    def _init_intensity_truth_with_air_shower_truth(self):
        self.__intensity_air_shower_em_z = []

        for lix in range(len(self.intensity.lixel_intensities)):
            em_z = []
            for simulation_truth_id in self.intensity.lixel_intensities[lix]:
                if simulation_truth_id > self.intensity.MCTRACER_DEFAULT:
                    em_z.append(
                        self.air_shower_photon_bunches.emission_height[
                            simulation_truth_id
                        ]
                    )
            em_z = np.array(em_z)
            self.__intensity_air_shower_em_z.append(em_z)

    def intensity_for_production_height(self, min_height, max_height):
        intensity = np.zeros(len(self.__intensity_air_shower_em_z))

        for lix, photon_emission_height in enumerate(self.__intensity_air_shower_em_z):
            intensity[lix] = np.logical_and(
                np.count_nonzero(photon_emission_height > min_height),
                np.count_nonzero(photon_emission_height <= max_height)
            )
        return intensity

    def _read_273_float_header(self, path):
        raw = np.fromfile(path, dtype=np.float32)
        return raw

    def __repr__(self):
        out = ''
        out += Corsika.run_header_repr(self.corsika_run_header)
        out += '\n'
        out += Corsika.event_header_repr(self.corsika_event_header)
        return out

    def short_event_info(self):
        return Corsika.short_event_info(
            self.corsika_run_header,
            self.corsika_event_header)