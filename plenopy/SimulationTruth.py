#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, print_function, division
import numpy as np
import os
from .HeaderRepresentation import corsika_event_header_repr
from .HeaderRepresentation import corsika_run_header_repr

class AirShowerPhotons(object):
    """
    The CORSIKA air shower photons as they were used for the simulation

    x, y        x and y intersection position of photon on observation 
                plane/level, [m]

    cx, cy      incoming direction component of relative to the normal vector
                of the observation plane [rad]
                incomin_direction = [cx, cy, sqrt(1 - cx^2 - cy^2)]

    time_since_first_interaction                    [s]

    emission_height                                 [m]

    wavelength                                      [m]

    probability_to_reach_observation_level          In CORSIKA this is the 
                                                    photon bunch weight. However
                                                    In the mctracer there are 
                                                    only single photons, so this 
                                                    weight is supposed to be
                                                    less equal 1.0 and encodes
                                                    the survival probability of
                                                    this photon to reach the 
                                                    observation level [1]
    """
    def __init__(self, path):
        """
        Parameters
        ----------
        path        The path to the air shower photon block of this event 
        """
        floats_per_photon = 8
        raw = np.fromfile(path, dtype=np.float32)
        raw = raw.reshape([raw.shape[0]/floats_per_photon, floats_per_photon])
        self.x = raw[:, 0]/100 # in meters
        self.y = raw[:, 1]/100 # in meters
        self.cx = raw[:, 2] # in rad
        self.cy = raw[:, 3] # in rad
        self.time_since_first_interaction = raw[:, 4]/1e9 # in seconds
        self.emission_height = raw[:, 5]/100 # in meters
        self.probability_to_reach_observation_level = raw[:, 6] # in [1]
        self.wavelength = raw[:, 7]/1e9 # in meters

    def __repr__(self):
        out = 'AirShowerPhotons( '
        out += str(self.x.shape[0])+' photons'
        out += ' )\n'
        return out


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
        self.__read_raw__(path)
        self.__init_channels()

    def __read_raw__(self, path):
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

    def __init_channels(self):

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
        self.corsika_event_header = self.__read_273_float_header(
            os.path.join(path, 'corsika_event_header.bin'))
        self.corsika_run_header = self.__read_273_float_header(
            os.path.join(path, 'corsika_run_header.bin'))

        self.__read_optional_air_shower_photons(
            os.path.join(path, 'air_shower_photons.bin'))
        self.__read_optional_intensity_truth(
            os.path.join(path, 'intensity_truth.txt'))

    def __read_optional_air_shower_photons(self, path):
        try:
            self.air_shower_photons = AirShowerPhotons(path)
        except(FileNotFoundError):
            pass

    def __read_optional_intensity_truth(self, path):
        try:
            self.intensity = Intensity(path)
            self.__init_intensity_truth_with_air_shower_truth()
        except(FileNotFoundError):
            pass

    def __init_intensity_truth_with_air_shower_truth(self):
        self.__intensity_air_shower_em_z = []

        for lix in range(len(self.intensity.lixel_intensities)):
            em_z = []
            for simulation_truth_id in self.intensity.lixel_intensities[lix]:
                if simulation_truth_id > self.intensity.MCTRACER_DEFAULT:
                    em_z.append(
                        self.air_shower_photons.emission_height[
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

    def __read_273_float_header(self, path):
        raw = np.fromfile(path, dtype=np.float32)
        return raw

    def __repr__(self):
        out = ''
        out += corsika_run_header_repr(self.corsika_run_header)
        out += '\n'
        out += corsika_event_header_repr(self.corsika_event_header)
        return out

    def short_event_info(self):
        """
        Return string       A short string to summarize the simulation truth.
                            Can be added in e.g. plot headings.
        """
        az = str(round(np.rad2deg(self.corsika_event_header[12 - 1]), 2))
        zd = str(round(np.rad2deg(self.corsika_event_header[11 - 1]), 2))
        core_y = str(round(0.01 * self.corsika_event_header[118], 2))
        core_x = str(round(0.01 * self.corsika_event_header[98], 2))
        E = str(round(self.corsika_event_header[4 - 1], 2))
        PRMPAR = self.corsika_event_header[3 - 1]
        run_id = str(int(self.corsika_run_header[2 - 1]))
        evt_id = str(int(self.corsika_event_header[2 - 1]))
        return str(
            "Run: " + run_id + ", Event: " + evt_id + ", " +
            CORSIKA_primary_ID2str(PRMPAR) + ', ' +
            "E: " + E + "GeV, \n" +
            "core pos: x=" + core_x + 'm, ' +
            "y=" + core_y + 'm, ' +
            "direction: Zd=" + zd + 'deg, ' +
            "Az=" + az + 'deg')


def CORSIKA_primary_ID2str(PRMPAR):
    """
    Return string   Convert the CORSIKA primary particle ID into a human 
                    readable string.

    Parameter
    ---------
    PRMPAR          The CORSIKA primary particle ID
    """
    PRMPAR = int(PRMPAR)
    if PRMPAR == 1:
        return 'gamma'
    elif PRMPAR == 2:
        return 'e^+'
    elif PRMPAR == 3:
        return 'e^-'
    elif PRMPAR == 5:
        return 'muon^+'
    elif PRMPAR == 6:
        return 'muon^-'
    elif PRMPAR == 7:
        return 'pion^0'
    elif PRMPAR == 8:
        return 'pion^+'
    elif PRMPAR == 9:
        return 'pion^-'
    elif PRMPAR == 14:
        return 'p'
    elif PRMPAR > 200:

        out = ''
        A = int(np.floor(PRMPAR / 100))
        Z = PRMPAR - A * 100
        if A == 4:
            out += 'He '
        elif A == 56:
            out += 'Fe '

        return out + 'A' + str(round(A)) + ' Z' + str(round(Z))
    else:
        return str(PRMPAR)