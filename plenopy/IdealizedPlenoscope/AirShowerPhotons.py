#!/usr/bin/env python
# coding: utf-8
import numpy as np

class AirShowerPhotons(object):
    """
    The CORSIKA air shower photon bunches

    x, y        x and y intersection position of photon on observation 
                plane/level, [m]

    cx, cy      incoming direction component of relative to the normal vector
                of the observation plane [rad]
                incomin_direction = [cx, cy, sqrt(1 - cx^2 - cy^2)]

    arrival_time_since_first_interaction            [s]

    emission_height                                 [m]

    wavelength                                      [m]
    """
    def __init__(self, photon_bunces):
        """
        Parameters
        ----------
        AirShowerPhotonBunches    the raw corsika photon bunches
        """

        number_of_bunches = photon_bunces.probability_to_reach_observation_level.shape[0]
        reached_observation_level = photon_bunces.probability_to_reach_observation_level > np.random.rand(number_of_bunches)

        self.x = photon_bunces.x[reached_observation_level]
        self.y = photon_bunces.y[reached_observation_level]

        self.cx = photon_bunces.cx[reached_observation_level]
        self.cy = photon_bunces.cy[reached_observation_level]

        self.arrival_time_since_first_interaction = photon_bunces.arrival_time_since_first_interaction[reached_observation_level]

        self.emission_height = photon_bunces.emission_height[reached_observation_level]
        self.wavelength = photon_bunces.wavelength[reached_observation_level]

    def __repr__(self):
        out = 'AirShowerPhotons( '
        out += str(self.x.shape[0])+' photons'
        out += ' )\n'
        return out