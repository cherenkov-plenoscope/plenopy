import numpy as np

class AirShowerPhotonBunches(object):
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
        self.arrival_time_since_first_interaction = raw[:, 4]/1e9 # in seconds
        self.emission_height = raw[:, 5]/100 # in meters
        self.probability_to_reach_observation_level = raw[:, 6] # in [1]
        self.wavelength = raw[:, 7]/1e9 # in meters

    def __repr__(self):
        out = 'AirShowerPhotonBunches( '
        out += str(self.x.shape[0])+' bunches, '
        out += str(self.probability_to_reach_observation_level.sum())+' photons'
        out += ' )\n'
        return out