import numpy as np

class LightField(object):
    """
    The ideal Light Field

    x, y        x and y intersection position of photon on observation 
                plane/level, [m]

    cx, cy      incoming direction component of relative to the normal vector
                of the observation plane [rad]
                incomin_direction = [cx, cy, sqrt(1 - cx^2 - cy^2)]

    arrival_time            [s]

    wavelength              [m]
    """
    def __init__(self, photon_bunches, random_seed=0):
        """
        Parameters
        ----------
        PhotonBunches          the raw corsika photon bunches
        """

        reached_observation_level = self._collapse_probability_to_reach_observation_level(
            photon_bunches.probability_to_reach_observation_level,
            random_seed)

        self.x = photon_bunches.x[reached_observation_level]
        self.y = photon_bunches.y[reached_observation_level]

        self.cx = photon_bunches.cx[reached_observation_level]
        self.cy = photon_bunches.cy[reached_observation_level]

        self.arrival_time = photon_bunches.arrival_time_since_first_interaction[reached_observation_level]
        self.arrival_time-= self.arrival_time.min() #knows only relative arrival times

        self.wavelength = photon_bunches.wavelength[reached_observation_level]

        self.intensity = np.ones(reached_observation_level.sum())

    def _collapse_probability_to_reach_observation_level(
        self, 
        probability_to_reach_observation_level,
        random_seed):

        number_of_bunches = probability_to_reach_observation_level.shape[0]

        np.random.seed(random_seed)
        reached_observation_level = probability_to_reach_observation_level > np.random.rand(
            number_of_bunches)

        self._ids = np.arange(number_of_bunches)[reached_observation_level]

        return reached_observation_level

    def __repr__(self):
        out = 'IdealizedLightField( '
        out += str(self.x.shape[0])+' photons'
        out += ' )\n'
        return out