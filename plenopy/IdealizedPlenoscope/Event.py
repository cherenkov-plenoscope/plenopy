import numpy as np
import os
from .. import Corsika
from .AirShowerPhotons import AirShowerPhotons
from .SimulationTruth import SimulationTruth
import matplotlib.pyplot as plt

class Event(object):
    """
    Idealized Plenoscope Event

    number                      The event number in the run

    type                        A string as type indicator.
                                Idealized Plenoscope Events are always 
                                'simulation'

    air_shower_photons          The air shower Cherenkov photons of an extensive
                                air shower which have passed the atmosphere and
                                reached the observation level

    simulation_truth            Additional 'true' information known from the 
                                Corsika simulation itself
    """
    def __init__(self, path):
        self.path = path
        self._init_air_shower_photons()
        self._init_simulation_truth()
        self.type = 'simulation'
        self.number = int(os.path.basename(self.path))

    def _init_air_shower_photons(self):
        bunches = Corsika.AirShowerPhotonBunches(
            os.path.join(self.path, 'air_shower_photons.bin'))
        self.air_shower_photons = AirShowerPhotons(bunches)        

    def _init_simulation_truth(self):
        evth = Corsika.EventHeader(os.path.join(self.path, 'corsika_event_header.bin'))
        runh = Corsika.RunHeader(os.path.join(self.path, '../corsika_run_header.bin'))
        ids = self.air_shower_photons.ids
        self.simulation_truth = SimulationTruth(evth, runh, ids)

    def __repr__(self):
        out = 'IdealizedPlenoscopeEvent( '
        out += "path='" + self.path
        out += ' )\n'
        return out

    def plot(self):
        """
        This will open a plot showing:

        1   Directional intensity distribution accross the field of view
            (the classical IACT image)

        2   Positional intensity distribution on the principal aperture plane
        """
        fig, axs = plt.subplots(2)
        plt.suptitle(self.simulation_truth.short_event_info())

        guess_number_bins = int(np.sqrt(self.air_shower_photons.x.shape[0]))

        axs[0].set_title('directional image')
        axs[0].hist2d(
            np.rad2deg(self.air_shower_photons.cx), 
            np.rad2deg(self.air_shower_photons.cy),
            cmap='viridis',
            bins=guess_number_bins)
        axs[0].set_aspect('equal')
        axs[0].set_xlabel('cx/deg')
        axs[0].set_ylabel('cy/deg')

        axs[1].set_title('principal aperture plane')
        axs[1].hist2d(
            self.air_shower_photons.x, 
            self.air_shower_photons.y,
            cmap='viridis',
            bins=int(guess_number_bins/4))
        axs[1].set_aspect('equal')
        axs[1].set_xlabel('x/m')
        axs[1].set_ylabel('y/m')

        plt.show()