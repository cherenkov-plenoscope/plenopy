import numpy as np
import os
from .. import Corsika
from .LightField import LightField
from .IdealizedPlenoscopeSimulationTruthDetector import IdealizedPlenoscopeSimulationTruthDetector
from .. import SimulationTruth
import matplotlib.pyplot as plt

class Event(object):
    """
    Idealized Plenoscope Event

    number                      The event number in the run

    type                        A string as type indicator.
                                Idealized Plenoscope Events are always 
                                'simulation'

    light_field                 The air shower Cherenkov photons of an extensive
                                air shower which have passed the atmosphere and
                                reached the observation level seen by an
                                idealized light field sensor

    simulation_truth            Additional 'true' information known from the 
                                Corsika simulation itself
    """
    def __init__(self, path):
        self.path = path

        evth = Corsika.EventHeader(os.path.join(self.path, 'corsika_event_header.bin'))
        runh = Corsika.RunHeader(os.path.join(self.path, '../corsika_run_header.bin'))
        
        simulation_truth_event = SimulationTruth.Event(evth=evth, runh=runh)

        simulation_truth_air_shower_photon_bunches = Corsika.PhotonBunches(
                     os.path.join(self.path, 'air_shower_photon_bunches.bin'))

        self.light_field = LightField(
            simulation_truth_air_shower_photon_bunches) 

        simulation_truth_detector = IdealizedPlenoscopeSimulationTruthDetector(
            self.light_field._ids)

        self.simulation_truth = SimulationTruth.SimulationTruth(                
            event=simulation_truth_event,
            air_shower_photon_bunches=simulation_truth_air_shower_photon_bunches,
            detector=simulation_truth_detector)

        self.type = 'simulation'
        self.number = int(os.path.basename(self.path))

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
        plt.suptitle(self.simulation_truth.event.short_event_info())

        guess_number_bins = int(np.sqrt(self.light_field.x.shape[0]))

        axs[0].set_title('field of view intensity')
        axs[0].hist2d(
            np.rad2deg(self.light_field.cx), 
            np.rad2deg(self.light_field.cy),
            cmap='viridis',
            bins=guess_number_bins)
        axs[0].set_aspect('equal')
        axs[0].set_xlabel('cx/deg')
        axs[0].set_ylabel('cy/deg')

        axs[1].set_title('principal aperture plane intensity')
        axs[1].hist2d(
            self.light_field.x, 
            self.light_field.y,
            cmap='viridis',
            bins=int(guess_number_bins/4))
        axs[1].set_aspect('equal')
        axs[1].set_xlabel('x/m')
        axs[1].set_ylabel('y/m')

        plt.show()