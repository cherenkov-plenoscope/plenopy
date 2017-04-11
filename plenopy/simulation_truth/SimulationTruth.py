import numpy as np
import os
from .. import Corsika

class SimulationTruth(object):
    """
    SimulationTruth
    """

    def __init__(self, event, air_shower_photon_bunches=None, detector=None):
        self.event = event
        self.air_shower_photon_bunches = air_shower_photon_bunches
        self.detector = detector

    def __repr__(self):
        out = 'SimulationTruth('
        out += 'primary: '+self.event.corsika_event_header.primary_particle+', '
        out += 'energy: '+str(self.event.corsika_event_header.total_energy_GeV)+'GeV'
        out += ')'
        return out