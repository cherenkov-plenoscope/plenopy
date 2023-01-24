import numpy as np
from .. import corsika


class SimulationTruth(object):
    """
    SimulationTruth
    """

    def __init__(
        self,
        event,
        air_shower_photon_bunches=None,
        detector=None,
        photon_propagator=None,
    ):
        self.event = event
        self.air_shower_photon_bunches = air_shower_photon_bunches
        self.detector = detector
        self.photon_propagator = photon_propagator

    def __repr__(self):
        out = self.__class__.__name__
        out += "("
        out += "primary: " + self.event.corsika_event_header.primary_particle
        out += ", "
        out += "energy: " + str(
            self.event.corsika_event_header.total_energy_GeV
        )
        out += "GeV"
        out += ")"
        return out
