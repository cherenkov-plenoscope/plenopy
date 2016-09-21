import numpy as np
import os
from .. import Corsika

class SimulationTruth(object):
    """
    Idealized Plenoscope Event
    """
    def __init__(self, evth, runh, truth_ids):
        self.corsika_event_header = evth
        self.corsika_run_header = runh
        self.simulation_truth_ids = truth_ids

    def __repr__(self):
        out = 'IdealizedPlenoscopeSimulationTruth('
        out += ')\n'
        return out

    def short_event_info(self):
        return Corsika.short_event_info(
            self.corsika_run_header.raw,
            self.corsika_event_header.raw)