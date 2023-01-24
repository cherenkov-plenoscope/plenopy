import numpy as np
from .. import corsika
from ..tools import header273float32 as hr


class PhotonPropagator:
    """
    Additional truth known from the photon-propagator 'merlict'.
    """

    def __init__(self, raw_header):
        self.raw = raw_header
        if hr.str2int32("MERL") != self.raw[0]:
            print("Warning, expected 'MERL' marker.")

    def __repr__(self):
        out = "PhotonPropagator(MERL)"
        return out

    def random_seed(self):
        """
        The seed used in the merlict-simulation of this event.
        """
        b = np.array(self.raw[1 - 1], dtype="float32").tobytes()
        return hr.str2int32(b)

    def nsb_exposure_start_time(self):
        """
        The start-time/s of the exposure-time-window applied by the
        merlict-simulation.
        """
        return self.raw[10 - 1] * 1e-9
