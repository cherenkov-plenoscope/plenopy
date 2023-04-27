import numpy as np
from .. import corsika
from ..tools import header273float32 as hr


class PhotonPropagator:
    """
    Additional truth known from the photon-propagator 'merlict'.
    """

    def __init__(self, raw_header):
        self.raw = raw_header
        hr.assert_shape_is_valid(self.raw)
        # hr.assert_marker_of_header_is(self.raw, 'MERL')

    def __repr__(self):
        out = "PhotonPropagator(MERL)"
        return out

    def random_seed(self):
        """
        The seed used in the merlict-simulation of this event.
        """
        return hr.interpret_bytes_from_float32_as_int32(self.raw[2 - 1])

    def nsb_exposure_start_time(self):
        """
        The start-time/s of the exposure-time-window applied by the
        merlict-simulation.
        """
        return self.raw[10 - 1] * 1e-9
