import numpy as np
from .. import HeaderRepresentation

class EventHeader(object):
    """
    The CORSIKA event header
    """
    def __init__(self, path):
        """
        Parameters
        ----------
        path        The path to event header binary 
        """  

        self.raw = HeaderRepresentation.read_float32_header(path)
        HeaderRepresentation.assert_shape_is_valid(self.raw)
        HeaderRepresentation.assert_marker_of_header_is(self.raw, 'EVTH')

        self.event_number = int(self.raw[2 - 1])
        self.particle_id  = int(self.raw[3 - 1])
        self.total_energy_GeV = self.raw[4 - 1]

    def __repr__(self):
        out = 'CorsikaEventHeader( '
        out += ' )\n'
        return out