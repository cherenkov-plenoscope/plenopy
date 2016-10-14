import numpy as np
from .. import HeaderRepresentation

class RunHeader(object):
    """
    The CORSIKA run header
    """
    def __init__(self, path):
        """
        Parameters
        ----------
        path        The path to event header binary 
        """  

        self.raw = HeaderRepresentation.read_float32_header(path)
        HeaderRepresentation.assert_shape_is_valid(self.raw)
        HeaderRepresentation.assert_marker_of_header_is(self.raw, 'RUNH')

        self.number = int(self.raw[2 - 1])
        self.xscatt = self.raw[248 - 1]/100 # cm -> m
        self.yscatt = self.raw[249 - 1]/100 # cm -> m

    def __repr__(self):
        out = 'CorsikaRunHeader( '
        out +='number '+str(self.number)
        out += ' )\n'
        return out