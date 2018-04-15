import numpy as np
from ..tools import header273float32 as hr


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
        self.raw = hr.read_float32_header(path)
        hr.assert_shape_is_valid(self.raw)
        hr.assert_marker_of_header_is(self.raw, 'RUNH')
        self.number = int(self.raw[2 - 1])
        self.xscatt = self.raw[248 - 1]/100  # cm -> m
        self.yscatt = self.raw[249 - 1]/100  # cm -> m

    def observation_level(self):
        number_observation_levels = self.raw[5 - 1]
        assert number_observation_levels == 1
        return self.raw[6 - 1]/1e2

    def __repr__(self):
        out = self.__class__.__name__
        out += '('
        out += 'number '+str(self.number)
        out += ')'
        return out
