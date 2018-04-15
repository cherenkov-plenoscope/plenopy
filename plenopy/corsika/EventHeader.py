import numpy as np
from ..tools import HeaderRepresentation
from .utils import prmpar_repr


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

        self.number = int(self.raw[2 - 1])
        self.primary_particle_id = int(self.raw[3 - 1])
        self.primary_particle = prmpar_repr(
            self.primary_particle_id)
        self.total_energy_GeV = self.raw[4 - 1]
        assert self.number_of_reuses() == 1

    def momentum(self):
        return self.raw[8-1 : 11-1]

    def zenith_angle_theta_rad(self):
        return self.raw[11 - 1]

    def azimuth_angle_phi_rad(self):
        return self.raw[12 - 1]

    def number_of_reuses(self):
        return int(self.raw[98 - 1])

    def core_position_x_meter(self):
        return self.raw[98 - 1 + 1]/1e2

    def core_position_y_meter(self):
        return self.raw[118 - 1 + 1]/1e2

    def __repr__(self):
        out = self.__class__.__name__
        out += '('
        out += 'number: '+str(self.number)+', '
        out += 'primary: '+self.primary_particle+', '
        out += 'energy: '+str(self.total_energy_GeV)+'GeV'
        out += ' )'
        return out
