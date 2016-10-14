import numpy as np
from .. import HeaderRepresentation
from .primary_particle_id2str import primary_particle_id2str

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
        self.primary_particle_id  = int(self.raw[3 - 1])
        self.primary_particle  = primary_particle_id2str(self.primary_particle_id)
        self.total_energy_GeV = self.raw[4 - 1]

    def __repr__(self):
        out = 'CorsikaEventHeader( '
        out += 'number: '+str(self.number)+', '
        out += 'primary: '+self.primary_particle+', '
        out += 'energy: '+str(self.total_energy_GeV)+'GeV'
        out += ' )\n'
        return out