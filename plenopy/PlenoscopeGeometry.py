import numpy as np
from .tools import HeaderRepresentation

class PlenoscopeGeometry(object):
    """
    The Geometry of the Atmospheric Cherenkov Plenoscope.
    """
    def __init__(self, raw):
        HeaderRepresentation.assert_shape_is_valid(raw)
        
        self.sensor_plane2imaging_system = self._read(raw)
        self.sensor_plane_distance = self.sensor_plane2imaging_system[2, 3]

        self.expected_imaging_system_focal_length = raw[ 23-1]
        self.expected_imaging_system_max_aperture_radius = raw[ 24-1]

        self.max_FoV_diameter = raw[ 25-1]
        self.pixel_FoV_hex_flat2flat = raw[ 26-1]
        self.number_of_paxel_on_pixel_diagonal = raw[ 27-1]
        self.housing_overhead = raw[ 28-1]

    def _read(self, raw):
        return np.array([
            [raw[11 - 1], raw[14 - 1], raw[17 - 1], raw[20 - 1]],
            [raw[12 - 1], raw[15 - 1], raw[18 - 1], raw[21 - 1]],
            [raw[13 - 1], raw[16 - 1], raw[19 - 1], raw[22 - 1]],
            [0.0,       0.0,       0.0,       1.0],
        ])

    def __repr__(self):
        out = 'PlenoscopeGeometry('
        out += str(self.expected_imaging_system_focal_length) + ' focal length, '
        out += str(self.expected_imaging_system_max_aperture_radius*2) + ' mirror diameter'
        out += ')\n'
        return out