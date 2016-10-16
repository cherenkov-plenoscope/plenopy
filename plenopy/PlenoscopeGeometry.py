import numpy as np

class PlenoscopeGeometry(object):

    def __init__(self, event_header):
        self.sensor_plane2imaging_system = self._read(event_header)
        self.sensor_plane_distance = self.sensor_plane2imaging_system[2, 3]

        self.expected_imaging_system_focal_length = event_header[ 23-1]
        self.expected_imaging_system_max_aperture_radius = event_header[ 24-1]

        self.max_FoV_diameter = event_header[ 25-1]
        self.pixel_FoV_hex_flat2flat = event_header[ 26-1]
        self.number_of_paxel_on_pixel_diagonal = event_header[ 27-1]
        self.housing_overhead = event_header[ 28-1]

    def _read(self, gh):
        return np.array([
            [gh[11 - 1], gh[14 - 1], gh[17 - 1], gh[20 - 1]],
            [gh[12 - 1], gh[15 - 1], gh[18 - 1], gh[21 - 1]],
            [gh[13 - 1], gh[16 - 1], gh[19 - 1], gh[22 - 1]],
            [0.0,       0.0,       0.0,       1.0],
        ])
