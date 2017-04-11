import numpy as np
from ..tools import HeaderRepresentation

class PlenoscopeGeometry(object):
    """
    The Geometry of the Atmospheric Cherenkov Plenoscope (ACP).

    sensor_plane2imaging_system                 A homogenoues transformation to 
                                                describe the relative position
                                                and orientation of the imaging 
                                                system and the light field 
                                                sensor.

    sensor_plane_distance                       The distance of the light field
                                                sensor to the imaging system's 
                                                principal aperture plane along 
                                                the optical axis of the imaging     
                                                system.

    expected_imaging_system_focal_length        The focal length the light field    
                                                sensor was designed for.

    expected_imaging_system_max_aperture_radius The radius of the imaging 
                                                system's aperture radius the 
                                                light field sensor was designed
                                                for. 

    max_FoV_diameter                            The max. diameter of the Field 
                                                of View (FoV) of the Plenoscope.

    pixel_FoV_hex_flat2flat                     The FoV of a single pixel 
                                                hexagon (flat to flat).

    number_of_paxel_on_pixel_diagonal           The number of paxel on the 
                                                diagonal of the aperture.

    housing_overhead                            Overhead of the light field 
                                                sensor housing.
    """
    def __init__(self, raw):
        """
        Parameters
        ----------
        raw         The raw 273 float32 array.
        """
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