import numpy as np

class Binning(object):
    """
    Binning of the 3D atmospheric detector volume

    The binned volume is defined in the frame of the principal aperture plane, 
    i.e. distances in positive z correspond to positve object distances. 

                                          
                                         /\z-axis
                                          |
                                          |
       z_max    ..........................|...........................
                .    .                    |                          .
                .    .  voxel / bin       |                          .
                .    .                    |                          .
                ......                    |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
                .                         |                          .
       z_min    ..........................|...........................
                \_________________________|_____  ___________________/
                                          |     \/
                                          |     xy_diameter
                 principal aperture plane |
    _xy-plane,_z=0_________|--------------|--------------|___observation_level__
    """
    def __init__(
        self,
        z_min, z_max, number_z_bins,
        xy_diameter, number_xy_bins):
        """
        Parameters
        ----------
        z_min               Lower starting height of histogram above principal 
                            aperture plane

        z_max               Upper ending height of histogram above principal 
                            aperture plane

        number_z_bins       Number of bins along the z-axis 

        xy_diameter         Edge length of the quadratic histogram in x and y.
                            The histogram is centered around the z-axis

        number_xy_bins      Number of bins along the x- and y-axis 
        """

        self.z_min = z_min
        self.z_max = z_max
        self.number_z_bins = number_z_bins

        self.xy_diameter = xy_diameter
        self.number_xy_bins = number_xy_bins

        self.number_bins =  self.number_z_bins*self.number_xy_bins**2

        self.xy_bin_edges = np.linspace(
            -xy_diameter/2.0, 
            xy_diameter/2.0, 
            number_xy_bins+1)

        self.z_bin_edges = np.linspace(
            z_min, 
            z_max, 
            number_z_bins+1)

        self.voxel_xy_radius = 0.5*xy_diameter/number_xy_bins
        self.voxel_z_radius = 0.5*(z_max - z_min)/number_z_bins

        self.xy_bin_centers = self.xy_bin_edges[:-1]+self.voxel_xy_radius
        self.z_bin_centers = self.z_bin_edges[:-1]+self.voxel_z_radius

        self.total_volume = (z_max - z_min)*xy_diameter**2.0

        self.voxel_xy_area = (2.0*self.voxel_xy_radius)**2.0
        self.voxel_volume = self.voxel_xy_area*(2.0*self.voxel_z_radius)

    def __repr__(self):
        out = 'Binning( '
        out += str(self.total_volume/1e9)+'km^3 from '
        out += 'z_min='+str(self.z_min)+'m to z_max='+str(self.z_max)+'m and '
        out += 'xy_diameter='+str(self.xy_diameter)+'m in '
        out += str(self.number_xy_bins)+' x '+str(self.number_xy_bins)+' x '+str(self.number_z_bins)+' bins'
        out += ' )\n'
        return out
