import numpy as np

class Binning3D(object):

    def __init__(
        self,
        z_min, z_max, number_z_bins,
        xy_diameter, number_xy_bins):

        self.z_min = z_min
        self.z_max = z_max
        self.number_z_bins = number_z_bins

        self.xy_diameter = xy_diameter
        self.number_xy_bins = number_xy_bins

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
