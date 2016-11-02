import numpy as np
from .filtered_back_projection import histogram 
from .filtered_back_projection import normalize_ray_histograms 
from .filtered_back_projection import ramp_kernel_in_frequency_space 
from .filtered_back_projection import frequency_filter 
from ..LixelRays import LixelRays 

class AirShowerReconstruction(object):
    """
    Tomographic Air Shower Reconstruction
    using filtered back projection

    Member
    ------

    rays            The lixel rays defining the back projection geometry

    intensities     The intensities of the lixel rays (photon eqivalent)

    binning         The 3D binning of the atmospheric detector volume above the
                    principal aperture plane

    intensity_hist  3D back projection histogram of the lixel rays weighted with
                    the intensities of each ray.

    ray_count_hist  3D back projection histogram of the only the lixel rays.
                    Here the intensity is the number count of rays in a voxel of
                    the 3D histogram

    normalized_intensity_hist   3D back projection histogram of the weighted
                                lixel rays, but correctod for the asymetric 
                                sampling of rays in the voxels.

    filter_kernel   The 3D high frequency pass filter kernel in the frequency 
                    space to supress low frequencies

    intensity_volume            The final 3D reconstructed density of photon 
                                production locations.           
    """
    def __init__(self, rays, intensities, binning):
        """
        Parameters
        ----------
        rays            The lixel rays defining the back projection geometry

        intensities     The intensities of the lixel rays (photon eqivalent)

        binning         The 3D binning of the atmospheric detector volume 
                        above the principal aperture plane    
        """
        self.binning = binning
        self.rays = rays
        self.intensities = intensities

        self.intensity_hist = histogram(
            self.rays, 
            self.binning, 
            self.intensities)

        self.ray_count_hist = histogram(
            self.rays, 
            self.binning)

        self.normalized_intensity_hist = normalize_ray_histograms(
            self.intensity_hist, 
            self.ray_count_hist)

        self.filter_kernel = ramp_kernel_in_frequency_space(self.binning)
        self.intensity_volume = frequency_filter(
            self.normalized_intensity_hist, 
            self.filter_kernel)

    def flat_xyz_intensity(self):
        """
        Returns a flat array of all voxels on axis 0 and x,y,z position and 
        intensity on axis 1.
        """
        xyz = self.binning.flat_xyz_voxel_positions()
        i = self.intensity_volume.flatten()
        return np.hstack((xyz, i.reshape(i.shape[0],1)))


    @classmethod
    def from_plenoscope_event(cls, event, valid_lixels, binning):
        #intensity_threshold = 1
        #valid_geom = event.light_field.valid_lixel.flatten()
        #valid_intensity = event.light_field.intensity.flatten() >= intensity_threshold
        #valid_arrival_time = (event.light_field.arrival_time.flatten() > 30e-9)*(event.light_field.arrival_time.flatten() < 40e-9)
        #valid = valid_geom*valid_intensity*valid_arrival_time

        rays = LixelRays(
            x=event.light_field.x_mean.flatten()[valid_lixels],
            y=event.light_field.y_mean.flatten()[valid_lixels],
            cx=event.light_field.cx_mean.flatten()[valid_lixels], 
            cy=event.light_field.cy_mean.flatten()[valid_lixels]) 
            
        intensities = event.light_field.intensity.flatten()[valid_lixels]

        return cls(rays, intensities, binning)

    @classmethod
    def from_idealized_plenoscope_event(cls, event, valid_photons, binning):
        rays = LixelRays(
            x=event.air_shower_photons.x[valid_photons],
            y=event.air_shower_photons.y[valid_photons], 
            cx=event.air_shower_photons.cx[valid_photons], 
            cy=event.air_shower_photons.cy[valid_photons]) 
            
        intensities = event.light_field.intensity[valid_photons]

        return cls(rays, intensities, binning)        

def true_volume_intensity(event, binning):
    event.simulation_truth