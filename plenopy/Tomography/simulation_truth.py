import numpy as np


def histogram_photon_bunches(photon_bunches, binning, observation_level=5e3):
    """
    parameter
    ---------
    photon_bunches      The CORSIKA photon bunch

    binning             The 3D binning of the atmospheric detector volume

    observation_level   The observation level associated with the photon bunch

    return
    ------                    
    hist            A 3D histogram according to the specified binnings, where
                    the voxel intensity encodes the number count of photon 
                    production positions in this voxel.
    """
    emission_positions = emission_positions_of_photon_bunches(
        photon_bunches=photon_bunches, 
        observation_level=observation_level)

    hist = np.histogramdd(
        emission_positions, bins=(
            binning.xy_bin_edges, 
            binning.xy_bin_edges, 
            binning.z_bin_edges),
            weights=photon_bunches.probability_to_reach_observation_level)

    return hist[0]


def emission_positions_of_photon_bunches(photon_bunches, observation_level=5e3):
    """
    parameter
    ---------
    photon_bunches          The CORSIKA photon bunches.

    observation_level       The observation level associated with the photon 
                            bunch.

    return
    ------                    
    emission_positions      An array of emission positions for each photon bunch
                            in the cartesian frame of the plenoscope.
    """
    supports = np.array([
        photon_bunches.x,
        photon_bunches.y,
        observation_level*np.ones(photon_bunches.x.shape[0])]).T

    directions = np.array([
        photon_bunches.cx,
        photon_bunches.cy,
        np.sqrt(1.0 - photon_bunches.cx**2 - photon_bunches.cy**2)]).T

    a = (photon_bunches.emission_height - supports[:,2])/directions[:,2]

    emission_positions = np.array([
            supports[:,0] - a*directions[:,0],
            supports[:,1] - a*directions[:,1],
            photon_bunches.emission_height
        ]).T

    # transform to plenoscope frame
    emission_positions[:,2] = emission_positions[:,2] - observation_level

    return emission_positions