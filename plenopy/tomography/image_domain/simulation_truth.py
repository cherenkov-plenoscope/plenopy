import numpy as np
import os
from .. import simulation_truth
from ... import thin_lens


def init(event, binning):
    r = {}

    s2i = event.light_field_geometry.sensor_plane2imaging_system
    crh = event.simulation_truth.event.corsika_run_header
    aspb = event.simulation_truth.air_shower_photon_bunches

    ep = simulation_truth.emission_positions_of_photon_bunches(
        aspb,
        limited_aperture_radius=s2i.
        expected_imaging_system_max_aperture_radius,
        limited_fov_radius=0.5*s2i.max_FoV_diameter,
        observation_level=crh.observation_level(),
    )

    r['emission_positions'] = ep['emission_positions'][ep['valid_acceptence']]

    true_emission_positions_image_domain = thin_lens.xyz2cxcyb(
        r['emission_positions'][:, 0],
        r['emission_positions'][:, 1],
        r['emission_positions'][:, 2],
        binning['focal_length']).T
    tepid = true_emission_positions_image_domain

    # directions to positions on image screen
    tepid[:, 0] = -binning['focal_length']*np.tan(tepid[:, 0])
    tepid[:, 1] = -binning['focal_length']*np.tan(tepid[:, 1])

    r['emission_positions_image_domain'] = tepid

    hist = np.histogramdd(
        r['emission_positions_image_domain'],
        bins=(
            binning['sen_x_bin_edges'],
            binning['sen_y_bin_edges'],
            binning['sen_z_bin_edges']
        ),
        weights=aspb.probability_to_reach_observation_level[
            ep['valid_acceptence']]
    )

    r['true_volume_intensity'] = hist[0].flatten()
    return r
