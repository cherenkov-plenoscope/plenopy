import numpy as np
import ray_voxel_overlap
import json
import os
from skimage.measure import LineModelND, ransac
from .. import system_matrix

from ... import classify
from ... import image
from ... import trigger
from ..simulation_truth import emission_positions_of_photon_bunches
from ...plot import slices
from . import binning
from . import reconstruction
from ... import thin_lens



def init_simulation_truth_from_event(
    event,
    binning
):
    r = {}
    r['binning'] = binning

    s2i = event.light_field_geometry.sensor_plane2imaging_system
    crh = event.simulation_truth.event.corsika_run_header
    aspb = event.simulation_truth.air_shower_photon_bunches

    ep = emission_positions_of_photon_bunches(
        aspb,
        limited_aperture_radius=s2i.
        expected_imaging_system_max_aperture_radius,
        limited_fov_radius=0.5*s2i.max_FoV_diameter,
        observation_level=crh.observation_level(),)

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


def save_imgae_slice_stack(
    reconstruction,
    simulation_truth=None,
    out_dir='./tomography',
    sqrt_intensity=False,
    event_info_repr=None,
):
    r = reconstruction
    os.makedirs(out_dir, exist_ok=True)

    if simulation_truth is None:
        intensity_volume_2 = None
    else:
        assert binning.is_equal(
            simulation_truth['binning'],
            reconstruction['binning'])
        intensity_volume_2 = binning.volume_intensity_as_cube(
            volume_intensity=simulation_truth['true_volume_intensity'],
            binning=r['binning'],
        )

    slices.save_slice_stack(
        intensity_volume=binning.volume_intensity_as_cube(
            volume_intensity=r['reconstructed_volume_intensity'],
            binning=r['binning'],
        ),
        event_info_repr=event_info_repr,
        xy_extent=[
            r['binning']['sen_x_bin_edges'].min(),
            r['binning']['sen_x_bin_edges'].max(),
            r['binning']['sen_y_bin_edges'].min(),
            r['binning']['sen_y_bin_edges'].max(),
        ],
        z_bin_centers=r['binning']['sen_z_bin_centers'],
        output_path=out_dir,
        image_prefix='slice_',
        intensity_volume_2=intensity_volume_2,
        xlabel='x/m',
        ylabel='y/m',
        sqrt_intensity=sqrt_intensity,
    )


def volume_intensity_sensor_frame_to_xyzi_object_frame(
    volume_intensity,
    binning,
    threshold=0
):
    cx_bin_centers = binning['cx_bin_centers']
    cy_bin_centers = binning['cy_bin_centers']
    obj_bin_centers = binning['obj_bin_centers']

    xyzi = []
    for x in range(volume_intensity.shape[0]):
        for y in range(volume_intensity.shape[1]):
            for z in range(volume_intensity.shape[2]):
                if volume_intensity[x, y, z] > threshold:
                    xyzi.append(
                        np.array([
                            np.tan(cx_bin_centers[x])*obj_bin_centers[z],
                            np.tan(cy_bin_centers[y])*obj_bin_centers[z],
                            obj_bin_centers[z],
                            volume_intensity[x, y, z]]
                        )
                    )
    xyzi = np.array(xyzi)
    return xyzi


def xyzi_2_xyz(
    xyzi,
    maxP=25
):
    maxI = np.max(xyzi[:, 3])
    xyz = []
    for p in xyzi:
        intensity = int(np.round(maxP*(p[3]/maxI)))
        for i in range(intensity):
            xyz.append(np.array([p[0], p[1], p[2]]))
    xyz = np.array(xyz)
    return xyz


def fit_trajectory_to_point_cloud(
    xyz,
    residual_threshold=50
):
    model_robust, inliers = ransac(
        xyz,
        LineModelND,
        min_samples=200,
        residual_threshold=residual_threshold,
        max_trials=50)

    if model_robust.params[1][2] < 0:
        direction = model_robust.params[1]
    else:
        direction = -model_robust.params[1]

    support = np.zeros(3)
    ri = model_robust.params[0][2]/direction[2]
    support[0] = model_robust.params[0][0] - ri*direction[0]
    support[1] = model_robust.params[0][1] - ri*direction[1]
    return support, direction


def init_reconstruction_from_event(event, trigger_geometry, binning):

    trigger_responses = pl.trigger.io.read_trigger_response_from_path(
        path=os.path.join(event._path, 'refocus_sum_trigger.json')
    )
    roi =pl.trigger.region_of_interest.from_trigger_response(
        trigger_response=trigger_responses,
        trigger_geometry=trigger_geometry,
        time_slice_duration=event.raw_sensor_response.time_slice_duration,
    )
    (
        air_shower_photon_ids, lixel_ids_of_photons
    ) = classify.classify_air_shower_photons_from_trigger_response(
        event, trigger_region_of_interest=roi)

    return init_reconstruction(
        event=event,
        binning=binning,
        air_shower_photon_ids=air_shower_photon_ids,
        lixel_ids_of_photons=lixel_ids_of_photons)


def overlap_2_xyzI(overlap, x_bin_edges, y_bin_edges, z_bin_edges):
    '''
    For plotting using the xyzI representation.
    Returns a 2D matrix (Nx4) of N overlaps of a ray with xoxels. Each row is
    [x,y,z positions and overlapping distance].
    '''
    x_bin_centers = (x_bin_edges[0:-1] + x_bin_edges[1:])/2
    y_bin_centers = (y_bin_edges[0:-1] + y_bin_edges[1:])/2
    z_bin_centers = (z_bin_edges[0:-1] + z_bin_edges[1:])/2
    xyzI = np.zeros(shape=(len(overlap['overlap']), 4))
    for i in range(len(overlap['overlap'])):
        xyzI[i] = np.array([
            x_bin_centers[overlap['x'][i]],
            y_bin_centers[overlap['y'][i]],
            z_bin_centers[overlap['z'][i]],
            overlap['overlap'][i],
        ])
    return xyzI
