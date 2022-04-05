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


def init_reconstruction(
    light_field_geometry,
    photon_lixel_ids,
    binning,
    sparse_system_matrix,
):
    image_rays = image.ImageRays(light_field_geometry)

    intensities = np.zeros(light_field_geometry.number_lixel)
    for lixel_id in photon_lixel_ids:
        intensities[lixel_id] += 1

    psf = system_matrix.to_numpy_csr_matrix(
        sparse_system_matrix=sparse_system_matrix,
        number_beams=light_field_geometry.number_lixel,
        number_volume_cells=binning["number_bins"],
    )

    r = {}
    r['binning'] = binning
    r['point_spread_function'] = psf
    r['image_ray_supports'] = image_rays.support
    r['image_ray_directions'] = image_rays.direction
    r['image_ray_intensities'] = intensities
    r['photon_lixel_ids'] = photon_lixel_ids

    r['reconstructed_volume_intensity'] = np.zeros(
        binning['number_bins'], dtype=np.float32)

    r['iteration'] = 0

    # Total length of ray
    r['image_ray_integral'] = psf.sum(axis=0).T

    # Total distance of all rays in this voxel
    r['voxel_integral'] = psf.sum(axis=1)

    # The sum of the length of all rays hiting this voxel weighted with the
    # overlap of the ray and this voxel
    voxel_cross_psf = psf.dot(r['image_ray_integral'])
    voxel_cross_psf = np.array(
        voxel_cross_psf
    ).reshape((voxel_cross_psf.shape[0],))

    image_ray_cross_psf = psf.T.dot(r['voxel_integral'])
    image_ray_cross_psf = np.array(
        image_ray_cross_psf
    ).reshape((image_ray_cross_psf.shape[0],))

    r['voxel_cross_psf'] = voxel_cross_psf
    r['image_ray_cross_psf'] = image_ray_cross_psf
    return r


def reconstructed_volume_intensity_as_cube(
    reconstructed_volume_intensity,
    binning
):
    return reconstructed_volume_intensity.reshape((
            binning['number_sen_x_bins'],
            binning['number_sen_y_bins'],
            binning['number_sen_z_bins']),
        order='C')


def one_more_iteration(reconstruction):
    r = reconstruction
    reconstructed_voxel_I = r['reconstructed_volume_intensity'].copy()
    measured_image_ray_I = r['image_ray_intensities']
    voxel_cross_psf = r['voxel_cross_psf']
    image_ray_cross_psf = r['image_ray_cross_psf']
    psf = r['point_spread_function']

    measured_I_voxel = psf.dot(measured_image_ray_I)
    voxel_overlap = voxel_cross_psf > 0.0
    measured_I_voxel[voxel_overlap] /= voxel_cross_psf[voxel_overlap]

    projected_image_ray_I = psf.T.dot(reconstructed_voxel_I)
    image_ray_overlap = image_ray_cross_psf > 0.0
    projected_image_ray_I[image_ray_overlap] /= image_ray_cross_psf[
        image_ray_overlap]

    proj_I_voxel = psf.dot(projected_image_ray_I)

    voxel_diffs = measured_I_voxel - proj_I_voxel

    reconstructed_voxel_I += voxel_diffs

    reconstructed_voxel_I[reconstructed_voxel_I < 0.0] = 0.0

    diff = np.abs(
        reconstructed_voxel_I - r['reconstructed_volume_intensity']).sum()
    print('Intensity difference to previous iteration '+str(diff))

    r['reconstructed_volume_intensity'] = reconstructed_voxel_I.copy()
    r['iteration'] += 1
    return r


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

    true_emission_positions_image_domain = xyz2cxcyb(
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
        assert binning_is_equal(
            simulation_truth['binning'],
            reconstruction['binning'])
        intensity_volume_2 = reconstructed_volume_intensity_as_cube(
            simulation_truth['true_volume_intensity'],
            r['binning'])

    slices.save_slice_stack(
        intensity_volume=reconstructed_volume_intensity_as_cube(
            r['reconstructed_volume_intensity'],
            r['binning']),
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
