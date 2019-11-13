import numpy as np
import ray_voxel_overlap
import json
import os
from skimage.measure import LineModelND, ransac
from . import system_matrix

from ..thin_lens import object_distance_2_image_distance as g2b
from ..thin_lens import image_distance_2_object_distance as b2g
from ..thin_lens import xyz2cxcyb

from .. import classify
from .. import image
from .. import trigger
from .simulation_truth import emission_positions_of_photon_bunches
from ..plot import slices


def linspace_edges_centers(start, stop, num):
    bin_edges = np.linspace(start, stop, num+1)
    width = stop - start
    bin_radius = 0.5*width/num
    bin_centers = bin_edges[: -1] + bin_radius
    return bin_edges, bin_centers, width, bin_radius


def init_binning_for_depth_of_field(
    focal_length,
    cx_min=np.deg2rad(-3.5),
    cx_max=np.deg2rad(+3.5),
    number_cx_bins=96,
    cy_min=np.deg2rad(-3.5),
    cy_max=np.deg2rad(3.5),
    number_cy_bins=96,
    obj_min=5e3,
    obj_max=25e3,
    number_obj_bins=32,
):
    b = {}
    b['focal_length'] = focal_length

    b['cx_min'] = cx_min
    b['cx_max'] = cx_max
    b['number_cx_bins'] = number_cx_bins
    (
        b['cx_bin_edges'],
        b['cx_bin_centers'],
        b['cx_width'],
        b['cx_bin_radius']) = linspace_edges_centers(
        start=cx_min, stop=cx_max, num=number_cx_bins)
    b['sen_x_min'] = np.tan(cx_min)*focal_length
    b['sen_x_max'] = np.tan(cx_max)*focal_length
    b['number_sen_x_bins'] = number_cx_bins
    b['sen_x_width'] = b['sen_x_max'] - b['sen_x_min']
    b['sen_x_bin_radius'] = np.tan(b['cx_bin_radius'])*focal_length
    b['sen_x_bin_edges'] = np.tan(b['cx_bin_edges'])*focal_length
    b['sen_x_bin_centers'] = np.tan(b['cx_bin_centers'])*focal_length

    b['cy_min'] = cy_min
    b['cy_max'] = cy_max
    b['number_cy_bins'] = number_cy_bins
    (
        b['cy_bin_edges'],
        b['cy_bin_centers'],
        b['cy_width'],
        b['cy_bin_radius']) = linspace_edges_centers(
        start=cy_min, stop=cy_max, num=number_cy_bins)
    b['sen_y_min'] = np.tan(cy_min)*focal_length
    b['sen_y_max'] = np.tan(cy_max)*focal_length
    b['number_sen_y_bins'] = number_cy_bins
    b['sen_y_width'] = b['sen_y_max'] - b['sen_y_min']
    b['sen_y_bin_radius'] = np.tan(b['cy_bin_radius'])*focal_length
    b['sen_y_bin_edges'] = np.tan(b['cy_bin_edges'])*focal_length
    b['sen_y_bin_centers'] = np.tan(b['cy_bin_centers'])*focal_length

    b['number_bins'] = number_cx_bins*number_cy_bins*number_obj_bins

    b['sen_z_min'] = g2b(obj_max, focal_length)
    b['sen_z_max'] = g2b(obj_min, focal_length)
    b['number_sen_z_bins'] = number_obj_bins
    (
        b['sen_z_bin_edges'],
        b['sen_z_bin_centers'],
        b['sen_z_width'],
        b['sen_z_bin_radius']) = linspace_edges_centers(
        start=b['sen_z_min'], stop=b['sen_z_max'], num=b['number_sen_z_bins'])

    b['obj_min'] = obj_min
    b['obj_max'] = obj_max
    b['number_obj_bins'] = number_obj_bins
    b['obj_bin_edges'] = b2g(b['sen_z_bin_edges'], focal_length)
    b['obj_bin_centers'] = b2g(b['sen_z_bin_centers'], focal_length)

    return b


__KEY_CTOR_BINNING = {
    "focal_length": float,
    "cx_min": float,
    "cx_max": float,
    "number_cx_bins": int,
    "cy_min": float,
    "cy_max": float,
    "number_cy_bins": int,
    "obj_min": float,
    "obj_max": float,
    "number_obj_bins": int}


def write_binning(binning, path):
    binning_ctor_dict = {}
    for key in __KEY_CTOR_BINNING:
        _dtype = __KEY_CTOR_BINNING[key]
        binning_ctor_dict[key] = _dtype(binning[key])
    with open(path, "wt") as f:
        f.write(json.dumps(binning_ctor_dict, indent=4))


def read_binning(path):
    with open(path, "rt") as f:
        binning_ctor_dict = json.loads(f.read())
    return init_binning_for_depth_of_field(**binning_ctor_dict)


def binning_is_equal(binning_a, binning_b):
    keys = [
        'focal_length',
        'cx_min',
        'cx_max',
        'cy_min',
        'cy_max',
        'number_cx_bins',
        'number_cy_bins',
        'obj_min',
        'obj_max',
        'number_obj_bins']
    for key in keys:
        if binning_a[key] != binning_b[key]:
            return False
    return True


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

    psf = system_matrix.__make_matrix(
        sparse_system_matrix=sparse_system_matrix,
        light_field_geometry=light_field_geometry,
        binning=binning)

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


def init_reconstruction_from_event(event, binning):
    trigger_response = trigger.read_trigger_response_of_event(event)

    roi = trigger.region_of_interest_from_trigger_response(
        trigger_response=trigger_response,
        time_slice_duration=event.light_field.time_slice_duration,
        pixel_pos_cx=event.light_field.pixel_pos_cx,
        pixel_pos_cy=event.light_field.pixel_pos_cy)

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
