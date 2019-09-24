import numpy as np
from .image import ImageRays
import scipy.spatial.distance
from scipy.ndimage import convolve1d
import array
from scipy.sparse import coo_matrix
import json
import os
# from .import plot
# from .image import Image


r"""
                          Light-Field-Trigger
             A refocused and patch-wise sum-trigger for the
                    Atmospheric Cherenkov Plenoscope
          which can be implemented today with proven technology

Abstract
--------

The light-field-trigger searches for coincident flashes of light from
air-showers in the pool of night-sky-background light.
The trigger must achive two goals:

1) It must find flashes of coincident light.

2) It must achive the first goal while the rate of accidental triggers is low.

For the 71 meter ACP we forsee a rate of air-showers in the 100,000s^-1 regime.
The accidental rate of triggers shall be below the rate of air-showers.

Method
------

The light-field-trigger has multiple and independent layers which are optimized
for distinct object-distances above the aperture-plane of the plenoscope.
Air-showers of low energy particles seem to emitt most of the photons from
10km to 20km above the aperture. Each layer implements the projection of the
recorded light-field onto an image which is focused to the object-distance of
the layer.
In each layer the signals of the individual read-out-channels (lixels) are
summed up to a group which corresponds to a trigger-patch focused to a the
layer's object-distance. The corresponding connection-diagram is represented by
the lixel_summations matrix. A trigger-patch here is a pixel 'c' and its
direct neighbors:

    A trigger-patch of pixels for pixel 'c':
                     ___
                 ___/   \___
                /   \___/   \
                \___/ c \___/
                /   \___/   \
                \___/   \___/
                    \___/

Since the trigger-patches overlap, the trigger-image in each layer has the same
number of pixels as the original hardware layout. Thus the amlitudes of the
trigger-patches in the trigger-image are approx. seven times the amplitude of a
single pixel in the regular image.
A trigger-patch covers about half the solid-angle of the smallest air-showers
we are looking for. The trigger also summs up along the time in a window which
is slided along the sequence of time-slices. The idae is that the fluctuations
of the night-sky-background are reduced when summed in a trigger-patch.
When a group of naighboring trigger-patches reaches amplitudes above a
pre-determined threshold. The trigger on this layer is activated.

The final trigger decision is not taken here. Here we only condese the
information to take the decision. We write out the triggers of the individual
layers.

Exposure-time
-------------

The accidental rate of the trigger must be estimated. Therefore we need to keep
track of the exposure-time. Each simulated event is a sequence of light-fields
over independent time-slices. Usually the records contain 50ns in 100
time-slices at 2GHz sampling-frequency.
"""


def prepare_refocus_sum_trigger(
    light_field_geometry,
    object_distances=[7.5e3, 15e3, 22.5e3]
):
    """
    Prepare the simulation of the refocus-sum-trigger. This shall be done
    before simulating the trigger in each event of a 'run'. A 'run' here means
    a list of events which all share the same light-field-geometry and
    ambient-conditions.

    Parameters
    ----------

    light_field_geometry    The light-field-geometry of the plenoscope.

    object_distances        An array of object-distances to focus the trigger
                            on.

    Returns
    -------

    trigger_matrices        A boolean matrix [number_lixel x number_pixel] for
                            each object-distance of the trigger. The
                            trigger-matrix describes which lixels are added up
                            (True in the matrix) to compose a pixel.
                            In the hardware implementation this matrix
                            corresponds to the wires from the read-out-channels
                            to the analog-adders which form pixels.

    neighborhood_of_pixel   A boolean matrix [number_pixel x number_pixel] to
                            represent the nearest neighbors of a pixel. The
                            reference pixel itself is excluded. Only the
                            neighbours.

    object_distances        As the input.
    """
    image_rays = ImageRays(light_field_geometry)
    lixel_summations = []
    for object_distance in object_distances:
        lixel_summation = create_lixel_summation(
            light_field_geometry=light_field_geometry,
            image_rays=image_rays,
            object_distance=object_distance)
        lixel_summations.append(
            lixel_summation_to_sparse_matrix(
                lixel_summation=lixel_summation,
                number_lixel=light_field_geometry.number_lixel,
                number_pixel=light_field_geometry.number_pixel))

    pixel_diameter = (
        1.1*light_field_geometry.
        sensor_plane2imaging_system.pixel_FoV_hex_flat2flat)
    neighborhood_of_pixel = neighborhood(
        x=light_field_geometry.pixel_pos_cx,
        y=light_field_geometry.pixel_pos_cy,
        epsilon=pixel_diameter,
        itself=False)

    return {'object_distances': object_distances,
            'lixel_summations': lixel_summations,
            'neighborhood_of_pixel': neighborhood_of_pixel}


def pivot(threshold_lower, threshold_upper):
    return np.floor(.5*(threshold_lower + threshold_upper))


def estimate_number_neighbors(
    image_sequence,
    threshold,
    neighborhood_of_pixel,
):
    patches_max_neighbors = []
    patches_argmax_neighbors = []
    patch_mask_sequence = image_sequence >= threshold
    for time_slice, patch_mask in enumerate(patch_mask_sequence):
        am, m = argmax_number_of_active_neighboring_patches_and_active_itself(
            patch_mask=patch_mask,
            neighborhood_of_pixel=neighborhood_of_pixel)
        patches_max_neighbors.append(m)
        patches_argmax_neighbors.append(am)
    time_slice_with_most_active_neighboring_patches = np.argmax(
        patches_max_neighbors)
    number_neighbors = patches_max_neighbors[
        time_slice_with_most_active_neighboring_patches]
    return (
        number_neighbors,
        time_slice_with_most_active_neighboring_patches,
        patches_argmax_neighbors)


def apply_refocus_sum_trigger(
    event,
    trigger_preparation,
    min_number_neighbors=3,
    integration_time_in_slices=5,
):
    """
    Estimates the lowest patch_threshold that would still have triggered.
    This is called for each event in a run. The number of neighboring
    trigger-patches of pixels which are above the patch_threshold is counted
    for each recorded time-slice.

    Parameters
    ----------

    event                       The recorded event of the plenoscope.

    trigger_preparation         The configuration of the trigger-wiring.

    integration_time_in_slices  The length of the sliding integration-window of
                                the trigger. A.k.a coincidence-window.

    min_number_neighbors        The minimum number of neighboring
                                patches which have to be above the
                                threshold.

    Returns
    -------                     A list of dicts is returned. One dict for each
                                object-distance of the refocus-sum-trigger.

    patch_threshold             The lowest threshold for the patches which
                                still triggers.

    object_distance             The object-distance the trigger focused on.

    integration_time_in_slices  As in the input.

    patch_median                The median amplitude of all trigger-patches in
                                the sequence along time.

    patch_max                   The max amplitude of all trigger-patches in the
                                the sequence along time.

    max_number_neighbors        The maximum number of neighboring
                                trigger-patches which are above the
                                patch-threshold.

    argmax_coincident_patches   The time-slice in the sequence along time which
                                needs the lowest trigger-threshold.
    """
    tp = trigger_preparation
    results = []
    for obj, object_distance in enumerate(tp['object_distances']):
        image_sequence = sum_trigger_image_sequence(
            light_field_sequence=event.light_field_sequence_for_isochor_image(),
            lixel_summation=tp['lixel_summations'][obj],
            integration_time_in_slices=integration_time_in_slices)

        threshold_upper = np.max(image_sequence)
        threshold_lower = threshold_upper

        number_neighbors = 0
        while number_neighbors < min_number_neighbors:
            threshold_lower = np.floor(0.95*threshold_lower)
            (   number_neighbors,
                time_slice_with_most_active_neighboring_patches,
                patches_argmax_neighbors
            ) = estimate_number_neighbors(
                image_sequence=image_sequence,
                threshold=threshold_lower,
                neighborhood_of_pixel=tp['neighborhood_of_pixel'])


        threshold = pivot(threshold_lower, threshold_upper)
        iteration = 0
        while True:
            (   number_neighbors,
                time_slice_with_most_active_neighboring_patches,
                patches_argmax_neighbors
            ) = estimate_number_neighbors(
                image_sequence=image_sequence,
                threshold=threshold,
                neighborhood_of_pixel=tp['neighborhood_of_pixel'])

            if threshold == threshold_lower:
                break

            if number_neighbors >= min_number_neighbors:
                threshold_lower = threshold
                threshold = pivot(threshold_lower, threshold_upper)

            if number_neighbors < min_number_neighbors:
                threshold_upper = threshold
                threshold = pivot(threshold_lower, threshold_upper)

            iteration += 1

        results.append({
            'exposure_time_in_slices': int(image_sequence.shape[0]),
            'object_distance': float(object_distance),
            'integration_time_in_slices': int(integration_time_in_slices),
            'patch_threshold': int(threshold),
            'patch_median': int(np.median(image_sequence)),
            'patch_max': int(np.max(image_sequence)),
            'max_number_neighbors': int(number_neighbors),
            'time_slice_with_most_active_neighboring_patches': int(
                time_slice_with_most_active_neighboring_patches),
            'number_iterations': bool(iteration),
            'min_number_neighbors': int(min_number_neighbors),
            'patches': [int(pp) for pp in patches_argmax_neighbors[
                time_slice_with_most_active_neighboring_patches]],
        })
    return results


def create_lixel_summation(
    light_field_geometry,
    image_rays,
    object_distance,
    number_nearest_neighbors=7,
):
    """
    Find which lixel belongs to which patch of pixels for a given
    object-distance. Here for the trigger pixels are organized in patches of
    pixels of seven pixels. A patch of pixel 'c' includes its nearest
    neighbors.

    In the plenoscope each pixel has multiple lixels.

    Returns
    -------

    lixel_summation         A list over the patches of pixels. Each element
                            contains a list of the ids of the lixels which
                            belong to the corresponding patch.
    """
    epsilon = (
        2.2*light_field_geometry.
        sensor_plane2imaging_system.pixel_FoV_hex_flat2flat)
    lixel_summation = list_of_empty_lists(light_field_geometry.number_pixel)
    cx, cy = image_rays.cx_cy_in_object_distance(object_distance)
    cxy = np.vstack((cx, cy)).T
    distances, pixel_indicies = image_rays.pixel_pos_tree.query(
        x=cxy,
        k=number_nearest_neighbors)
    for nei in range(number_nearest_neighbors):
        for lix in range(light_field_geometry.number_lixel):
            if distances[lix, nei] <= epsilon:
                lixel_summation[pixel_indicies[lix, nei]].append(lix)
    return lixel_summation


def lixel_summation_to_sparse_matrix(
    lixel_summation,
    number_lixel,
    number_pixel
):
    """
    Converts the lixel_summation_list into a sparse matrix. This is only for
    computation speed. The lixel_summation_matrix contains the same information
    as the lixel_summation_list.

    Returns
    -------

    lixel_summation_matrix      A boolean matrix [number_pixel x number_lixel]
                                which expresses what lixels belong to a
                                trigger-patch of pixels.
    """
    pixel_indicies = array.array('L')
    lixel_indicies = array.array('L')
    for pix, lixels in enumerate(lixel_summation):
        for lix in lixels:
            pixel_indicies.append(pix)
            lixel_indicies.append(lix)
    lixel_summation_matrix = coo_matrix(
        (np.ones(len(pixel_indicies), dtype=np.bool),
            (pixel_indicies, lixel_indicies)),
        shape=(number_pixel, number_lixel),
        dtype=np.bool)
    return lixel_summation_matrix.tocsr()


def sum_trigger_image_sequence(
    light_field_sequence,
    lixel_summation,
    integration_time_in_slices=5,
):
    """
    Sums up the signals of the lixels into the trigger-patches according to the
    relations in the lixel_summation.
    """
    number_pixel = lixel_summation.shape[0]
    trigger_image_seq = np.zeros(
        shape=(light_field_sequence.shape[0], number_pixel))
    for t in range(light_field_sequence.shape[0]):
        trigger_image_seq[t, :] = lixel_summation.dot(
            light_field_sequence[t, :])
    trigger_window_seq = convole_sequence(
        sequence=trigger_image_seq,
        integration_time_in_slices=integration_time_in_slices)
    return trigger_window_seq


def neighborhood(x, y, epsilon, itself=False):
    dists = scipy.spatial.distance_matrix(
        np.vstack([x, y]).T,
        np.vstack([x, y]).T)
    nn = (dists <= epsilon)
    if not itself:
        nn = nn * (dists != 0)
    return nn


def convole_sequence(
    sequence,
    integration_time_in_slices=5,
):
    return convolve1d(
        sequence,
        np.ones(integration_time_in_slices, dtype=np.uint16),
        axis=0,
        mode='constant',
        cval=0)


def list_of_empty_lists(n):
    return [[] for i in range(n)]


def max_number_of_neighboring_trigger_patches(
    patch_mask,
    neighborhood_of_pixel
):
    if np.sum(patch_mask) == 0:
        return 0
    else:
        return np.max(
            (neighborhood_of_pixel[patch_mask]*patch_mask).sum(axis=1)
        )


def number_of_active_neighboring_patches_and_active_itself(
    patch_mask,
    neighborhood_of_pixel
):
    num_active_neighbors = number_of_active_neighboring_patches(
        patch_mask,
        neighborhood_of_pixel)
    return patch_mask*num_active_neighbors


def number_of_active_neighboring_patches(
    patch_mask,
    neighborhood_of_pixel
):
    return (neighborhood_of_pixel*patch_mask).sum(axis=1)


def argmax_number_of_active_neighboring_patches_and_active_itself(
    patch_mask,
    neighborhood_of_pixel
):
    if np.sum(patch_mask) == 0:
        return [], 0

    n = (neighborhood_of_pixel[patch_mask]*patch_mask).sum(axis=1)
    patch_idx = np.arange(patch_mask.shape[0])[patch_mask]
    an = np.argmax(n)
    am = patch_idx[an]
    m = n[an]
    args_max = patch_idx[n == m]
    return args_max, m


def region_of_interest_from_trigger_response(
    trigger_response,
    time_slice_duration,
    pixel_pos_cx,
    pixel_pos_cy,
):
    patch_thresholds = []
    for refocus_layer in trigger_response:
        patch_thresholds.append(refocus_layer['patch_threshold'])
    m = np.argmax(patch_thresholds)

    time_slice = trigger_response[m][
        'time_slice_with_most_active_neighboring_patches']

    return {
        'time_center_roi': time_slice*time_slice_duration,
        'cx_center_roi': pixel_pos_cx[trigger_response[m]['patches'][0]],
        'cy_center_roi': pixel_pos_cy[trigger_response[m]['patches'][0]],
        'object_distance': trigger_response[m]['object_distance']}


def read_trigger_response(path):
    with open(path, 'rt') as fin:
        t = json.loads(fin.read())
    return t


def read_trigger_response_of_event(event):
    path = os.path.join(event._path, 'refocus_sum_trigger.json')
    return read_trigger_response(path)
