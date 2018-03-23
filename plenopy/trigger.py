import numpy as np
from .light_field.sequence import integrate_around_arrival_peak
from .image import ImageRays
import scipy.spatial.distance
from scipy.ndimage import convolve1d
import array
from scipy.sparse import coo_matrix


def trigger_on_light_field_sequence(
    light_field,
    pixel_neighborhood,
    min_photons_in_lixel=2,
    min_paxels_above_threshold_in_pixel=3,
    trigger_integration_time_window_in_slices=5,
):
    trigger_window = trigger_integration_time_window_in_slices
    num_time_slices = light_field.sequence.shape[0]
    num_trigger_trials = num_time_slices - trigger_window
    trigger_trials = np.zeros(num_trigger_trials, dtype=np.bool)

    for s in range(num_trigger_trials):
        lf = light_field.sequence[s:s+trigger_window, :].sum(axis=0)
        lf = lf.reshape((
            light_field.number_pixel,
            light_field.number_paxel))
        trigger_trials[s] = trigger_1(
            light_field_pixel_paxel=lf,
            min_photons_in_lixel=min_photons_in_lixel,
            min_paxels_above_threshold_in_pixel=min_paxels_above_threshold_in_pixel,
            pixel_neighborhood=pixel_neighborhood)

    return trigger_trials


def trigger_1(
    light_field_pixel_paxel,
    pixel_neighborhood,
    min_photons_in_lixel=2,
    min_paxels_above_threshold_in_pixel=3,
):
    trigger_image = make_trigger_image_based_on_light_field(
        light_field_pixel_paxel=light_field_pixel_paxel,
        min_photons_in_lixel=min_photons_in_lixel,
        min_paxels_above_threshold_in_pixel=min_paxels_above_threshold_in_pixel)

    return coincident_pixels_in_trigger_image(
        trigger_image=trigger_image,
        pixel_neighborhood=pixel_neighborhood)


def make_trigger_image_based_on_light_field(
    light_field_pixel_paxel,
    min_photons_in_lixel=2,
    min_paxels_above_threshold_in_pixel=3,
):
    trigger_field = light_field_pixel_paxel >= min_photons_in_lixel
    trigger_image = trigger_field.sum(axis=1)
    trigger_image = trigger_image >= min_paxels_above_threshold_in_pixel
    return trigger_image


def coincident_pixels_in_trigger_image(trigger_image, pixel_neighborhood):
    neighbors_of_pixels_with_trigger = np.multiply(
        pixel_neighborhood[trigger_image, :],
        trigger_image)
    return neighbors_of_pixels_with_trigger.sum() > 0


def neighborhood(x, y, epsilon, pixel_itself=False):
    dists = scipy.spatial.distance_matrix(
        np.vstack([x, y]).T,
        np.vstack([x, y]).T)
    nn = (dists <= epsilon)
    if not pixel_itself:
        nn = nn * (dists != 0)
    return nn

















def trigger_windows(
    light_field_sequence,
    trigger_integration_time_window_in_slices=5,
):
    return convolve1d(
        light_field_sequence,
        np.ones(trigger_integration_time_window_in_slices, dtype=np.uint16),
        axis=0)


def prepare_trigger_3(
    light_field_geometry,
    object_distances=[7.5e3, 15e3, 22.5e3]
):
    lfg = light_field_geometry
    num_refocuses = len(object_distances)
    image_rays = ImageRays(lfg)
    image_fov = lfg.sensor_plane2imaging_system.max_FoV_diameter
    pixel_fov = lfg.sensor_plane2imaging_system.pixel_FoV_hex_flat2flat
    num_pixel_1D = int(np.ceil(image_fov/pixel_fov))
    pixel_edges = np.linspace(-image_fov/2, image_fov/2, num_pixel_1D + 1)

    refocus_cx = np.zeros(shape=(num_refocuses, lfg.number_lixel))
    refocus_cy = np.zeros(shape=(num_refocuses, lfg.number_lixel))

    for i, obj in enumerate(object_distances):
        cx, cy = image_rays.cx_cy_in_object_distance(obj)
        refocus_cx[i, :] = cx
        refocus_cy[i, :] = cy

    return {
        'refocus_cx': refocus_cx,
        'refocus_cy': refocus_cy,
        'pixel_edges': pixel_edges
    }


def trigger_3(
    light_field,
    refocus_cx,
    refocus_cy,
    pixel_edges,
    min_photons_in_lixel=2,
):
    num_refocuses = refocus_cx.shape[0]
    num_pixel_1D = pixel_edges.shape[0] - 1
    refocused_images = np.zeros(
        shape=(
            num_refocuses,
            num_pixel_1D,
            num_pixel_1D))

    truncated_light_field = light_field.copy()
    truncated_light_field[light_field < min_photons_in_lixel] = 0

    for i in range(num_refocuses):
        refocused_images[i, :, :] = np.histogram2d(
            refocus_cx[i, :],
            refocus_cy[i, :],
            weights=truncated_light_field,
            bins=[pixel_edges, pixel_edges])[0]

    max_pixel_for_object_distance = refocused_images.max(axis=1).max(axis=1)

    return max_pixel_for_object_distance, refocused_images







def prepare_sum_trigger(light_field_geometry):
    epsilon = 1.1*light_field_geometry.sensor_plane2imaging_system.pixel_FoV_hex_flat2flat
    pixel_and_neighborhood = neighborhood(
        x=light_field_geometry.pixel_pos_cx,
        y=light_field_geometry.pixel_pos_cy,
        epsilon=np.deg2rad(0.1),
        pixel_itself=True)
    return pixel_and_neighborhood


def sum_trigger_image(image, pixel_and_neighborhood):
    return np.dot(pixel_and_neighborhood, image)


def sum_trigger(
    light_field,
    pixel_and_neighborhood,
    trigger_integration_time_window_in_slices=5
):
    image_sequence = light_field.sequence.reshape((
        light_field.number_time_slices,
        light_field.number_pixel,
        light_field.number_paxel)).sum(axis=2)

    trigger_window_image_sequece = trigger_windows(
        image_sequence,
        trigger_integration_time_window_in_slices)

    sum_trigger_window_image_sequece = trigger_window_image_sequece.dot(
        pixel_and_neighborhood)

    return np.max(sum_trigger_window_image_sequece)








def prepare_refocus_sum_trigger(
    light_field_geometry,
    object_distances=[7.5e3, 15e3, 22.5e3]
):
    image_rays = ImageRays(light_field_geometry)
    trigger_matrices = []
    for object_distance in object_distances:
        trigger_map = create_trigger_map(
            light_field_geometry=light_field_geometry,
            image_rays=image_rays,
            object_distance=object_distance)
        trigger_matrices.append(
            trigger_map_to_trigger_matrix(
                trigger_map=trigger_map,
                number_lixel=light_field_geometry.number_lixel,
                number_pixel=light_field_geometry.number_pixel))
    return {
        'object_distances': object_distances,
        'trigger_matrices': trigger_matrices}


def create_trigger_map(
    light_field_geometry,
    image_rays,
    object_distance,
):
    number_nearest_neighbors=7
    epsilon = 2.2*light_field_geometry.sensor_plane2imaging_system.pixel_FoV_hex_flat2flat
    trigger_map = []
    for pix in range(light_field_geometry.number_pixel):
            trigger_map.append([])
    cx, cy = image_rays.cx_cy_in_object_distance(object_distance)
    cxy = np.vstack((cx, cy)).T
    distances, pixel_indicies = image_rays.pixel_pos_tree.query(
        x=cxy,
        k=number_nearest_neighbors)
    for nei in range(number_nearest_neighbors):
        for lix in range(light_field_geometry.number_lixel):
            if distances[lix, nei] <= epsilon:
                trigger_map[pixel_indicies[lix, nei]].append(lix)
    return trigger_map


def trigger_map_to_trigger_matrix(
    trigger_map,
    number_lixel,
    number_pixel
):
    pixel_indicies = array.array('L')
    lixel_indicies = array.array('L')
    for pix, lixels in enumerate(trigger_map):
        for lix in lixels:
            pixel_indicies.append(pix)
            lixel_indicies.append(lix)
    trigger_matrix = coo_matrix(
        (np.ones(len(pixel_indicies), dtype=np.bool),
            (pixel_indicies, lixel_indicies)),
        shape=(number_pixel, number_lixel),
        dtype=np.bool)
    return trigger_matrix.tocsr()


def max_and_median(
    light_field,
    trigger_matrix,
    trigger_integration_time_window_in_slices=5
):
    trigger_image_seq = np.zeros(
        shape=(light_field.sequence.shape[0], light_field.number_pixel))
    for t in range(light_field.sequence.shape[0]):
        trigger_image_seq[t, :] = trigger_matrix.dot(light_field.sequence[t, :])
    trigger_window_seq = trigger_windows(
        trigger_image_seq,
        trigger_integration_time_window_in_slices)
    return np.max(trigger_window_seq), np.median(trigger_window_seq)


def refocus_sum_trigger(
    light_field,
    trigger_preparation,
    trigger_integration_time_window_in_slices=5
):
    tp = trigger_preparation
    result = []
    for i, trigger_matrix in enumerate(tp['trigger_matrices']):
        max_pe, med_pe = max_and_median(
            light_field,
            trigger_matrix,
            trigger_integration_time_window_in_slices)
        result.append({
            'object_distance': tp['object_distances'][i],
            'max': max_pe,
            'median': med_pe,
        })
    return result
