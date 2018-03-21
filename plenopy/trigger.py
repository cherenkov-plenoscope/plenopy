import numpy as np
from .light_field.sequence import integrate_around_arrival_peak
from .image import ImageRays
import scipy.spatial.distance


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


def neighborhood(x, y, epsilon):
    dists = scipy.spatial.distance_matrix(
        np.vstack([x, y]).T,
        np.vstack([x, y]).T)
    nn = (dists <= epsilon) * (dists != 0)
    return nn

















def trigger_windows(
    light_field_sequence,
    trigger_integration_time_window_in_slices=5,
):
    trigger_window = trigger_integration_time_window_in_slices
    num_time_slices = light_field_sequence.shape[0]
    num_trigger_trials = num_time_slices - trigger_window
    trigger_windows = np.zeros(
        shape=(num_trigger_trials, light_field_sequence.shape[1]))
    for s in range(num_trigger_trials):
        trigger_windows[s, :] = light_field_sequence[s:s+trigger_window, :].sum(axis=0)
    return trigger_windows


def prepare_trigger_3(
    light_field_geometry,
    object_distances=[5e3, 10e3, 25e3, 20e3, 25e3]
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



