import numpy as np
from .light_field.sequence import integrate_around_arrival_peak
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
    time_integration_radius_in_slices=2,
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



