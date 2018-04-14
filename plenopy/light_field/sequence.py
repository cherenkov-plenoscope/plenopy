import numpy as np


def time_slice_with_max_intensity(sequence):
    number_time_slices = sequence.shape[0]
    max_along_slices = np.zeros(number_time_slices, dtype=np.uint16)
    for s, slic in enumerate(sequence):
        max_along_slices[s] = slic.max()
    return np.argmax(max_along_slices)


def integrate_around_arrival_peak(
    sequence,
    integration_radius=1
):
    '''
    Reduces a sequence of light fields or images to just one light field or
    image integrated around the main arrival intensity.

    Parameters
    ----------

    sequence                Matrix 2D, sequence of e.g. light fields or images,
                            where the time slices go along axis=0.

    integration_radius      Integer, integration radius in units of time slices.
    '''
    peak_slice = time_slice_with_max_intensity(sequence)
    start_slice = np.max([peak_slice - integration_radius, 0])
    stop_slice = np.min([peak_slice + integration_radius + 1, sequence.shape[0]-1])
    return {
        'integral': np.sum(sequence[start_slice:stop_slice, :], axis=0),
        'peak_slice': peak_slice,
        'start_slice': start_slice,
        'stop_slice': stop_slice,
    }


def pixel_sequence(lixel_sequence, number_pixel, number_paxel):
    return _to_image_sequence(
        lixel_sequence,
        number_pixel,
        number_paxel,
        axis=1)


def paxel_sequence(lixel_sequence, number_pixel, number_paxel):
    return _to_image_sequence(
        lixel_sequence,
        number_pixel,
        number_paxel,
        axis=0)


def _to_image_sequence(lixel_sequence, number_pixel, number_paxel, axis):
    if axis == 1:
        bins = number_pixel
    else:
        bins = number_paxel

    number_time_slices = lixel_sequence.shape[0]
    imgs = np.zeros(
        shape=(number_time_slices, bins),
        dtype=np.uint16)

    for t in range(number_time_slices):
        imgs[t, :] = lixel_sequence[t, :].reshape(
            number_pixel,
            number_paxel).sum(axis=axis)

    return imgs