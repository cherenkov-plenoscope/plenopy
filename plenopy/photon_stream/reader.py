import numpy as np


def py_stream2sequence(
    photon_stream,
    time_slice_duration,
    NEXT_READOUT_CHANNEL_MARKER,
    sequence,
    time_delay_mean
):
    number_time_slices = sequence.shape[0]

    lixel = 0
    for symbol in photon_stream:
        if symbol == NEXT_READOUT_CHANNEL_MARKER:
            lixel += 1
        else:
            time_slice = symbol

            arrival_time = time_slice*time_slice_duration
            arrival_time -= time_delay_mean[lixel]

            arrival_slice = int(
                np.round(
                    arrival_time/time_slice_duration))

            if arrival_slice < number_time_slices and arrival_slice >= 0:
                sequence[arrival_slice, lixel] += 1.0
            else:
                pass


def py_photon_stream_to_image_sequence(
    photon_stream,
    photon_stream_next_channel_marker,
    time_slice_duration,
    time_delay_image_mean,
    projection_links,
    projection_starts,
    projection_lengths,
    number_lixel,
    number_pixel,
    number_time_slices,
):
    photon_stream_length = photon_stream.shape[0]

    assert time_slice_duration >= 0.0
    assert number_lixel == projection_starts.shape[0]
    assert number_lixel == projection_lengths.shape[0]
    assert number_lixel == time_delay_image_mean.shape[0]

    out_image_sequence = np.zeros(
        shape=(number_time_slices, number_pixel),
        dtype=np.uint32
    )

    phs_lixel = 0
    for phs_i in range(photon_stream_length):
        phs_symbol = photon_stream[phs_i]

        if phs_symbol == photon_stream_next_channel_marker:
            phs_lixel += 1
        else:
            raw_arrival_time_slice = phs_symbol

            arrival_time = raw_arrival_time_slice*time_slice_duration
            arrival_time -= time_delay_image_mean[phs_lixel]

            arrival_slice = int(np.round(arrival_time/time_slice_duration))

            if arrival_slice < number_time_slices and arrival_slice >= 0:

                for p in range(projection_lengths[phs_lixel]):
                    pp = projection_starts[phs_lixel] + p
                    pixel = projection_links[pp]
                    out_image_sequence[arrival_slice, pixel] += 1

    return out_image_sequence
