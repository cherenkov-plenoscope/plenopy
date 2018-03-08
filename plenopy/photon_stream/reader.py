import numpy as np

def py_stream2sequence(
    photon_stream,
    time_slice_duration,
    NEXT_READOUT_CHANNEL_MARKER,
    sequence,
    time_delay_mean):

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
