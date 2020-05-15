import numpy as np
from ..photon_stream.cython_reader import photon_stream_to_image_sequence
from scipy.ndimage import convolve1d as scipy_ndimage_convolve1d


def estimate_trigger_image_sequences(
    raw_sensor_response,
    light_field_geometry,
    trigger_geometry,
    integration_time_in_slices=10,
):
    tg = trigger_geometry
    lfg = light_field_geometry

    assert tg['number_lixel'] == lfg.number_lixel

    time_integration_kernel = np.ones(
        integration_time_in_slices,
        dtype=np.uint16
    )

    foci_trigger_image_sequences = []
    for focus in range(tg['number_foci']):

        trigger_image_sequence = photon_stream_to_image_sequence(
            photon_stream=raw_sensor_response.photon_stream,
            photon_stream_next_channel_marker=raw_sensor_response.NEXT_READOUT_CHANNEL_MARKER,
            time_slice_duration=raw_sensor_response.time_slice_duration,
            time_delay_image_mean=lfg.time_delay_image_mean,
            projection_links=tg['foci'][focus]['links'],
            projection_starts=tg['foci'][focus]['starts'],
            projection_lengths=tg['foci'][focus]['lengths'],
            number_lixel=lfg.number_lixel,
            number_pixel=tg['image']['number_pixel'],
            number_time_slices=raw_sensor_response.number_time_slices,
        )

        trigger_image_sequence_integrated = scipy_ndimage_convolve1d(
            input=trigger_image_sequence,
            weights=time_integration_kernel,
            axis=0,
            mode='constant',
            cval=0
        )
        foci_trigger_image_sequences.append(trigger_image_sequence_integrated)

    return foci_trigger_image_sequences


def estimate_max_responses_from_trigger_image_sequences(
    foci_trigger_image_sequences
):
    out = []
    for focus in range(len(foci_trigger_image_sequences)):
        out.append(
            _find_max_response_in_image_sequence(
                image_sequence=foci_trigger_image_sequences[focus])
        )
    return out


def _find_max_response_in_image_sequence(image_sequence):
    PIXEL_AXIS = 1
    max_responses_across_pixels = np.max(
        image_sequence,
        axis=PIXEL_AXIS
    )
    time_slice = np.argmax(max_responses_across_pixels)
    response_pe = max_responses_across_pixels[time_slice]
    pixel = np.argmax(image_sequence[time_slice, :])
    return {
        "response_pe": int(response_pe),
        "time_slice": int(time_slice),
        "pixel": int(pixel),
    }
