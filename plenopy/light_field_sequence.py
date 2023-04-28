import numpy as np
from . import raw_light_field_sensor_response
from . import photon_stream


def make(raw_sensor_response, time_delays_to_be_subtracted):
    raw = raw_sensor_response
    lixel_sequence = np.zeros(
        shape=(raw["number_time_slices"], raw["number_lixel"]), dtype=np.uint16
    )
    photon_stream.cython_reader.stream2sequence(
        photon_stream=raw["photon_stream"],
        time_slice_duration=raw["time_slice_duration"],
        NEXT_READOUT_CHANNEL_MARKER=raw_light_field_sensor_response.NEXT_READOUT_CHANNEL_MARKER,
        sequence=lixel_sequence,
        time_delay_mean=time_delays_to_be_subtracted,
    )
    return lixel_sequence


def make_isochor_image(raw_sensor_response, time_delay_image_mean):
    return make(
        raw_sensor_response=raw_sensor_response,
        time_delays_to_be_subtracted=-1.0 * time_delay_image_mean,
    )


def make_isochor_aperture(time_delay_mean):
    return make(
        raw_sensor_response=raw_sensor_response,
        time_delays_to_be_subtracted=+1.0 * time_delay_mean,
    )


def make_raw(raw_sensor_response):
    zeros = np.zeros(raw_sensor_response["number_lixel"], dtype=np.float32)
    return make(
        raw_sensor_response=raw_sensor_response,
        time_delays_to_be_subtracted=zeros,
    )


def photon_arrival_times_and_lixel_ids(raw_sensor_response):
    """
    Returns (arrival_slices, lixel_ids) of photons.
    """
    (
        arrival_time_slices,
        lixel_ids,
    ) = photon_stream.cython_reader.arrival_slices_and_lixel_ids(
        raw_sensor_response
    )
    return (
        arrival_time_slices * raw_sensor_response["time_slice_duration"],
        lixel_ids,
    )
