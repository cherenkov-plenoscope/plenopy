"""
The raw response of the Cherenkov-Plenoscope's sensor.

photon_stream           A stream of arrival-time-slices of photons
                        separated by a delimiter symbol to indicate the
                        next read-out-channel (lixel).

time_slice_duration     The duration of one time-slice in the
                        photon-stream.

number_photons          The number of photons in the stream.

number_lixel            The number of read-out-channels (lixels).

number_time_slices      The number of time-slices of the photon-stream.

number_symbols          The number of pulses plus the number of lixels.
                        This is the size of the photon-stream in bytes.
"""

import numpy as np


NEXT_READOUT_CHANNEL_MARKER = 255


def read(f):
    out = {}
    out["time_slice_duration"] = np.frombuffer(
        f.read(4), dtype=np.float32, count=1
    )[0]
    out["number_lixel"] = np.frombuffer(f.read(4), dtype=np.uint32, count=1)[0]
    out["number_time_slices"] = np.frombuffer(
        f.read(4), dtype=np.uint32, count=1
    )[0]
    out["number_symbols"] = np.frombuffer(f.read(4), dtype=np.uint32, count=1)[
        0
    ]
    out["photon_stream"] = np.frombuffer(
        f.read(out["number_symbols"]), dtype=np.uint8
    )

    out["number_photons"] = out["photon_stream"].shape[0] - (
        out["number_lixel"] - 1
    )
    return out


def write(f, raw_sensor_response):
    raw = raw_sensor_response
    assert isinstance(raw["time_slice_duration"], np.float32)
    f.write(raw["time_slice_duration"].tobytes())

    assert isinstance(raw["number_lixel"], np.uint32)
    f.write(raw["number_lixel"].tobytes())

    assert isinstance(raw["number_time_slices"], np.uint32)
    f.write(raw["number_time_slices"].tobytes())

    assert isinstance(raw["number_symbols"], np.uint32)
    f.write(raw["number_symbols"].tobytes())

    assert isinstance(raw["photon_stream"], np.ndarray)
    assert raw["photon_stream"].dtype == np.uint8
    f.write(raw["photon_stream"].tobytes())


def is_euqal(a, b):
    if a["time_slice_duration"] != b["time_slice_duration"]:
        return False

    if a["number_lixel"] != b["number_lixel"]:
        return False

    if a["number_time_slices"] != b["number_time_slices"]:
        return False

    if a["number_symbols"] != b["number_symbols"]:
        return False

    if a["number_photons"] != b["number_photons"]:
        return False

    for s in range(a["number_symbols"]):
        if a["photon_stream"][s] != b["photon_stream"][s]:
            return False

    return True
