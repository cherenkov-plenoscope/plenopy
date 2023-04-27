import numpy as np
from array import array
from .tools.acp_format import gz_transparent_open


class RawLightFieldSensorResponse(object):
    """
    The raw sensor-response of the Atmospheric-Cherenkov-Plenoscope.

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

    def __init__(self, path):
        """
        Parameter
        ---------
        path        path to raw light field response in photon-stream (phs)
                    format.
        """

        self.NEXT_READOUT_CHANNEL_MARKER = 255
        with gz_transparent_open(path, "rb") as f:
            raw = read(f=f)
            self.time_slice_duration = raw["time_slice_duration"]
            self.number_lixel = raw["number_lixel"]
            self.number_time_slices = raw["number_time_slices"]
            self.number_symbols = raw["number_symbols"]
            self.photon_stream = raw["photon_stream"]
            self.number_photons = self.photon_stream.shape[0] - (
                self.number_lixel - 1
            )

    def todict(self):
        out = {}
        out["time_slice_duration"] = self.time_slice_duration
        out["number_lixel"] = self.number_lixel
        out["number_time_slices"] = self.number_time_slices
        out["number_symbols"] = self.number_symbols
        out["photon_stream"] = self.photon_stream
        return out

    def __repr__(self):
        exposure_time = self.number_time_slices * self.time_slice_duration
        out = "RawLightFieldSensorResponse("
        out += (
            "exposure time " + str(np.round(exposure_time * 1e9, 1)) + "ns, "
        )
        out += str(self.number_lixel) + " lixel, "
        out += "time slice duration "
        out += str(np.round(self.time_slice_duration * 1e12)) + "ps)"
        return out

"""
The raw sensor-response of the Atmospheric-Cherenkov-Plenoscope.

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
    return out


def number_photons(raw):
    return raw["photon_stream"].shape[0] - raw["number_lixel"] - 1


def write(f, raw):
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
