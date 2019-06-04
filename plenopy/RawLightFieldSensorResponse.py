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
        with gz_transparent_open(path, 'rb') as f:
            # header
            # ------
            self.time_slice_duration = np.frombuffer(
                f.read(4),
                dtype=np.float32,
                count=1)[0]
            self.number_lixel = np.frombuffer(
                f.read(4),
                dtype=np.uint32,
                count=1)[0]
            self.number_time_slices = np.frombuffer(
                f.read(4),
                dtype=np.uint32,
                count=1)[0]
            self.number_symbols = np.frombuffer(
                f.read(4),
                dtype=np.uint32,
                count=1)[0]

            # raw photon-stream
            # -----------------
            self.photon_stream = np.frombuffer(
                f.read(self.number_symbols),
                dtype=np.uint8)

            self.number_photons = (
                self.photon_stream.shape[0] - (self.number_lixel - 1))

    def __repr__(self):
        exposure_time = self.number_time_slices*self.time_slice_duration
        out = 'RawLightFieldSensorResponse('
        out += 'exposure time '+str(np.round(exposure_time*1e9, 1)) + 'ns, '
        out += str(self.number_lixel) + ' lixel, '
        out += 'time slice duration '
        out += str(np.round(self.time_slice_duration*1e12))+'ps)'
        return out
