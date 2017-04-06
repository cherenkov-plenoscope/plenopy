import numpy as np
from array import array


class RawLightFieldSensorResponse(object):
    """
    The raw sensor response of the Atmospheric Cherenkov Plenoscope.

    photon_stream           A stream of arrival time slices of photons separated
                            by a delimiter symbol to indicate the line break 
                            into the next read out channel (lixel).

    time_slice_duration     The duration of one time slice in the photon-stream.

    number_lixel            The number of read out channels (lixels).

    number_time_slices      The number of time slices of the photon-stream.

    number_symbols          The number of pulses plus the number of lixels. This
                            is the size of the photon-stream payload in bytes.
    """

    def __init__(self, path):
        """
        Parameter
        ---------
        path        path to raw light field response in photon-stream (phs) 
                    format.
        """

        self.NEXT_READOUT_CHANNEL_MARKER = 255

        with open(path, 'rb') as f:

            # HEADER
            # ------
            self.time_slice_duration = np.fromstring(
                f.read(4), 
                dtype=np.float32, 
                count=1)[0]

            self.number_lixel = np.fromstring(
                f.read(4), 
                dtype=np.uint32, 
                count=1)[0]

            self.number_time_slices = np.fromstring(
                f.read(4), 
                dtype=np.uint32, 
                count=1)[0]

            self.number_symbols = np.fromstring(
                f.read(4), 
                dtype=np.uint32, 
                count=1)[0]

            # PAYLOAD
            # -------
            self.photon_stream = np.fromstring(
                f.read(self.number_symbols),
                dtype=np.uint8)

    def __repr__(self):
        exposure_time = self.number_time_slices*self.time_slice_duration
        out = 'RawLightFieldSensorResponse('
        out += 'exposure time '+str(np.round(exposure_time*1e9, 1)) + 'ns, '
        out += str(self.number_lixel) + ' lixel, '
        out += 'time slice duration '
        out += str(np.round(self.time_slice_duration*1e12))+'ps)'
        return out