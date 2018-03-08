import numpy as np
from ..photon_stream.cython_reader import stream2sequence


class LightField(object):
    def __init__(self, raw_light_field_sensor_response, lixel_statistics):
        """
        Parameters
        ----------
        raw_light_field_sensor_response

        lixel_statistics
        """
        self.__dict__ = lixel_statistics.__dict__.copy()
        self.__doc__ = """
    The 5 dimensional Light Field Sequence recorded by the
    Atmospheric Cherenkov Plenoscope (ACP).

    sequence        A sequence of light fields.
        """
        self.__doc__ += lixel_statistics.__doc__

        raw = raw_light_field_sensor_response
        self.number_time_slices = raw.number_time_slices
        self.time_slice_duration = raw.time_slice_duration

        self.sequence = np.zeros(
            shape=(
                raw.number_time_slices,
                raw.number_lixel),
            dtype=np.uint16)

        stream2sequence(
            photon_stream=raw.photon_stream,
            time_slice_duration=raw.time_slice_duration,
            NEXT_READOUT_CHANNEL_MARKER=raw.NEXT_READOUT_CHANNEL_MARKER,
            sequence=self.sequence,
            time_delay_mean=self.time_delay_mean)

    def _to_image_sequence(self, axis=1):
        if axis == 1:
            bins = self.number_pixel
        else:
            bins = self.number_paxel

        imgs = np.zeros(
            shape=(self.number_time_slices, bins),
            dtype=np.uint16)

        for t in range(self.number_time_slices):
            imgs[t, :] = self.sequence[t, :].reshape(
                self.number_pixel,
                self.number_paxel).sum(axis=axis)

        return imgs

    def pixel_sequence(self):
        return self._to_image_sequence(axis=1)

    def paxel_sequence(self):
        return self._to_image_sequence(axis=0)

    def pixel_sequence_refocus(self, lixels2pixel):
        imgs = np.zeros(
            shape=(self.number_time_slices, self.number_pixel),
            dtype=np.uint16)

        for lix, lixel2pixel in enumerate(lixels2pixel):
            imgs[:, lixel2pixel] += self.sequence[:, lix]

        return imgs

    def __repr__(self):
        out = 'LightField('
        out += str(self.number_lixel) + ' lixel, '
        out += str(self.number_pixel) + ' pixel, '
        out += str(self.number_paxel) + ' paxel, '
        out += str(self.number_time_slices) + ' time slices'
        out += ')'
        return out
