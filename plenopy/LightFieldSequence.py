import numpy as np
from .cython_tools import stream2sequence

class LightFieldSequence(object):

    def __init__(
        self, 
        raw_light_field_sensor_response, 
        lixel_statistics, 
        plenoscope_geometry):
        """
        Parameters
        ----------
        raw_light_field_sensor_response 

        lixel_statistics

        plenoscope_geometry
        """
        self.__dict__ = lixel_statistics.__dict__.copy()
        self.__doc__ = """
        sequence        The light field sequence
        """
        self.__doc__ += lixel_statistics.__doc__

        self.plenoscope_geometry = plenoscope_geometry
        self.number_time_slices = raw_light_field_sensor_response.number_time_slices
        self.time_slice_duration = raw_light_field_sensor_response.time_slice_duration
    
        self.sequence = np.zeros(
            shape=(
                raw_light_field_sensor_response.number_time_slices,
                raw_light_field_sensor_response.number_lixel), 
            dtype=np.uint16)


        time_delay_mean = self.time_delay_mean.copy()
        #print('time_delay_mean flags', time_delay_mean.flags)

        stream2sequence(
            photon_stream=raw_light_field_sensor_response.photon_stream,
            time_slice_duration=raw_light_field_sensor_response.time_slice_duration,
            NEXT_READOUT_CHANNEL_MARKER=raw_light_field_sensor_response.NEXT_READOUT_CHANNEL_MARKER,
            sequence=self.sequence,
            time_delay_mean=time_delay_mean)

    def _to_image_sequence(self, axis=1):
        if axis==1:
            bins = self.number_pixel
        else:
            bins = self.number_paxel

        imgs = np.zeros(
            shape=(self.number_time_slices, bins), 
            dtype=np.uint16)

        for t in range(self.number_time_slices):
            imgs[t,:] = self.sequence[t,:].reshape(
                self.number_pixel, 
                self.number_paxel).sum(axis=axis)

        return imgs

    def pixel_sequence(self):
        return self._to_image_sequence(axis=1)

    def paxel_sequence(self):
        return self._to_image_sequence(axis=0)

    def __repr__(self):
        out = 'LightField('
        out += str(self.number_lixel) + ' lixel, '
        out += str(self.number_pixel) + ' pixel, '
        out += str(self.number_paxel) + ' paxel'
        out += ')\n'
        return out


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