import numpy as np
cimport numpy as np
cimport cython

cdef extern void c_stream2sequence (
    unsigned char* photon_stream,
    unsigned int photon_stream_length,
    unsigned char NEXT_READOUT_CHANNEL_MARKER,
    unsigned short* sequence,
    unsigned int number_time_slices,
    unsigned int number_lixel,
    float *time_delay_mean,
    float time_slice_duration)

@cython.boundscheck(False)
@cython.wraparound(False)
def stream2sequence(
    np.ndarray[unsigned char, ndim=1, mode="c"] photon_stream not None,
    float time_slice_duration,
    unsigned char NEXT_READOUT_CHANNEL_MARKER,
    np.ndarray[unsigned short, ndim=2, mode="c"] sequence not None,
    np.ndarray[float, ndim=1, mode="c"] time_delay_mean not None):

    cdef unsigned int photon_stream_length
    photon_stream_length = photon_stream.shape[0]

    cdef unsigned int number_lixel
    number_lixel = sequence.shape[1]

    cdef unsigned int number_time_slices
    number_time_slices = sequence.shape[0]

    c_stream2sequence(
        &photon_stream[0],
        photon_stream_length,
        NEXT_READOUT_CHANNEL_MARKER,
        &sequence[0,0],
        number_time_slices,
        number_lixel,
        &time_delay_mean[0],
        time_slice_duration)

    return None