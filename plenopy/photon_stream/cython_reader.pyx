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


cdef extern void c_stream2_cx_cy_arrivaltime_point_cloud (
    unsigned char* photon_stream,
    unsigned int photon_stream_length,
    unsigned char NEXT_READOUT_CHANNEL_MARKER,
    float* point_cloud,
    float* cx,
    float* cy,
    float* time_delay_mean,
    float time_slice_duration,
    unsigned int* lixel_ids)

@cython.boundscheck(False)
@cython.wraparound(False)
def stream2_cx_cy_arrivaltime_point_cloud(
    np.ndarray[unsigned char, ndim=1, mode="c"] photon_stream not None,
    float time_slice_duration,
    unsigned char NEXT_READOUT_CHANNEL_MARKER,
    cx,
    cy,
    time_delay):

    cdef unsigned int photon_stream_length
    photon_stream_length = photon_stream.shape[0]

    number_lixel = 1 + np.sum(photon_stream == NEXT_READOUT_CHANNEL_MARKER)

    number_photons = (photon_stream.shape[0] - (number_lixel - 1))
    cdef np.ndarray[float, mode = "c"] point_cloud = np.ascontiguousarray(
        np.zeros(3*number_photons, dtype=np.float32),
        dtype=np.float32)

    cdef np.ndarray[unsigned int, mode = "c"] lixel_ids = np.ascontiguousarray(
        np.zeros(number_photons, dtype=np.uint32),
        dtype=np.uint32)

    assert cx is not None
    assert len(cx) == number_lixel, 'There must be a cx for each lixel.'
    cdef np.ndarray[float, mode = "c"] _cx = np.ascontiguousarray(
        cx,
        dtype=np.float32)

    assert cy is not None
    assert len(cy) == number_lixel, 'There must be a cy for each lixel.'
    cdef np.ndarray[float, mode = "c"] _cy = np.ascontiguousarray(
        cy,
        dtype=np.float32)

    assert time_delay is not None
    assert len(time_delay) == number_lixel, 'There must be a time_delay for each lixel.'
    cdef np.ndarray[float, mode = "c"] _time_delay = np.ascontiguousarray(
        time_delay,
        dtype=np.float32)

    c_stream2_cx_cy_arrivaltime_point_cloud(
        &photon_stream[0],
        photon_stream_length,
        NEXT_READOUT_CHANNEL_MARKER,
        &point_cloud[0],
        &_cx[0],
        &_cy[0],
        &_time_delay[0],
        time_slice_duration,
        &lixel_ids[0])

    return point_cloud.reshape((number_photons, 3)), lixel_ids
