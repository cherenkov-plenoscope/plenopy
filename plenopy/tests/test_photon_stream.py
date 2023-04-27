import numpy as np
import plenopy as pl

NEXT_READOUT_CHANNEL_MARKER = 255
time_slice_duration = 0.5e-9


def test_cx_cy_t_point_cloud():

    number_lixel = 10
    cx = np.arange(0, number_lixel)
    cy = np.arange(0, number_lixel)
    time_delay = np.zeros(number_lixel)

    photon_stream = np.array(
        [
            1,
            2,
            3,
            4,
            NEXT_READOUT_CHANNEL_MARKER,
            1,
            21,
            NEXT_READOUT_CHANNEL_MARKER,
            NEXT_READOUT_CHANNEL_MARKER,
            1,
            2,
            3,
            NEXT_READOUT_CHANNEL_MARKER,
            1,
            NEXT_READOUT_CHANNEL_MARKER,
            2,
            NEXT_READOUT_CHANNEL_MARKER,
            4,
            5,
            6,
            NEXT_READOUT_CHANNEL_MARKER,
            67,
            NEXT_READOUT_CHANNEL_MARKER,
            7,
            7,
            7,
            7,
            NEXT_READOUT_CHANNEL_MARKER,
        ],
        dtype=np.uint8,
    )

    (
        xyt,
        lix,
    ) = pl.photon_stream.cython_reader.stream2_cx_cy_arrivaltime_point_cloud(
        photon_stream=photon_stream,
        time_slice_duration=time_slice_duration,
        NEXT_READOUT_CHANNEL_MARKER=NEXT_READOUT_CHANNEL_MARKER,
        cx=cx,
        cy=cy,
        time_delay=time_delay,
    )

    number_photons = photon_stream.shape[0] - (number_lixel - 1)

    print(lix)

    assert xyt.shape[0] == number_photons
    assert xyt.shape[1] == 3
    assert lix.shape[0] == number_photons

    # Test photon to lixel-id assignment
    assert lix[0] == 0
    assert lix[1] == 0
    assert lix[2] == 0
    assert lix[3] == 0
    assert lix[4] == 1
    assert lix[5] == 1
    assert lix[6] == 3
    assert lix[7] == 3
    assert lix[8] == 3
    assert lix[9] == 4
    assert lix[10] == 5
    assert lix[11] == 6

    # photon 0
    assert xyt[0, 0] == cx[0]
    assert xyt[0, 1] == cy[0]
    assert np.isclose(xyt[0, 2], 1 * time_slice_duration)

    # photon 1
    assert xyt[1, 0] == cx[0]
    assert xyt[1, 1] == cy[0]
    assert np.isclose(xyt[1, 2], 2 * time_slice_duration)

    # photon 2
    assert xyt[2, 0] == cx[0]
    assert xyt[2, 1] == cy[0]
    assert np.isclose(xyt[2, 2], 3 * time_slice_duration)

    # photon 3
    assert xyt[3, 0] == cx[0]
    assert xyt[3, 1] == cy[0]
    assert np.isclose(xyt[3, 2], 4 * time_slice_duration)

    # photon 4
    assert xyt[4, 0] == cx[1]
    assert xyt[4, 1] == cy[1]
    assert np.isclose(xyt[4, 2], 1 * time_slice_duration)

    # photon 5
    assert xyt[5, 0] == cx[1]
    assert xyt[5, 1] == cy[1]
    assert np.isclose(xyt[5, 2], 21 * time_slice_duration)
