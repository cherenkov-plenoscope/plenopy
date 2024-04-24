import numpy as np
import plenopy as pl
import tempfile
import os


def test_list_of_list_to_array():
    lol = [
        [1, 2, 3],
        [2, 3],
        [3, 4, 5],
        [],
    ]
    arr = pl.trigger.utils.list_of_lists_to_arrays(lol)
    assert arr["starts"].shape[0] == 4
    assert arr["lengths"].shape[0] == 4

    assert arr["lengths"][0] == 3
    assert arr["lengths"][1] == 2
    assert arr["lengths"][2] == 3
    assert arr["lengths"][3] == 0

    assert arr["starts"][0] == 0
    assert arr["starts"][1] == 3
    assert arr["starts"][2] == 5
    assert arr["starts"][3] == 8

    lol_back = pl.trigger.utils.arrays_to_list_of_lists(
        starts=arr["starts"], lengths=arr["lengths"], links=arr["links"]
    )

    assert len(lol) == len(lol_back)
    for ii in range(len(lol)):
        assert len(lol[ii]) == len(lol_back[ii])
        for jj in range(len(lol[ii])):
            assert lol[ii][jj] == lol_back[ii][jj]


def test_invert_projection():
    number_lixel = 4
    number_pixel = 6
    lol = [
        [1, 2, 3],
        [2, 3],
        [3, 4, 5],
        [],
    ]
    assert len(lol) == number_lixel

    inv_lol = pl.trigger.geometry.invert_projection_matrix(
        lol, number_lixel=number_lixel, number_pixel=number_pixel
    )

    assert len(inv_lol) == number_pixel
    assert len(inv_lol[0]) == 0

    assert len(inv_lol[1]) == 1
    assert 0 in inv_lol[1]

    assert len(inv_lol[2]) == 2
    assert 0 in inv_lol[2]
    assert 1 in inv_lol[2]

    assert len(inv_lol[3]) == 3
    assert 0 in inv_lol[3]
    assert 1 in inv_lol[3]
    assert 2 in inv_lol[3]

    assert len(inv_lol[4]) == 1
    assert 2 in inv_lol[4]

    assert len(inv_lol[5]) == 1
    assert 2 in inv_lol[5]


def test_max_response_unique_maximum():
    number_time_slices = 5
    number_pixel = 6

    for tt in range(number_time_slices):
        for pp in range(number_pixel):
            image_sequence = np.zeros(
                shape=(number_time_slices, number_pixel), dtype=np.uint32
            )
            image_sequence[tt, pp] = 10
            assert image_sequence.shape[0] == number_time_slices
            assert image_sequence.shape[1] == number_pixel

            mres = pl.trigger.estimate._find_max_response_in_image_sequence(
                image_sequence=image_sequence
            )

            assert mres["response_pe"] == 10
            assert mres["time_slice"] == tt
            assert mres["pixel"] == pp


def test_io():
    trigger_geometry = {}
    tg = trigger_geometry
    tg["number_foci"] = np.uint32(2)
    tg["number_lixel"] = np.uint32(6)

    tg["image"] = {}
    tg["image"]["number_pixel"] = np.uint32(3)
    tg["image"]["pixel_cx_rad"] = np.array([-1, 0, 1])
    tg["image"]["pixel_cy_rad"] = np.array([1, 0, 1])
    tg["image"]["pixel_radius_rad"] = np.float32(2.2)
    tg["image"]["max_number_nearest_lixel_in_pixel"] = np.uint32(2)

    tg["foci"] = []

    projection = [
        [0],
        [1],
        [0, 1],
        [0, 1],
        [1],
        [1, 2],
    ]
    focus = pl.trigger.utils.list_of_lists_to_arrays(projection)
    focus["object_distance_m"] = np.float32(1e3)
    tg["foci"].append(focus)

    projection = [
        [0, 1],
        [1],
        [0, 2],
        [1, 2],
        [0],
        [0, 1],
    ]
    focus = pl.trigger.utils.list_of_lists_to_arrays(projection)
    focus["object_distance_m"] = np.float32(2e3)
    tg["foci"].append(focus)

    pl.trigger.geometry.assert_trigger_geometry_consistent(trigger_geometry=tg)

    with tempfile.TemporaryDirectory(prefix="test_plenopy_trigger") as tmp:
        pl.trigger.geometry.write(
            trigger_geometry=tg,
            path=os.path.join(
                tmp,
                "dummy" + pl.trigger.geometry.suggested_filename_extension(),
            ),
        )

        tg_back = pl.trigger.geometry.read(
            path=os.path.join(
                tmp,
                "dummy" + pl.trigger.geometry.suggested_filename_extension(),
            )
        )

    assert tg["image"]["number_pixel"] == tg_back["image"]["number_pixel"]
    assert (
        tg["image"]["pixel_radius_rad"] == tg_back["image"]["pixel_radius_rad"]
    )
    assert (
        tg["image"]["max_number_nearest_lixel_in_pixel"]
        == tg_back["image"]["max_number_nearest_lixel_in_pixel"]
    )
    np.testing.assert_array_equal(
        tg["image"]["pixel_cx_rad"],
        tg_back["image"]["pixel_cx_rad"],
    )
    np.testing.assert_array_equal(
        tg["image"]["pixel_cy_rad"],
        tg_back["image"]["pixel_cy_rad"],
    )

    for focus in range(tg["number_foci"]):
        assert (
            tg["foci"][focus]["object_distance_m"]
            == tg_back["foci"][focus]["object_distance_m"]
        )
        for key in ["starts", "lengths", "links"]:
            np.testing.assert_array_equal(
                tg["foci"][focus][key], tg_back["foci"][focus][key]
            )


def test_sliding_coincidence_window():
    sequence = np.array([], dtype=np.uint16)
    rs = pl.trigger.estimate.integrate_in_sliding_window(
        sequence, integration_time_slices=5
    )
    assert rs.shape == sequence.shape

    np.testing.assert_array_equal(
        pl.trigger.estimate.integrate_in_sliding_window(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint16),
            integration_time_slices=5,
        ),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        pl.trigger.estimate.integrate_in_sliding_window(
            np.array([0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2], dtype=np.uint16),
            integration_time_slices=1,
        ),
        np.array([0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        pl.trigger.estimate.integrate_in_sliding_window(
            np.array([0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2], dtype=np.uint16),
            integration_time_slices=2,
        ),
        np.array([1, 3, 5, 5, 3, 1, 1, 3, 5, 5, 2], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        pl.trigger.estimate.integrate_in_sliding_window(
            np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint16),
            integration_time_slices=5,
        ),
        np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        pl.trigger.estimate.integrate_in_sliding_window(
            np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint16),
            integration_time_slices=5,
        ),
        np.array([0, 0, 1, 2, 2, 2, 2, 1, 0, 0, 0], dtype=np.uint16),
    )
