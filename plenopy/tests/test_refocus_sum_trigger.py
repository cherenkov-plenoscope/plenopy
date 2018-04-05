import pytest
import numpy as np
import plenopy as pl
import pkg_resources


def test_empty_lists():
    els = pl.trigger.list_of_empty_lists(42)
    assert len(els) == 42
    for el in els:
        assert len(el) == 0


def test_neighborhood():
    #  /\y
    #  | (0,1)3   (1,1)2
    #  |
    #  | (0,0)0   (1,0)1
    #  ---------------> x
    x = [0, 1, 1, 0]
    y = [0, 0, 1, 1]
    epsilon = 1.0
    nn = pl.trigger.neighborhood(x=x, y=y, epsilon=epsilon, itself=False)
    assert nn.shape[0] == 4
    assert nn.shape[1] == 4
    # itself is False
    assert nn[0, 0] == False
    assert nn[1, 1] == False
    assert nn[2, 2] == False
    assert nn[3, 3] == False

    assert nn[0, 1] == True
    assert nn[0, 3] == True
    assert nn[0, 2] == False

    assert nn[1, 0] == True
    assert nn[1, 2] == True
    assert nn[1, 3] == False

    for ix in range(nn.shape[0]):
        for iy in range(nn.shape[1]):
            assert nn[ix, iy] == nn[iy, ix]

    nni = pl.trigger.neighborhood(x=x, y=y, epsilon=epsilon, itself=True)

    # itself is True
    assert nni[0, 0] == True
    assert nni[1, 1] == True
    assert nni[2, 2] == True
    assert nni[3, 3] == True

    for ix in range(nni.shape[0]):
        for iy in range(nni.shape[1]):
            assert nni[ix, iy] == nni[iy, ix]



def test_max_number_neighboring_trigger_patches():
    #
    #  0  1  2
    #  3  4  5
    #  6  7  8
    #
    x = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    epsilon = 1.1
    nn = pl.trigger.neighborhood(x=x, y=y, epsilon=epsilon, itself=False)

    def assert_number_of_active_neighboring_trigger_patches(
        patch_mask,
        number_active_neighbors,
        number_active_neighbors_and_active_itself,
        max_active_neighbors,
        argmax_active_neighbors,
        neighborhood_of_pixel
    ):
        print(patch_mask)
        patch_mask = np.array(patch_mask).flatten()
        number_active_neighbors = np.array(number_active_neighbors).flatten()
        number_active_neighbors_and_active_itself = np.array(
            number_active_neighbors_and_active_itself).flatten()

        np.testing.assert_array_equal(
            pl.trigger.number_of_active_neighboring_patches(
                patch_mask=patch_mask.astype(np.bool),
                neighborhood_of_pixel=neighborhood_of_pixel),
            number_active_neighbors)

        np.testing.assert_array_equal(
            pl.trigger.number_of_active_neighboring_patches_and_active_itself(
                patch_mask=patch_mask.astype(np.bool),
                neighborhood_of_pixel=neighborhood_of_pixel),
            number_active_neighbors_and_active_itself)

        assert (
            pl.trigger.max_number_of_neighboring_trigger_patches(
            patch_mask=patch_mask.astype(np.bool),
            neighborhood_of_pixel=neighborhood_of_pixel) == max_active_neighbors
        )

        am, m = pl.trigger.argmax_number_of_active_neighboring_patches_and_active_itself(
            patch_mask=patch_mask.astype(np.bool),
            neighborhood_of_pixel=neighborhood_of_pixel)
        print(patch_mask)
        print(am, m)
        assert m == max_active_neighbors
        np.testing.assert_array_equal(
            am,
            argmax_active_neighbors)


    assert_number_of_active_neighboring_trigger_patches(
        patch_mask=[
            [0, 0, 0],
            [0, 0, 0],
            [0 ,0, 0]],
        number_active_neighbors=[
            [0, 0, 0],
            [0, 0, 0],
            [0 ,0, 0]],
        number_active_neighbors_and_active_itself=[
            [0, 0, 0],
            [0, 0, 0],
            [0 ,0, 0]],
        max_active_neighbors=0,
        argmax_active_neighbors=[],
        neighborhood_of_pixel=nn)

    assert_number_of_active_neighboring_trigger_patches(
        patch_mask=[
            [1, 1, 0],
            [1, 0, 0],
            [0 ,0, 0]],
        number_active_neighbors=[
            [2, 1, 1],
            [1, 2, 0],
            [1 ,0, 0]],
        number_active_neighbors_and_active_itself=[
            [2, 1, 0],
            [1, 0, 0],
            [0 ,0, 0]],
        max_active_neighbors=2,
        argmax_active_neighbors=[0,],
        neighborhood_of_pixel=nn)

    assert_number_of_active_neighboring_trigger_patches(
        patch_mask=[
            [0, 0, 0],
            [1, 1, 0],
            [0 ,0, 0]],
        number_active_neighbors=[
            [1, 1, 0],
            [1, 1, 1],
            [1 ,1, 0]],
        number_active_neighbors_and_active_itself=[
            [0, 0, 0],
            [1, 1, 0],
            [0 ,0, 0]],
        max_active_neighbors=1,
        argmax_active_neighbors=[3, 4,],
        neighborhood_of_pixel=nn)

    assert_number_of_active_neighboring_trigger_patches(
        patch_mask=[
            [0, 1, 0],
            [0, 1, 0],
            [0 ,0, 0]],
        number_active_neighbors=[
            [1, 1, 1],
            [1, 1, 1],
            [0 ,1, 0]],
        number_active_neighbors_and_active_itself=[
            [0, 1, 0],
            [0, 1, 0],
            [0 ,0, 0]],
        max_active_neighbors=1,
        argmax_active_neighbors=[1, 4,],
        neighborhood_of_pixel=nn)

    assert_number_of_active_neighboring_trigger_patches(
        patch_mask=[
            [0, 1, 0],
            [0, 1, 0],
            [0 ,1, 0]],
        number_active_neighbors=[
            [1, 1, 1],
            [1, 2, 1],
            [1 ,1, 1]],
        number_active_neighbors_and_active_itself=[
            [0, 1, 0],
            [0, 2, 0],
            [0 ,1, 0]],
        max_active_neighbors=2,
        argmax_active_neighbors=[4,],
        neighborhood_of_pixel=nn)

    assert_number_of_active_neighboring_trigger_patches(
        patch_mask=[
            [0, 1, 0],
            [1, 1, 1],
            [0 ,1, 0]],
        number_active_neighbors=[
            [2, 1, 2],
            [1, 4, 1],
            [2 ,1, 2]],
        number_active_neighbors_and_active_itself=[
            [0, 1, 0],
            [1, 4, 1],
            [0 ,1, 0]],
        max_active_neighbors=4,
        argmax_active_neighbors=[4,],
        neighborhood_of_pixel=nn)

    assert_number_of_active_neighboring_trigger_patches(
        patch_mask=[
            [0, 0, 1],
            [1, 1, 1],
            [1 ,0, 0]],
        number_active_neighbors=[
            [1, 2, 1],
            [2, 2, 2],
            [1 ,2, 1]],
        number_active_neighbors_and_active_itself=[
            [0, 0, 1],
            [2, 2, 2],
            [1 ,0, 0]],
        max_active_neighbors=2,
        argmax_active_neighbors=[3, 4, 5],
        neighborhood_of_pixel=nn)


def test_sliding_coincidence_window():
    sequence = np.array([], dtype=np.uint16)
    rs = pl.trigger.convole_sequence(
        sequence,
        integration_time_in_slices=5)
    assert rs.shape == sequence.shape

    np.testing.assert_array_equal(
        pl.trigger.convole_sequence(
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint16),
            integration_time_in_slices=5),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint16))

    np.testing.assert_array_equal(
        pl.trigger.convole_sequence(
        np.array([0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2], dtype=np.uint16),
            integration_time_in_slices=1),
        np.array([0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2], dtype=np.uint16))

    np.testing.assert_array_equal(
        pl.trigger.convole_sequence(
        np.array([0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2], dtype=np.uint16),
            integration_time_in_slices=2),
        np.array([1, 3, 5, 5, 3, 1, 1, 3, 5, 5, 2], dtype=np.uint16))

    np.testing.assert_array_equal(
        pl.trigger.convole_sequence(
        np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint16),
            integration_time_in_slices=5),
        np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint16))

    np.testing.assert_array_equal(
        pl.trigger.convole_sequence(
        np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint16),
            integration_time_in_slices=5),
        np.array([0, 0, 1, 2, 2, 2, 2, 1, 0, 0, 0], dtype=np.uint16))