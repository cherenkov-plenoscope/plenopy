import pytest
import numpy as np
import plenopy as plp

def test_binning_flat_voxel_center_positions():
    binning = plp.Tomography.Binning(
        z_min=0.0,
        z_max=1e3,
        number_z_bins=101,
        xy_diameter=18.0,
        number_xy_bins=33)

    xyz_flat = binning.flat_xyz_voxel_positions()

    i = 0
    for x in range(binning.number_xy_bins):
        for y in range(binning.number_xy_bins):
            for z in range(binning.number_z_bins):
                assert xyz_flat[i,0] == binning.xy_bin_centers[x]
                assert xyz_flat[i,1] == binning.xy_bin_centers[y]
                assert xyz_flat[i,2] == binning.z_bin_centers[z]
                i+=1


def test_matrix_2_image_shape():
    matrix2d = np.eye(3)
    image_bgr = plp.plot.Tomography.matrix_2_rgb_image(matrix=matrix2d)

    assert matrix2d.shape[0] == image_bgr.shape[0]
    assert matrix2d.shape[1] == image_bgr.shape[1]
    assert matrix2d.shape == (3,3)
    assert image_bgr.shape == (3,3,3)


def test_matrix_2_image_rectangular_shape():
    matrix2d = np.ones(shape=(16,9))
    image_bgr = plp.plot.Tomography.matrix_2_rgb_image(matrix=matrix2d)

    assert matrix2d.shape[0] == image_bgr.shape[0]
    assert matrix2d.shape[1] == image_bgr.shape[1]
    assert matrix2d.shape == (16,9)
    assert image_bgr.shape == (16,9,3)


def test_matrix_2_image_color_channel():
    for color_channel in range(3):
        matrix2d = np.eye(3)
        image_bgr = plp.plot.Tomography.matrix_2_rgb_image(
            matrix=matrix2d,
            color_channel=color_channel)

        # again test shape
        assert matrix2d.shape[0] == image_bgr.shape[0]
        assert matrix2d.shape[1] == image_bgr.shape[1]
        assert matrix2d.shape == (3,3)
        assert image_bgr.shape == (3,3,3)

        # here is the matrix content
        assert image_bgr[0,0,color_channel] == 1.0
        assert image_bgr[1,1,color_channel] == 1.0
        assert image_bgr[2,2,color_channel] == 1.0

        # here is the background
        assert image_bgr[0,1,color_channel] == 0
        assert image_bgr[0,2,color_channel] == 0
        assert image_bgr[1,0,color_channel] == 0
        assert image_bgr[1,2,color_channel] == 0
        assert image_bgr[2,0,color_channel] == 0
        assert image_bgr[2,1,color_channel] == 0

def test_matrix_2_image_min_max():
    color_channel = 1
    matrix2d = np.diag([0,0, 0.365, 1.0])
    image_bgr = plp.plot.Tomography.matrix_2_rgb_image(
        matrix=matrix2d,
        color_channel=color_channel)

    assert matrix2d[0,0] == image_bgr[0,0,color_channel]
    assert matrix2d[1,1] == image_bgr[1,1,color_channel]
    assert matrix2d[2,2] == image_bgr[2,2,color_channel]