import pytest
import numpy as np
import plenopy as pl
import pkg_resources

def test_array_chnunking():
    chunking = pl.tomography.plot_xyzI._start_and_end_slices_for_1D_array_chunking

    (s,e) = chunking(number_of_chunks=10, array_length=10)
    assert len(s) == 10
    assert len(e) == 10
    for i in range(len(s)):
        assert s[i] < e[i]

    (s,e) = chunking(number_of_chunks=10, array_length=100)
    assert len(s) == 10
    assert len(e) == 10
    for i in range(len(s)):
        assert s[i] < e[i]


def test_binning_flat_voxel_center_positions():
    binning = pl.tomography.Binning(
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
    image_bgr = pl.tomography.plot.matrix_2_rgb_image(matrix=matrix2d)

    assert matrix2d.shape[0] == image_bgr.shape[0]
    assert matrix2d.shape[1] == image_bgr.shape[1]
    assert matrix2d.shape == (3,3)
    assert image_bgr.shape == (3,3,3)


def test_matrix_2_image_rectangular_shape():
    matrix2d = np.ones(shape=(16,9))
    image_bgr = pl.tomography.plot.matrix_2_rgb_image(matrix=matrix2d)

    assert matrix2d.shape[0] == image_bgr.shape[0]
    assert matrix2d.shape[1] == image_bgr.shape[1]
    assert matrix2d.shape == (16,9)
    assert image_bgr.shape == (16,9,3)


def test_matrix_2_image_color_channel():
    for color_channel in range(3):
        matrix2d = np.eye(3)
        image_bgr = pl.tomography.plot.matrix_2_rgb_image(
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
    image_bgr = pl.tomography.plot.matrix_2_rgb_image(
        matrix=matrix2d,
        color_channel=color_channel)

    assert matrix2d[0,0] == image_bgr[0,0,color_channel]
    assert matrix2d[1,1] == image_bgr[1,1,color_channel]
    assert matrix2d[2,2] == image_bgr[2,2,color_channel]


def prep_nat1_nat2():
    run_path = pkg_resources.resource_filename(
        'plenopy',
        'tests/resources/run.acp')
    r = pl.Run(run_path)
    evt = r[0]
    rays = pl.tomography.Rays.from_light_field_geometry(r.light_field_geometry)
    binning = pl.tomography.Binning(
        z_min=1e3,
        z_max=6e3,
        number_z_bins=16,
        xy_diameter=1e3,
        number_xy_bins=16
    )

    t0 = pl.light_field.sequence.time_slice_with_max_intensity(
        evt.light_field.sequence
    )
    i = evt.light_field.sequence[t0-1:t0+2].sum(axis=0)

    nat = pl.tomography.narrow_angle.NarrowAngleTomography(
        rays=rays,
        intensities=i,
        binning=binning,
        ray_threshold=5
    )

    new_nat = pl.tomography.narrow_angle_sparse.NarrowAngleTomography(
        rays=rays,
        intensities=i,
        binning=binning,
        ray_threshold=5
    )

    return nat, new_nat, binning


def test_psf_masks():
    nat1, nat2, _ = prep_nat1_nat2()

    assert (nat1.psf_mask == nat2.psf_mask).all()


def test_number_of_voxels():
    nat1, nat2, _ = prep_nat1_nat2()

    nvs = []
    for voxel_id in np.arange(len(nat1.psf))[nat1.psf_mask]:
        nv = pl.tomography.narrow_angle.calc_number_of_voxels(
            voxel_id,
            nat1.psf,
            nat1.i_psf
        )
        nvs.append(nv)
    nvs = np.array(nvs)

    assert len(nvs) == len(nat2.number_of_voxels_in_psf_per_voxel)
    assert (nvs == nat2.number_of_voxels_in_psf_per_voxel).all()


def test_measured_intensities():
    nat1, nat2, binning = prep_nat1_nat2()

    mi = []
    for voxel_id in np.arange(len(nat1.psf))[nat1.psf_mask]:
        rays_in_voxel = nat1.psf[voxel_id]
        i = nat1.intensities[rays_in_voxel].sum()
        mi.append(i)
    mi = np.array(mi)

    mi2 = nat2.psf.dot(nat2.intensities)

    print(mi[:10])
    print(mi2[:10])
    assert (mi == mi2).all()


def test_measured_intensities_weighted():
    nat1, nat2, binning = prep_nat1_nat2()

    mi = []
    for voxel_id in np.arange(len(nat1.psf))[nat1.psf_mask]:
        rays_in_voxel = nat1.psf[voxel_id]
        i = nat1.intensities[rays_in_voxel].sum()
        voxel_z_index = pl.tomography.narrow_angle.flat_voxel_index_to_z_index(
            voxel_id, binning)
        image_2_cartesian_norm = (nat1.max_ray_count_vs_z[voxel_z_index])**(1/3)
        i *= image_2_cartesian_norm
        mi.append(i)

    mi = np.array(mi)

    mi2 = nat2.psf.dot(nat2.intensities) * nat2.max_ray_count_vs_z

    print(mi[:10])
    print(mi2[:10])
    assert (mi == mi2).all()
