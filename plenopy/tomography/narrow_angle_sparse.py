"""
This 'narrow angle tomography' or '3D deconvolution' is inspired by:

@article{levoy2006light,
    title={Light field microscopy},
    author={Levoy, Marc and Ng, Ren and Adams, Andrew and Footer, Matthew and Horowitz, Mark},
    journal={ACM Transactions on Graphics (TOG)},
    volume={25},
    number={3},
    pages={924--934},
    year={2006},
    publisher={ACM}
}
"""
import numpy as np
from scipy.sparse import coo_matrix
from .filtered_back_projection import max_intensity_vs_z
from .filtered_back_projection import histogram

import os
from joblib import Memory

cachedir = '/tmp/joblib_cache_plenopy'
os.makedirs(cachedir, exist_ok=True)
memory = Memory(cachedir=cachedir, verbose=0)


class NarrowAngleTomography(object):
    def __init__(
        self,
        rays,
        intensities,
        binning,
        ray_threshold=10,
        psf_cache_dir='.'
    ):
        self.rays = rays
        self.binning = binning
        self.intensities = intensities
        self.ray_threshold = ray_threshold
        self._psf_cache_dir = psf_cache_dir
        self.iteration = 0

        self.psf = tomographic_point_spread_function_sparse(
            rays=rays,
            binning=self.binning)

        self.psf_mask = mask_voxels_with_minimum_number_of_rays(
            psf=self.psf,
            ray_threshold=ray_threshold)

        self.number_of_voxels_in_psf_per_voxel = self.psf.dot(
            self.psf.T.sum(axis=1)
        ).A[:, 0][self.psf_mask]

        self.max_ray_count_vs_z = precompute__max_ray_count_vs_z(
            rays=self.rays,
            psf=self.psf,
            binning=self.binning,
        )[self.psf_mask]

        self.psf = self.psf[self.psf_mask]
        self.rec_I_vol = np.zeros(self.psf.shape[0], dtype=np.float32)

    def one_more_iteration(self):

        rec_I_vol_n = update_narrow_beam(
            vol_I=self.rec_I_vol,
            measured_I=self.intensities,
            psf=self.psf,
            max_ray_count_vs_z=self.max_ray_count_vs_z,
            number_of_voxels_in_psf_per_voxel=self.number_of_voxels_in_psf_per_voxel)

        self.rec_I_vol = rec_I_vol_n
        self.iteration += 1

        self.intensity_volume = flat_volume_intensity_3d_reshape(
            vol_I=unmask_intensity_volume(
                self.rec_I_vol,
                self.psf_mask,
                self.binning),
            binning=self.binning)


@memory.cache
def tomographic_point_spread_function_sparse(rays, binning):
    """
    returns a 2D scipy.sparse.coo_matrix of
        shape=(N_voxels, N_rays) and
        dtype=np.float32

    The index along the rows, i.e. along the axis-0 is the same index as
    one would use inside `plenopy.Binning.flat_xyz_voxel_positions`.
    The index along the columns, i.e. along axis-1 is the same index as used
    inside `plenopy.Rays`.

    The matrix contains the length of each ray inside a given voxel.


    Parameters
    ----------

    rays            The rays of a light field in cartesian space.

    binning         The binning of the cartesian volume above the principal
                    aperture plane.

    Here 'rays_in_voxels' is the estimate of which rays participate to which
    volume cells (voxels).
    """

    intersections = rays.intersections_with_xy_plane(
        binning.z_bin_centers
    ).reshape(-1, 3)
    # intersections = (n_rays * n_planes, 3)

    bins = np.ones_like(intersections, dtype=np.int32)
    bins[:, 0] = np.digitize(intersections[:, 0], binning.xy_bin_edges)-1
    bins[:, 1] = np.digitize(intersections[:, 1], binning.xy_bin_edges)-1
    bins[:, 2] = np.digitize(intersections[:, 2], binning.z_bin_edges)-1

    ray_ids = np.arange(rays.support.shape[0]).repeat(binning.number_z_bins)

    good = (bins[:, 0] >= 0) & (bins[:, 1] >= 0) & (bins[:, 2] >= 0)
    good &= (bins[:, 0] < binning.number_xy_bins)
    good &= (bins[:, 1] < binning.number_xy_bins)
    good &= (bins[:, 2] < binning.number_z_bins)

    bins = bins[good]
    ray_ids = ray_ids[good]

    voxel_ids = np.ravel_multi_index(
        bins.T,
        dims=binning.dims,
        order='F'
    )

    psf = coo_matrix(
        (
            np.ones_like(voxel_ids),
            (voxel_ids, ray_ids)
        ),
        shape=(
            binning.number_bins,
            rays.support.shape[0]
        ),
        dtype=np.float32
    )

    return psf.tocsr()


def update_narrow_beam(
    vol_I,
    measured_I,
    psf,
    max_ray_count_vs_z,
    number_of_voxels_in_psf_per_voxel,
):

    measured_I_of_voxel = (psf.dot(measured_I) * max_ray_count_vs_z)
    proj_I_of_voxel = psf.dot(psf.T.dot(vol_I))

    voxel_diffs = measured_I_of_voxel - proj_I_of_voxel
    voxel_diffs /= number_of_voxels_in_psf_per_voxel

    vol_I += voxel_diffs
    vol_I[vol_I < 0.0] = 0.0

    return vol_I


def mask_voxels_with_minimum_number_of_rays(psf, ray_threshold):
    mask = psf.sum(axis=1).A[:, 0] > ray_threshold
    return mask


def unmask_intensity_volume(vol_I, psf_mask, binning):
    foo = np.zeros(
        binning.number_bins,
        dtype=np.float32
    )
    foo[psf_mask] = vol_I[:]
    return foo


def flat_volume_intensity_3d_reshape(vol_I, binning):
    return vol_I.reshape(binning.dims, order='F')


def precompute__max_ray_count_vs_z(rays, psf, binning):
    '''
    A normalization factor needed when reconstructing
    directly in cartesian space to counter act the thinning
    of rays with rising altitude.
    '''
    ray_count_hist = histogram(
            rays=rays,
            binning=binning)

    max_ray_count_vs_z = max_intensity_vs_z(ray_count_hist)

    flat_voxel_indices = np.arange(psf.shape[0])
    voxel_z_indices = np.unravel_index(
        flat_voxel_indices,
        dims=binning.dims,
        order='F'
    )[2]
    return max_ray_count_vs_z[voxel_z_indices]**(1/3)