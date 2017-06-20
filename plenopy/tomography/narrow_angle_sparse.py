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
import tqdm
import os
import pickle
import shelve
from scipy.sparse import coo_matrix
from .filtered_back_projection import max_intensity_vs_z
from .filtered_back_projection import histogram


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

        self.psf, self.number_of_voxels_in_psf_per_voxel = cached_tomographic_point_spread_function(
            rays=rays,
            binning=self.binning,
            path=self._psf_cache_dir)

        self.psf_mask = mask_voxels_with_minimum_number_of_rays(
            psf=self.psf,
            ray_threshold=ray_threshold)

        ray_count_hist = histogram(
            rays=self.rays,
            binning=self.binning)
        self.max_ray_count_vs_z = max_intensity_vs_z(ray_count_hist)

        self.max_ray_count_vs_z = precompute__max_ray_count_vs_z(
            psf=self.psf,
            binning=self.binning,
            max_ray_count_vs_z=self.max_ray_count_vs_z
        )

        self.max_ray_count_vs_z = self.max_ray_count_vs_z[self.psf_mask]
        self.psf = self.psf[self.psf_mask]
        self.number_of_voxels_in_psf_per_voxel = self.number_of_voxels_in_psf_per_voxel[self.psf_mask]
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

        foo = np.zeros(self.binning.number_bins, dtype=np.float32)
        foo[self.psf_mask] = self.rec_I_vol[:]

        self.intensity_volume = flat_volume_intensity_3d_reshape(
            vol_I=foo,
            binning=self.binning)


def tomographic_point_spread_function_sparse(rays, binning, show_progress=False):
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

    show_progress   Prints a progressbar to std out. (Default: False)

    Here 'rays_in_voxels' is the estimate of which rays participate to which
    volume cells (voxels).
    """
    number_of_voxels = binning.number_bins
    number_of_rays = rays.support.shape[0]

    intersections = rays.xy_intersections_in_object_distance(binning.z_bin_centers)

    x_bins = np.digitize(x=intersections[:, :, 0], bins=binning.xy_bin_edges)-1
    y_bins = np.digitize(x=intersections[:, :, 1], bins=binning.xy_bin_edges)-1

    list_of_ray_ids = []
    list_of_voxel_ids = []

    for ray_id in range(number_of_rays):
        x_bins_of_ray = x_bins[ray_id]
        y_bins_of_ray = y_bins[ray_id]
        z_bins_of_ray = np.arange(len(binning.z_bin_centers))

        good = (x_bins_of_ray >= 0) & (x_bins_of_ray < binning.number_xy_bins)
        good &= (y_bins_of_ray >= 0) & (y_bins_of_ray < binning.number_xy_bins)

        voxel_indices = np.ravel_multi_index(
            [
                x_bins_of_ray[good],
                y_bins_of_ray[good],
                z_bins_of_ray[good]
            ],
            dims=binning.dims,
            order='F'
        )

        list_of_voxel_ids.append(voxel_indices)
        list_of_ray_ids.append(
            np.ones_like(voxel_indices) * ray_id
        )

    voxel_ids = np.concatenate(list_of_voxel_ids)
    ray_ids = np.concatenate(list_of_ray_ids)

    psf = coo_matrix(
        (
            np.ones_like(voxel_ids),
            (voxel_ids, ray_ids)
        ),
        shape=(number_of_voxels, number_of_rays)
    )

    return psf.tocsr()


def update_narrow_beam(
    vol_I,
    measured_I,
    psf,
    max_ray_count_vs_z,
    number_of_voxels_in_psf_per_voxel,
):

    measured_I_of_voxel = (
            psf.dot(measured_I) * max_ray_count_vs_z /
            number_of_voxels_in_psf_per_voxel
            )

    proj_I_of_voxel = psf.dot(psf.T.dot(vol_I)) / number_of_voxels_in_psf_per_voxel

    voxel_diffs = measured_I_of_voxel - proj_I_of_voxel

    vol_I += voxel_diffs
    vol_I[vol_I < 0.0] = 0.0

    return vol_I


def mask_voxels_with_minimum_number_of_rays(psf, ray_threshold):
    mask = psf.sum(axis=1).A[:, 0] > ray_threshold
    return mask


def flat_volume_intensity_3d_reshape(vol_I, binning):
    return vol_I.reshape((
        binning.number_xy_bins,
        binning.number_xy_bins,
        binning.number_z_bins), order='F')


def cached_tomographic_point_spread_function(
    rays,
    binning,
    show_progress=False,
    path='.'
):
    """
    Caches the tomographic point spread function.

    Parameters
    ----------

    path                The path to the directory where the hidden cache files
                        shall be written to.
    """
    with shelve.open(path+'_psf.shelve') as db:
        if 'psf' not in db:
            db['psf'] = tomographic_point_spread_function_sparse(
                rays=rays,
                binning=binning,
                show_progress=show_progress)

        psf = db['psf']

        # ~12min on my machine (DN)
        if 'number_of_voxels_in_psf_per_voxel' not in db:
            db['number_of_voxels_in_psf_per_voxel'] = precompute__number_of_voxels_in_psf(psf, chunksize=2000)

        return psf, db['number_of_voxels_in_psf_per_voxel']


def precompute__number_of_voxels_in_psf(psf, chunksize=1000):
    n = np.zeros(psf.shape[0])
    n_chunks = int(np.ceil(psf.shape[0] / chunksize))
    for i in range(n_chunks):
        chunk = slice(
            i*chunksize,
            (i+1)*chunksize if (i+1)*chunksize < psf.shape[0] else None)
        x = psf.dot(psf[chunk].T).sum(axis=0)
        n[chunk] = x[:]
    return n


def precompute__max_ray_count_vs_z(psf, binning, max_ray_count_vs_z):
    '''
    A normalization factor needed when reconstructing
    directly in cartesian space to counter act the thinning
    of rays with rising altitude.
    '''
    flat_voxel_indices = np.arange(psf.shape[0])
    voxel_z_indices = np.unravel_index(
        flat_voxel_indices,
        dims=binning.dims,
        order='F'
    )[2]
    return max_ray_count_vs_z[voxel_z_indices]**(1/3)

