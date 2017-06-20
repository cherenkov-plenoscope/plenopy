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
from .filtered_back_projection import max_intensity_vs_z
from .filtered_back_projection import histogram


class NarrowAngleTomography(object):
    def __init__(
        self,
        rays,
        intensities,
        binning,
        ray_threshold=10,
        show_progress=False,
        psf_cache_dir='.'):

        self.rays = rays
        self.binning = binning
        self.intensities = intensities
        self.ray_threshold = ray_threshold
        self._psf_cache_dir = psf_cache_dir

        self.show_progress = show_progress

        if self.show_progress:
            print('Estimate tomographic point spread function')

        psf, i_psf = cached_tomographic_point_spread_function(
            rays=rays,
            binning=self.binning,
            show_progress=self.show_progress,
            path=self._psf_cache_dir)

        self.psf = psf
        self.i_psf = i_psf

        if self.show_progress:
            print('Estimate thinning of rays in cartesian space based on max number of rays in a voxel per altitude slice in reconstruction volume')

        ray_count_hist = histogram(
            rays=self.rays,
            binning=self.binning)
        self.max_ray_count_vs_z =  max_intensity_vs_z(ray_count_hist)

        if self.show_progress:
            print('Exclude voxels from reconstruction with less than '+str(ray_threshold)+' rays')

        self.psf_mask = mask_voxels_with_minimum_number_of_rays(
            psf=psf,
            ray_threshold=ray_threshold)

        if self.show_progress:
            print('Init empty intensity volume')

        self.rec_I_vol = np.zeros(self.binning.number_bins, dtype=np.float32)

        self.iteration = 0
        initial_diff = 0.0


    def one_more_iteration(self):
        if self.show_progress:
            print('Reconstruction iteration '+str(self.iteration))

        rec_I_vol_n = update_narrow_beam(
            vol_I=self.rec_I_vol,
            measured_I=self.intensities,
            psf=self.psf,
            psf_mask=self.psf_mask,
            i_psf=self.i_psf,
            max_ray_count_vs_z=self.max_ray_count_vs_z,
            binning=self.binning,
            show_progress=self.show_progress)

        diff = np.abs(rec_I_vol_n - self.rec_I_vol).sum()
        if self.iteration == 0:
            initial_diff = diff

        if self.show_progress:
            print('Intensity difference to previous iteration '+str(diff))

        self.rec_I_vol = rec_I_vol_n
        self.iteration += 1

        self.intensity_volume = flat_volume_intensity_3d_reshape(
            vol_I=self.rec_I_vol,
            binning=self.binning)


def tomographic_point_spread_function(rays, binning, show_progress=False):
    """
    Returns a list of lists of rays in voxels and a list of lists of voxels in
    rays.

    Parameters
    ----------

    rays            The rays of a light field in cartesian space.

    binning         The binning of the cartesian volume above the principal
                    aperture plane.

    show_progress   Prints a progressbar to std out. (Default: False)

    Here 'rays_in_voxels' is the estimate of which rays participate to which
    volume cells (voxels). The voxels are defined by the binning parameter.
    Here the estimate is rather crude in a binary fashion. Either a ray is
    assigned to a cell by 100 percent or by 0 percent. The second return value
    'voxels_in_rays' is the inverse mapping along the rays and the voxels
    assigned to these rays.
    """
    number_of_voxels = binning.number_bins
    number_of_rays = rays.support.shape[0]

    rays_in_voxels = []
    for i in range(number_of_voxels):
        rays_in_voxels.append([])

    voxels_in_rays = []
    for i in range(number_of_rays):
        voxels_in_rays.append([])

    for z_bin, z in tqdm.tqdm(enumerate(binning.z_bin_centers), disable=(not show_progress)):
        xys = rays.xy_intersections_in_object_distance(z)
        x_bins = np.digitize(x=xys[:,0], bins=binning.xy_bin_edges)
        y_bins = np.digitize(x=xys[:,1], bins=binning.xy_bin_edges)

        for i in range(number_of_rays):

            # np.digitize has no over and under flow bins, so we ignore the
            # lowest and uppermost bins
            if (x_bins[i] > 0 and
                x_bins[i] < binning.number_xy_bins-1 and
                y_bins[i] > 0 and
                y_bins[i] < binning.number_xy_bins-1):

                x_bin = x_bins[i] - 1
                y_bin = y_bins[i] - 1

                flat_voxel_index = (
                    x_bin +
                    y_bin*binning.number_xy_bins +
                    z_bin*binning.number_xy_bins*binning.number_xy_bins)

                rays_in_voxels[flat_voxel_index].append(i)
                voxels_in_rays[i].append(flat_voxel_index)

    return rays_in_voxels, voxels_in_rays


def update_narrow_beam(
    vol_I,
    measured_I,
    psf,
    psf_mask,
    i_psf,
    max_ray_count_vs_z,
    binning,
    show_progress=False):
    """
    Returns an updated copy of the intensitiy volume 'vol_I'.

    Parameters
    ----------

    vol_I               The current intensity volume. A 1D array with the
                        intensities of the voxels.

    measured_I          The measured intensity of the lighfield. A 1D array
                        with the intensities of the light field rays.

    psf                 A list of rays in the voxels.

    psf_mask            A boolean array of the voxels which shall be taken into
                        account. Here a threshold on a minimum number of rays
                        per voxel can be applied.

    i_psf               A list of voxels on the rays. The inverse representation
                        of 'psf'.

    max_ray_count_vs_z  A normalization factor needed when reconstructing
                        directly in cartesian space to counter act the thinning
                        of rays with rising altitude.

    binning             The binning of the cartesian volume above the principal
                        aperture plane. This binning MUST match the binning used
                        to create 'psf' and 'i_psf'.

    show_progress       Prints a progressbar to std out. (Default: False)
    """
    psf_mask_indicies = np.arange(len(psf))[psf_mask]

    i = 0
    voxel_diffs = np.zeros(vol_I.shape[0], dtype=np.float32)
    for voxel_index in tqdm.tqdm(psf_mask_indicies, disable=(not show_progress)):

        voxel_z_index = flat_voxel_index_to_z_index(voxel_index, binning)
        rays_in_voxel = psf[voxel_index]

        number_of_voxels_in_psf = 0
        for ray in rays_in_voxel:
            number_of_voxels_in_psf += len(i_psf[ray])

        measured_I_of_voxel = measured_I[rays_in_voxel].sum()/number_of_voxels_in_psf
        image_2_cartesian_norm = (max_ray_count_vs_z[voxel_z_index])**(1/3)
        measured_I_of_voxel *= image_2_cartesian_norm

        proj_I_of_voxel = 0.0
        for ray in rays_in_voxel:
            proj_I_of_voxel += vol_I[i_psf[ray]].sum()
        proj_I_of_voxel /= number_of_voxels_in_psf

        voxel_diffs[voxel_index] = (measured_I_of_voxel - proj_I_of_voxel)

        if np.mod(i, 1000) == 0:
            print('number_of_voxels_in_psf', number_of_voxels_in_psf)
            print('measured_I_of_voxel', measured_I_of_voxel)
            print('proj_I_of_voxel', proj_I_of_voxel)
            print('voxel_z_index', voxel_z_index)
            print('image_2_cartesian_norm', image_2_cartesian_norm)
        i += 1

    vol_I += voxel_diffs
    vol_I[vol_I < 0.0] = 0.0

    return vol_I


def flat_voxel_index_to_z_index(flat_index, binning):
    return flat_index // (binning.number_xy_bins*binning.number_xy_bins)


def mask_voxels_with_minimum_number_of_rays(psf, ray_threshold):
    mask = np.zeros(len(psf), dtype=np.bool8)
    for i, voxel_psf in enumerate(psf):
        if len(voxel_psf) > ray_threshold:
            mask[i] = True
    return mask


def flat_volume_intensity_3d_reshape(vol_I, binning):
    return vol_I.reshape((
        binning.number_xy_bins,
        binning.number_xy_bins,
        binning.number_z_bins), order='F')


def cached_tomographic_point_spread_function(rays, binning, show_progress=False, path='.'):
    """
    Caches the tomographic point spread function.

    Parameters
    ----------

    path                The path to the directory where the hidden cache files
                        shall be written to.
    """
    psf_path = path+'_psf.pkl'
    i_psf_path = path+'_i_psf.pkl'

    if not os.path.exists(psf_path) and not os.path.exists(i_psf_path):
        if show_progress:
            print('Estimate psf from scratch and write it to '+psf_path)

        psf, i_psf = tomographic_point_spread_function(
            rays=rays,
            binning=binning,
            show_progress=show_progress)
        with open(psf_path, 'wb') as f:
            pickle.dump(psf, f)

        with open(i_psf_path, 'wb') as f:
            pickle.dump(i_psf, f)
    else:
        if show_progress:
            print('Read in cached psf from '+psf_path)

        with open(psf_path, 'rb') as f:
            psf = pickle.load(f)

        with open(i_psf_path, 'rb') as f:
            i_psf = pickle.load(f)

    return psf, i_psf
