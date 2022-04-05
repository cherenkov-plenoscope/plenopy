import numpy as np
import ray_voxel_overlap
import json
import os
from skimage.measure import LineModelND, ransac
from .. import system_matrix

from ... import classify
from ... import trigger
from ..simulation_truth import emission_positions_of_photon_bunches
from ...plot import slices
from . import binning


def init(
    light_field_geometry,
    photon_lixel_ids,
    binning,
    sparse_system_matrix,
):
    intensities = np.zeros(light_field_geometry.number_lixel)
    for lixel_id in photon_lixel_ids:
        intensities[lixel_id] += 1

    psf = system_matrix.to_numpy_csr_matrix(
        sparse_system_matrix=sparse_system_matrix,
        number_beams=light_field_geometry.number_lixel,
        number_volume_cells=binning["number_bins"],
    )

    r = {}
    r['binning'] = binning
    r['point_spread_function'] = psf
    r['image_ray_intensities'] = intensities
    r['photon_lixel_ids'] = photon_lixel_ids

    r['reconstructed_volume_intensity'] = np.zeros(
        binning['number_bins'], dtype=np.float32)

    r['iteration'] = 0

    # Total length of ray
    r['image_ray_integral'] = psf.sum(axis=0).T

    # Total distance of all rays in this voxel
    r['voxel_integral'] = psf.sum(axis=1)

    # The sum of the length of all rays hiting this voxel weighted with the
    # overlap of the ray and this voxel
    voxel_cross_psf = psf.dot(r['image_ray_integral'])
    voxel_cross_psf = np.array(
        voxel_cross_psf
    ).reshape((voxel_cross_psf.shape[0],))

    image_ray_cross_psf = psf.T.dot(r['voxel_integral'])
    image_ray_cross_psf = np.array(
        image_ray_cross_psf
    ).reshape((image_ray_cross_psf.shape[0],))

    r['voxel_cross_psf'] = voxel_cross_psf
    r['image_ray_cross_psf'] = image_ray_cross_psf
    return r


def iterate(reconstruction):
    """
    Retunrs an reconstruction-dict with one further iteration applied.

    reconstruction : dict
        The input to be iterated once.
    """
    r = reconstruction
    reconstructed_voxel_I = r['reconstructed_volume_intensity'].copy()
    measured_image_ray_I = r['image_ray_intensities']
    voxel_cross_psf = r['voxel_cross_psf']
    image_ray_cross_psf = r['image_ray_cross_psf']
    psf = r['point_spread_function']

    measured_I_voxel = psf.dot(measured_image_ray_I)
    voxel_overlap = voxel_cross_psf > 0.0
    measured_I_voxel[voxel_overlap] /= voxel_cross_psf[voxel_overlap]

    projected_image_ray_I = psf.T.dot(reconstructed_voxel_I)
    image_ray_overlap = image_ray_cross_psf > 0.0
    projected_image_ray_I[image_ray_overlap] /= image_ray_cross_psf[
        image_ray_overlap]

    proj_I_voxel = psf.dot(projected_image_ray_I)

    voxel_diffs = measured_I_voxel - proj_I_voxel

    reconstructed_voxel_I += voxel_diffs

    reconstructed_voxel_I[reconstructed_voxel_I < 0.0] = 0.0

    diff = np.abs(
        reconstructed_voxel_I - r['reconstructed_volume_intensity']).sum()
    print('Intensity difference to previous iteration '+str(diff))

    r['reconstructed_volume_intensity'] = reconstructed_voxel_I.copy()
    r['iteration'] += 1
    return r
