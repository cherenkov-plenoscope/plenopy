import numpy as np
from joblib import Memory
import os

cachedir = '/tmp/plenopy'
os.makedirs(cachedir, exist_ok=True)
memory = Memory(cachedir=cachedir, verbose=0)

from .. import ray_and_voxel


def update(
    vol_I, 
    system_matrix, 
    measured_lixel_I,  
    obj_dist_regularization, 
    valid_voxel,
    ray_length
):  
    measured_photon_density_along_rays = measured_lixel_I/ray_length
    measured_ph_I_voxel = (system_matrix.dot(measured_photon_density_along_rays))
    measured_ph_I_voxel /= obj_dist_regularization

    proj_ph_dist_int_lixel = system_matrix.T.dot(vol_I)
    proj_ph_lixel = proj_ph_dist_int_lixel/ray_length
    proj_ph_dst_lixel = proj_ph_lixel/ray_length
    
    proj_ph_I_voxel = system_matrix.dot(proj_ph_dst_lixel)

    voxel_diffs = measured_ph_I_voxel - proj_ph_I_voxel

    vol_I[valid_voxel] += voxel_diffs[valid_voxel]

    vol_I[vol_I < 0.0] = 0.0
    return vol_I


@memory.cache
def make_tomographic_system_matrix(
    supports, 
    directions, 
    x_bin_edges, 
    y_bin_edges,
    z_bin_edges
):
    return ray_and_voxel.system_matrix(
        supports=supports,
        directions=directions, 
        x_bin_edges=x_bin_edges, 
        y_bin_edges=y_bin_edges,
        z_bin_edges=z_bin_edges,
    )