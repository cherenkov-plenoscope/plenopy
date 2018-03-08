from scipy.sparse import coo_matrix
from . import overlap
import numpy as np
import array


def system_matrix(
    supports,
    directions,
    x_bin_edges,
    y_bin_edges,
    z_bin_edges,
):
    '''
    Returns a tomographic System Matrix.
    Along the rows are the voxels and along the columns are the rays.
    Each matrix element represents the overlap in euclidean distance of the
    corresponding ray with the voxel.

    As the system matrix is very sparse, we construct and return a
    scipy.sparse matrix.

    Parameters
    ----------
    supports        [N x 3] 2D array of N 3D support vectors of the rays.

    directions      [N x 3] 2D array of N 3D direction vectors of the rays.

    x_bin_edges     1D array of bin edge positions along x axis.

    y_bin_edges     1D array of bin edge positions along y axis.

    z_bin_edges     1D array of bin edge positions along z axis.
    '''
    assert supports.shape[0] == directions.shape[0], (
        'number of support vectors ({0:d}) must match ' +
        'the number of direction vectors ({1:d})'.format(
            supports.shape[0], directions.shape[0]
        )
    )
    x_num = x_bin_edges.shape[0] - 1
    y_num = y_bin_edges.shape[0] - 1
    z_num = z_bin_edges.shape[0] - 1
    rays_num = supports.shape[0]

    ray_voxel_overlap = array.array('f')
    ray_indicies = array.array('L')
    voxel_indicies = array.array('L')

    for ray_idx in range(rays_num):
        ov = overlap(
            support=supports[ray_idx],
            direction=directions[ray_idx],
            x_bin_edges=x_bin_edges,
            y_bin_edges=y_bin_edges,
            z_bin_edges=z_bin_edges
        )

        voxel_idxs = np.ravel_multi_index(
            np.array([ov['x'], ov['y'], ov['z']]),
            dims=(x_num, y_num, z_num),
            order='C'
        )

        ray_idxs = ray_idx*np.ones(
            voxel_idxs.shape[0],
            dtype=np.uint32
        )

        ray_voxel_overlap.extend(ov['overlap'])
        ray_indicies.extend(ray_idxs)
        voxel_indicies.extend(voxel_idxs)

    sys_matrix = coo_matrix(
        (ray_voxel_overlap, (voxel_indicies, ray_indicies)),
        shape=(x_num*y_num*z_num, rays_num),
        dtype=np.float32
    )

    return sys_matrix.tocsr()
