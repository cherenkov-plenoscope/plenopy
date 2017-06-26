from .cython_overlap import overlap_of_ray_with_voxels as _c_overlap_of_ray_with_voxels
from ._py_overlap import overlap_of_ray_with_voxels as _py_overlap_of_ray_with_voxels

def overlap(
    support, 
    direction, 
    x_bin_edges, 
    y_bin_edges, 
    z_bin_edges,
    x_range=None,
    y_range=None,
    z_range=None,
    implementation=_c_overlap_of_ray_with_voxels
):  
    '''
    Returns lists of the x,y, and z bin indices and the distance overlap of a
    ray and voxel.


    Parameters
    ----------

    support         3D support vector of the ray

    direction       3D direction vector of the ray

    x_bin_edges     1D array of bin edge positions in x

    y_bin_edges     1D array of bin edge positions in y

    z_bin_edges     1D array of bin edge positions in z
    '''
    if x_range is None:
        x_range = np.array([0, len(x_bin_edges) - 1])
    if y_range is None:
        y_range = np.array([0, len(y_bin_edges) - 1])
    if z_range is None:
        z_range = np.array([0, len(z_bin_edges) - 1])

    return implementation(
        support=np.ascontiguousarray(support, dtype=np.float64), 
        direction=np.ascontiguousarray(direction, dtype=np.float64), 
        x_bin_edges=np.ascontiguousarray(x_bin_edges, dtype=np.float64), 
        y_bin_edges=np.ascontiguousarray(y_bin_edges, dtype=np.float64), 
        z_bin_edges=np.ascontiguousarray(z_bin_edges, dtype=np.float64),
        x_range=np.ascontiguousarray(x_range, dtype=np.uint32),
        y_range=np.ascontiguousarray(y_range, dtype=np.uint32),
        z_range=np.ascontiguousarray(z_range, dtype=np.uint32),       
    )

