import numpy as np
cimport numpy as np
cimport cython

cdef extern double c_ray_box_overlap (        
    double *support,
    double *direction,
    double xl, double xu,
    double yl, double yu,
    double zl, double zu      
)

@cython.boundscheck(False)
@cython.wraparound(False)
def ray_box_overlap(
    np.ndarray[double, ndim=1, mode="c"] support not None,
    np.ndarray[double, ndim=1, mode="c"] direction not None,
    double xl, double xu,
    double yl, double yu,
    double zl, double zu
):
    assert support.shape[0] == 3
    assert direction.shape[0] == 3

    cdef double overlap = c_ray_box_overlap(
        &support[0],
        &direction[0],
        xl, xu,
        yl, yu,
        zl, zu
    )
    return overlap


cdef extern void c_overlap_of_ray_with_voxels(
    double *support,
    double *direction,
    double *x_bin_edges,
    double *y_bin_edges, 
    double *z_bin_edges,
    unsigned int *x_range,
    unsigned int *y_range,
    unsigned int *z_range,
    unsigned int *number_overlaps,
    unsigned int *x_idxs,
    unsigned int *y_idxs,
    unsigned int *z_idxs,
    double *overlaps
)

@cython.boundscheck(False)
@cython.wraparound(False)
def overlap_of_ray_with_voxels(
    np.ndarray[double, ndim=1, mode="c"] support not None,
    np.ndarray[double, ndim=1, mode="c"] direction not None,
    np.ndarray[double, ndim=1, mode="c"] x_bin_edges not None,
    np.ndarray[double, ndim=1, mode="c"] y_bin_edges not None,
    np.ndarray[double, ndim=1, mode="c"] z_bin_edges not None,
    np.ndarray[unsigned int, ndim=1, mode="c"] x_range not None,
    np.ndarray[unsigned int, ndim=1, mode="c"] y_range not None,
    np.ndarray[unsigned int, ndim=1, mode="c"] z_range not None
):
    assert support.shape[0] == 3
    assert direction.shape[0] == 3
    assert x_range.shape[0] == 2
    assert y_range.shape[0] == 2
    assert z_range.shape[0] == 2
    assert x_range[1] <= x_bin_edges.shape[0]
    assert y_range[1] <= y_bin_edges.shape[0]
    assert z_range[1] <= z_bin_edges.shape[0]

    x_range_width = x_range[1] - x_range[0]
    y_range_width = y_range[1] - y_range[0]
    z_range_width = z_range[1] - z_range[0]
    
    maximal_number_of_overlaps = int(
        4.0*np.sqrt(
           x_range_width**2 + 
           y_range_width**2 + 
           z_range_width**2
        )
    )

    cdef np.ndarray[unsigned int ,mode="c"] x_idxs = np.zeros(
        maximal_number_of_overlaps, dtype=np.uint32
    )
    cdef np.ndarray[unsigned int ,mode="c"] y_idxs = np.zeros(
        maximal_number_of_overlaps, dtype=np.uint32
    )
    cdef np.ndarray[unsigned int ,mode="c"] z_idxs = np.zeros(
        maximal_number_of_overlaps, dtype=np.uint32
    )
    cdef np.ndarray[double ,mode="c"] overlaps = np.zeros(
        maximal_number_of_overlaps, dtype=np.float64
    )
    cdef unsigned int number_overlaps = 0

    c_overlap_of_ray_with_voxels(
        &support[0],
        &direction[0],
        &x_bin_edges[0],
        &y_bin_edges[0],
        &z_bin_edges[0],
        &x_range[0],
        &y_range[0],
        &z_range[0],
        &number_overlaps,
        &x_idxs[0],
        &y_idxs[0],
        &z_idxs[0],
        &overlaps[0]
    )

    return {
        'x': x_idxs[0:number_overlaps],
        'y': y_idxs[0:number_overlaps],
        'z': z_idxs[0:number_overlaps],
        'overlap': overlaps[0:number_overlaps]
    }