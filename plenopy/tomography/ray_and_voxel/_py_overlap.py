import numpy as np
import array as ar
from .cython_ray_and_voxel import ray_box_overlap as _c_ray_box_overlap
from .cython_ray_and_voxel import overlap_of_ray_with_voxels as _c_overlap_of_ray_with_voxels


def overlap_of_ray_with_voxels(
    support, 
    direction, 
    x_bin_edges, 
    y_bin_edges, 
    z_bin_edges
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

    """
    overlaps = {
        'x': ar.array('L'),
        'y': ar.array('L'),
        'z': ar.array('L'),
        'overlap': ar.array('f'),
    }

    _overlap_of_ray_with_voxels(
        support=support, 
        direction=direction, 
        x_bin_edges=x_bin_edges, 
        y_bin_edges=y_bin_edges, 
        z_bin_edges=z_bin_edges,
        overlaps=overlaps,
        x_range=[0, len(x_bin_edges) - 1],
        y_range=[0, len(y_bin_edges) - 1],
        z_range=[0, len(z_bin_edges) - 1],
    )

    return overlaps
    """
    return _c_overlap_of_ray_with_voxels(
        support=np.ascontiguousarray(support, dtype=np.float64), 
        direction=np.ascontiguousarray(direction, dtype=np.float64), 
        x_bin_edges=np.ascontiguousarray(x_bin_edges, dtype=np.float64), 
        y_bin_edges=np.ascontiguousarray(y_bin_edges, dtype=np.float64), 
        z_bin_edges=np.ascontiguousarray(z_bin_edges, dtype=np.float64),
        x_range=np.ascontiguousarray(np.array([0, len(x_bin_edges) - 1]), dtype=np.uint32),
        y_range=np.ascontiguousarray(np.array([0, len(y_bin_edges) - 1]), dtype=np.uint32),
        z_range=np.ascontiguousarray(np.array([0, len(z_bin_edges) - 1]), dtype=np.uint32),       
    )


def _overlap_of_ray_with_voxels(
    support, 
    direction, 
    x_bin_edges, 
    y_bin_edges, 
    z_bin_edges,
    overlaps,
    x_range,
    y_range,
    z_range,
):
    x_partitions = _next_space_partitions(x_range)
    y_partitions = _next_space_partitions(y_range)
    z_partitions = _next_space_partitions(z_range)

    for xp in x_partitions:
        for yp in y_partitions:
            for zp in z_partitions:
                overlap = _c_ray_box_overlap(
                    support=support,
                    direction=direction,
                    xl=x_bin_edges[xp[0]], 
                    xu=x_bin_edges[xp[1]], 
                    yl=y_bin_edges[yp[0]], 
                    yu=y_bin_edges[yp[1]],
                    zl=z_bin_edges[zp[0]],
                    zu=z_bin_edges[zp[1]]
                )

                if (
                    xp[1]-xp[0]==1 and 
                    yp[1]-yp[0]==1 and 
                    zp[1]-zp[0]==1 and 
                    overlap > 0.0
                ):
                    overlaps['x'].append(xp[0])
                    overlaps['y'].append(yp[0])
                    overlaps['z'].append(zp[0])
                    overlaps['overlap'].append(overlap)

                elif overlap > 0.0:
                    _overlap_of_ray_with_voxels(
                        support=support, 
                        direction=direction, 
                        x_bin_edges=x_bin_edges, 
                        y_bin_edges=y_bin_edges, 
                        z_bin_edges=z_bin_edges,
                        overlaps=overlaps,
                        x_range=xp,
                        y_range=yp,
                        z_range=zp,
                    )
    return


def _next_space_partitions(dim_range):
    if dim_range[1] - dim_range[0] <= 1:
        return [[dim_range[0], dim_range[-1]],]
    else:
        cut = (dim_range[1] - dim_range[0])//2
        return [
            [dim_range[0], dim_range[0] + cut],
            [dim_range[0] + cut, dim_range[-1]]
        ]


def _ray_box_overlap(support, direction, xl, xu, yl, yu, zl, zu):
    s = support 
    d = direction
    hits_l = []
    hits_u = []

    if d[0] != 0.0:
        ixl = _intersection_plane(s, d, xl, dim=0)
        if (ixl[1]>=yl and ixl[1]<yu) and (ixl[2]>=zl and ixl[2]<zu):
            hits_l.append(ixl)

        ixu = _intersection_plane(s, d, xu, dim=0)
        if (ixu[1]>=yl and ixu[1]<=yu) and (ixu[2]>=zl and ixu[2]<=zu):
            hits_u.append(ixu)

    if d[1] != 0.0:
        iyl = _intersection_plane(s, d, yl, dim=1)
        if (iyl[0]>=xl and iyl[0]<xu) and (iyl[2]>=zl and iyl[2]<zu):
            hits_l.append(iyl)

        iyu = _intersection_plane(s, d, yu, dim=1)
        if (iyu[0]>=xl and iyu[0]<=xu) and (iyu[2]>=zl and iyu[2]<=zu):
            hits_u.append(iyu)

    if d[2] != 0.0:
        izl = _intersection_plane(s, d, zl, dim=2)
        if (izl[0]>=xl and izl[0]<xu) and (izl[1]>=yl and izl[1]<yu):
            hits_l.append(izl)

        izu = _intersection_plane(s, d, zu, dim=2)
        if (izu[0]>=xl and izu[0]<=xu) and (izu[1]>=yl and izu[1]<=yu):
            hits_u.append(izu)

    norm = np.linalg.norm

    if   len(hits_l) == 2 and len(hits_u) == 0:
        return norm(hits_l[0] - hits_l[1])

    elif len(hits_l) == 0 and len(hits_u) == 2:
        return norm(hits_u[0] - hits_u[1])

    elif len(hits_l) == 1 and len(hits_u) == 1:
        return norm(hits_u[0] - hits_l[0])

    elif len(hits_l) == 3 and len(hits_u) == 3:
        return norm(hits_u[0] - hits_l[0])

    elif len(hits_l) == 2 and len(hits_u) == 2:
        return norm(hits_u[0] - hits_l[0])

    elif len(hits_l) == 3 and len(hits_u) == 0:
        return np.sqrt((xu-xl)**2 + (yu-yl)**2 + (zu-zl)**2)

    elif len(hits_l) > 0 and len(hits_u) > 0:
        return norm(hits_u[0] - hits_l[0])

    elif len(hits_l) == 0 and len(hits_u) == 3:
        return np.sqrt((xu-xl)**2 + (yu-yl)**2 + (zu-zl)**2)    

    else:
        return 0.0


def _intersection_plane(support, direction, off, dim):
    a = (off - support[dim])/direction[dim]
    return support + direction*a