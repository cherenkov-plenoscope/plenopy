import numpy as np

def overlap_2_xyzI(overlap, x_bin_edges, y_bin_edges, z_bin_edges):
    '''
    For plotting using the xyzI representation.
    Returns a 2D matrix (Nx4) of N overlaps of a ray with xoxels. Each row is 
    [x,y,z positions and overlapping distance].
    '''
    x_bin_centers = (x_bin_edges[0:-1] + x_bin_edges[1:])/2
    y_bin_centers = (y_bin_edges[0:-1] + y_bin_edges[1:])/2
    z_bin_centers = (z_bin_edges[0:-1] + z_bin_edges[1:])/2
    xyzI = np.zeros(shape=(len(overlap['overlap']),4))
    for i in range(len(overlap['overlap'])):
        xyzI[i] = np.array([
            x_bin_centers[overlap['x'][i]],
            y_bin_centers[overlap['y'][i]],
            z_bin_centers[overlap['z'][i]],
            overlap['overlap'][i],
        ])
    return xyzI