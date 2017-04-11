import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

def flatten(hist, binning, threshold=0):
    """
    Returns a flat array of x,y,z positions and intensities for each voxel in
    the 3D histogram.

    Parameters
    ----------
    hist            A 3D intensity histogram with the shape specified in the 
                    binning parameter.

    binning         The 3D volume binning used to create the 3D intensity 
                    histogram.


    threshold       Voxels in the 3D intensity histogram below this threshold 
                    are neglected.
    """
    xyzi = []
    for x in range(hist.shape[0]):
        for y in range(hist.shape[1]):
            for z in range(hist.shape[2]):
                if hist[x,y,z] > threshold:
                    xyzi.append(np.array([
                        binning.xy_bin_centers[x],
                        binning.xy_bin_centers[y],
                        binning.z_bin_centers[z],
                        hist[x,y,z]]))
    xyzi = np.array(xyzi)
    return xyzi