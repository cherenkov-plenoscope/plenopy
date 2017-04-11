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


def plot(xyzIs, xyzIs2=None, alpha=0.01):
    """
    Plot a 3D intensity distribution. Can plot two distributions at the same 
    time using blue and red color.

    Parameters
    ----------

    xyzIs           An array of x,y,z positions and Intensities.

    xyzIs2          A second (optional) array of x,y,z positions and Intensities
                    to be plotted in red color.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    inte = xyzIs[:,3]
    inte = 500*inte/inte.max()

    ax.scatter(
        xyzIs[:,0], xyzIs[:,1], xyzIs[:,2],
        s=inte,
        depthshade=False,
        alpha=alpha,
        lw=0)

    if xyzIs2 is not None:
        inte2 = xyzIs2[:,3]
        inte2 = 500*inte2/inte2.max()

        ax.scatter(
            xyzIs2[:,0], xyzIs2[:,1], xyzIs2[:,2],
            s=inte2,
            c='r',
            depthshade=False,
            alpha=alpha,
            lw=0)