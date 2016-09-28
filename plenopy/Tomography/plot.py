import numpy as np
import matplotlib.pyplot as plt


def flatten(hist, binning, threshold=0):
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


def plot(xyzIs, xyzIs2=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    inte = xyzIs[:,3]
    inte = 500*inte/inte.max()

    ax.scatter(
        xyzIs[:,0], xyzIs[:,1], xyzIs[:,2],
        s=inte,
        depthshade=False,
        alpha=0.01,
        lw=0)

    if xyzIs2 is not None:
        inte2 = xyzIs2[:,3]
        inte2 = 500*inte2/inte2.max()

        ax.scatter(
            xyzIs2[:,0], xyzIs2[:,1], xyzIs2[:,2],
            s=inte,
            c='r',
            depthshade=False,
            alpha=0.01,
            lw=0)        

    plt.show()