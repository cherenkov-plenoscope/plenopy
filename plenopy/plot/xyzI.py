import numpy as np
import matplotlib.pyplot as plt


def plot_xyzI(xyzIs, xyzIs2=None, alpha_max=0.2, steps=32, ball_size=100.0):
    """
    Plot a 3D intensity distribution. Can plot two distributions at the same
    time using blue and red color.

    Parameters
    ----------

    xyzIs           An array of x,y,z positions and Intensities.

    xyzIs2          A second (optional) array of x,y,z positions and
                    intensities to be plotted in red color.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    add2ax_xyzI(
        ax,
        xyzIs,
        color="b",
        steps=steps,
        alpha_max=alpha_max,
        ball_size=ball_size,
    )

    if xyzIs2 is not None:
        add2ax_xyzI(
            ax,
            xyzIs2,
            color="r",
            steps=steps,
            alpha_max=alpha_max,
            ball_size=ball_size,
        )


def hist3D_to_xyzI(
    xyz_hist,
    x_bin_centers,
    y_bin_centers,
    z_bin_centers,
    threshold=0,
):
    """
    Returns a flat array of x,y,z positions and intensities for each voxel in
    the 3D histogram.

    Parameters
    ----------
    xyz_hist : array 3d
        A 3D intensity histogram with the shape specified in the
        binning parameter.

    x_bin_centers : array 1d
        The centers of the bins in x

    y_bin_centers : array 1d

    z_bin_centers : array 1d

    threshold : float
        Voxels in the 3D intensity histogram below this threshold
        are neglected.
    """
    xyzi = []
    for x in range(xyz_hist.shape[0]):
        for y in range(xyz_hist.shape[1]):
            for z in range(xyz_hist.shape[2]):
                if xyz_hist[x, y, z] > threshold:
                    xyzi.append(
                        np.array(
                            [
                                x_bin_centers[x],
                                y_bin_centers[y],
                                z_bin_centers[z],
                                xyz_hist[x, y, z],
                            ]
                        )
                    )
    xyzi = np.array(xyzi)
    return xyzi


def add2ax_xyzI(
    ax, xyzIs, color="b", alpha_max=0.2, steps=32, ball_size=100.0
):
    if xyzIs.shape[0] == 0:
        return

    intensities = xyzIs[:, 3]
    xyzIs_sorted = xyzIs[np.argsort(intensities)]
    length = xyzIs_sorted.shape[0]

    if steps > length:
        steps = length

    (starts, ends) = _start_and_end_slices_for_1D_array_chunking(
        number_of_chunks=steps, array_length=length
    )

    mean_chunk_intensities = []
    for i in range(steps):
        start = starts[i]
        end = ends[i]
        mean_chunk_intensities.append(xyzIs_sorted[start:end, 3].mean())
    mean_chunk_intensities = np.array(mean_chunk_intensities)

    max_chunk_intensities = mean_chunk_intensities.max()

    for i in range(steps):
        start = starts[i]
        end = ends[i]
        mean_I = mean_chunk_intensities[i]
        max_I = max_chunk_intensities
        relative_I = mean_I / max_I

        ax.scatter(
            xyzIs_sorted[start:end, 0],
            xyzIs_sorted[start:end, 1],
            xyzIs_sorted[start:end, 2],
            s=ball_size,
            depthshade=False,
            alpha=relative_I * alpha_max,
            c=color,
            lw=0,
        )


def _start_and_end_slices_for_1D_array_chunking(
    number_of_chunks, array_length
):
    assert array_length >= number_of_chunks
    assert array_length > 0
    assert number_of_chunks >= 1

    chunk_edges = np.array(
        np.floor(np.linspace(0.0, array_length - 1, number_of_chunks + 1)),
        dtype=np.int64,
    )

    return (chunk_edges[0:-1], chunk_edges[1:])
