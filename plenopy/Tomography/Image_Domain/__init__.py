import numpy as np
import ray_voxel_overlap
import json
import os
from .. import System_Matrix
from ... import plot
from . import Binning
from . import Point_Spread_Function
from . import Reconstruction
from . import Simulation_Truth


def save_imgae_slice_stack(
    binning,
    reconstruction,
    simulation_truth=None,
    out_dir="./tomography",
    sqrt_intensity=False,
    event_info_repr=None,
):
    r = reconstruction
    os.makedirs(out_dir, exist_ok=True)

    if simulation_truth is None:
        intensity_volume_2 = None
    else:
        intensity_volume_2 = Binning.volume_intensity_as_cube(
            volume_intensity=simulation_truth["true_volume_intensity"],
            binning=binning,
        )

    plot.slices.save_slice_stack(
        intensity_volume=Binning.volume_intensity_as_cube(
            volume_intensity=r["reconstructed_volume_intensity"],
            binning=binning,
        ),
        event_info_repr=event_info_repr,
        xy_extent=[
            binning["sen_x_bin_edges"].min(),
            binning["sen_x_bin_edges"].max(),
            binning["sen_y_bin_edges"].min(),
            binning["sen_y_bin_edges"].max(),
        ],
        z_bin_centers=binning["sen_z_bin_centers"],
        output_path=out_dir,
        image_prefix="slice_",
        intensity_volume_2=intensity_volume_2,
        xlabel="x/m",
        ylabel="y/m",
        sqrt_intensity=sqrt_intensity,
    )


def volume_intensity_sensor_frame_to_xyzi_object_frame(
    volume_intensity, binning, threshold=0
):
    cx_bin_centers = binning["cx_bin_centers"]
    cy_bin_centers = binning["cy_bin_centers"]
    obj_bin_centers = binning["obj_bin_centers"]

    xyzi = []
    for x in range(volume_intensity.shape[0]):
        for y in range(volume_intensity.shape[1]):
            for z in range(volume_intensity.shape[2]):
                if volume_intensity[x, y, z] > threshold:
                    xyzi.append(
                        np.array(
                            [
                                np.tan(cx_bin_centers[x]) * obj_bin_centers[z],
                                np.tan(cy_bin_centers[y]) * obj_bin_centers[z],
                                obj_bin_centers[z],
                                volume_intensity[x, y, z],
                            ]
                        )
                    )
    xyzi = np.array(xyzi)
    return xyzi


def xyzi_2_xyz(xyzi, maxP=25):
    maxI = np.max(xyzi[:, 3])
    xyz = []
    for p in xyzi:
        intensity = int(np.round(maxP * (p[3] / maxI)))
        for i in range(intensity):
            xyz.append(np.array([p[0], p[1], p[2]]))
    xyz = np.array(xyz)
    return xyz


def overlap_2_xyzI(overlap, x_bin_edges, y_bin_edges, z_bin_edges):
    """
    For plotting using the xyzI representation.
    Returns a 2D matrix (Nx4) of N overlaps of a ray with xoxels. Each row is
    [x,y,z positions and overlapping distance].
    """
    x_bin_centers = (x_bin_edges[0:-1] + x_bin_edges[1:]) / 2
    y_bin_centers = (y_bin_edges[0:-1] + y_bin_edges[1:]) / 2
    z_bin_centers = (z_bin_edges[0:-1] + z_bin_edges[1:]) / 2
    xyzI = np.zeros(shape=(len(overlap["overlap"]), 4))
    for i in range(len(overlap["overlap"])):
        xyzI[i] = np.array(
            [
                x_bin_centers[overlap["x"][i]],
                y_bin_centers[overlap["y"][i]],
                z_bin_centers[overlap["z"][i]],
                overlap["overlap"][i],
            ]
        )
    return xyzI
