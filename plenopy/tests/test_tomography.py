import pytest
import numpy as np
import plenopy as plp

def test_binning_flat_voxel_center_positions():
    binning = plp.Tomography.Binning(
        z_min=0.0,
        z_max=1e3,
        number_z_bins=101,
        xy_diameter=18.0,
        number_xy_bins=33)

    xyz_flat = binning.flat_xyz_voxel_positions()

    i = 0
    for x in range(binning.number_xy_bins):
        for y in range(binning.number_xy_bins):
            for z in range(binning.number_z_bins):
                assert xyz_flat[i,0] == binning.xy_bin_centers[x]
                assert xyz_flat[i,1] == binning.xy_bin_centers[y]
                assert xyz_flat[i,2] == binning.z_bin_centers[z]
                i+=1