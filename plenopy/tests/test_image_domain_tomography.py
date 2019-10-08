import pytest
import numpy as np
import plenopy as pl
import pkg_resources


def test_default_image_domain_binning():
    b = pl.tomography.image_domain.init_binning_for_depth_of_field(
        focal_length=1.0,
        cx_min=np.deg2rad(-3.5),
        cx_max=np.deg2rad(+3.5),
        number_cx_bins=96,
        cy_min=np.deg2rad(-3.5),
        cy_max=np.deg2rad(3.5),
        number_cy_bins=96,
        obj_min=5e3,
        obj_max=25e3,
        number_obj_bins=32
    )
    assert b['number_cx_bins'] == 96
    assert b['number_cy_bins'] == 96
    assert len(b['cx_bin_centers']) == 96
    assert len(b['cy_bin_centers']) == 96

    assert b['number_sen_x_bins'] == 96
    assert b['number_sen_y_bins'] == 96
    assert len(b['sen_x_bin_centers']) == 96
    assert len(b['sen_y_bin_centers']) == 96

    assert len(b['cx_bin_edges']) == 96 + 1
    assert len(b['cy_bin_edges']) == 96 + 1

    assert len(b['sen_x_bin_edges']) == 96 + 1
    assert len(b['sen_y_bin_edges']) == 96 + 1

    assert b['number_sen_z_bins'] == 32
    assert b['number_obj_bins'] == 32

    assert len(b['sen_z_bin_centers']) == 32
    assert len(b['obj_bin_centers']) == 32

    assert len(b['sen_z_bin_edges']) == 32 + 1
    assert len(b['obj_bin_edges']) == 32 + 1

    assert b['number_bins'] == 96 * 96 * 32