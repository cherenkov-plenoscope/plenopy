import pytest
import numpy as np
import plenopy as pl
import pkg_resources


def test_default_image_domain_binning():
    b = pl.tomography.image_domain.DepthOfFieldBinning()
    assert b.x_img_num == 32
    assert b.y_img_num == 32
    assert b.b_img_num == 32

    assert len(b.x_img_bin_edges) == 33
    assert len(b.y_img_bin_edges) == 33
    assert len(b.b_img_bin_edges) == 33

    assert len(b.x_img_bin_centers) == 32
    assert len(b.y_img_bin_centers) == 32
    assert len(b.b_img_bin_centers) == 32


def test_image_domain_binning_repr():
    b = pl.tomography.image_domain.DepthOfFieldBinning()
    print(b)
