import pytest
import numpy as np
import plenopy as pl
import pkg_resources


def test_default_image_domain_binning():
    binning = pl.tomography.image_domain.Binning()
    assert binning.cx_num == 32
    assert binning.cy_num == 32
    assert binning.obj_num == 32

    assert len(binning.cx_bin_edges) == 33
    assert len(binning.cy_bin_edges) == 33
    assert len(binning.obj_bin_edges) == 33

    assert len(binning.cx_bin_centers) == 32
    assert len(binning.cy_bin_centers) == 32
    assert len(binning.obj_bin_centers) == 32


def test_image_domain_binning_repr():
    binning = pl.tomography.image_domain.Binning()
    print(binning)
