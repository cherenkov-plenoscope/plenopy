import pytest
import numpy as np
import plenopy as pl
import pkg_resources
import tempfile
import os


def test_init():
    lixel_statistics_path = pkg_resources.resource_filename(
        'plenopy', 
        'tests/resources/run.acp/input/plenoscope')
    ls = pl.LixelStatistics(lixel_statistics_path)    

    # A 'small' MAGIC 17m class ACP
    number_pixel = 1039
    number_paxel = 19
    number_lixel = number_pixel*number_paxel
    
    assert ls.number_lixel == number_lixel
    assert ls.number_pixel == number_pixel
    assert ls.number_paxel == number_paxel

    assert ls.expected_focal_length_of_imaging_system == 17.0
    assert ls.expected_aperture_radius_of_imaging_system == 8.5

    assert ls.x_mean.shape == (number_lixel,)
    assert ls.x_std.shape == (number_lixel,)

    assert ls.y_mean.shape == (number_lixel,)
    assert ls.y_std.shape == (number_lixel,)

    assert ls.cx_mean.shape == (number_lixel,)
    assert ls.cx_std.shape == (number_lixel,)

    assert ls.cy_mean.shape == (number_lixel,)
    assert ls.cy_std.shape == (number_lixel,)

    assert ls.time_delay_mean.shape == (number_lixel,)
    assert ls.time_delay_std.shape == (number_lixel,)

    assert ls.efficiency.shape == (number_lixel,)
    assert ls.efficiency_std.shape == (number_lixel,)


def test_plot():
    lixel_statistics_path = pkg_resources.resource_filename(
        'plenopy', 
        'tests/resources/run.acp/input/plenoscope')
    lixel_statistics = pl.LixelStatistics(lixel_statistics_path)

    with tempfile.TemporaryDirectory(prefix='plenopy') as tmp:
        ls_plot = pl.plot.lixel_statistics.PlotLixelStatistics.PlotLixelStatistics(lixel_statistics, tmp)
        ls_plot.save()

        assert os.path.exists(os.path.join(tmp, 'cx_mean.png'))
        assert os.path.exists(os.path.join(tmp, 'cy_mean.png'))
        assert os.path.exists(os.path.join(tmp, 'cy_stddev.png'))
        assert os.path.exists(os.path.join(tmp, 'cy_stddev.png'))

        assert os.path.exists(os.path.join(tmp, 'x_mean.png'))
        assert os.path.exists(os.path.join(tmp, 'y_mean.png'))

        assert os.path.exists(os.path.join(tmp, 'time_delay_mean.png'))
        assert os.path.exists(os.path.join(tmp, 'time_stddev.png'))
        assert os.path.exists(os.path.join(tmp, 'efficiency.png'))
        assert os.path.exists(os.path.join(tmp, 'efficiency_error.png'))
        assert os.path.exists(os.path.join(tmp, 'c_mean_vs_c_std.png'))
        assert os.path.exists(os.path.join(tmp, 'x_y_mean_hist2d.png'))
        assert os.path.exists(os.path.join(tmp, 'cx_cy_mean_hist2d.png'))
