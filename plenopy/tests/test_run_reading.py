import numpy as np
import plenopy as pl
import os


def test_open_run():
    path = os.path.join(pl.testing.pkg_dir(), "tests", "resources", "run.acp")
    run = pl.Run(path)

    assert run.number_events == 32
    np.testing.assert_equal(run.event_numbers, np.arange(32) + 1)
    assert run.path == path

    # A 'small' MAGIC 17m class ACP
    assert run.light_field_geometry.number_lixel == 1039 * 19
    assert run.light_field_geometry.number_pixel == 1039
    assert run.light_field_geometry.number_paxel == 19

    assert len(run) == 32


def test_open_event_in_run():
    path = os.path.join(pl.testing.pkg_dir(), "tests", "resources", "run.acp")
    run = pl.Run(path)

    for n, event in enumerate(run):
        assert event.number == n + 1
