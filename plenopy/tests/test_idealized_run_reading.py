import pytest
import numpy as np
import plenopy as pl
import pkg_resources


def test_open_idealied_run():
    run_path = pkg_resources.resource_filename(
        'plenopy', 
        'tests/resources/idealized_example_run.corsika')
    run = pl.idealized_plenoscope.Run(run_path)
    assert run.number_events == 5
    assert run_path in run.path
    assert np.array_equal(run.event_numbers, np.array([1,2,3,4,5]))


def test_open_idealized_event():
    run_path = pkg_resources.resource_filename(
        'plenopy', 
        'tests/resources/idealized_example_run.corsika')
    run = pl.idealized_plenoscope.Run(run_path)
    evt = run[0]
    assert evt.number == 1