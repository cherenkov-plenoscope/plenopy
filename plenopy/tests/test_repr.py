import pytest
import plenopy as pl
import pkg_resources


def test_light_field_geometry():
    path = pkg_resources.resource_filename(
        'plenopy', 
        'tests/resources/run.acp/input/plenoscope')
    print(pl.LightFieldGeometry(path).__repr__())


def test_run():
    path = pkg_resources.resource_filename(
        'plenopy', 
        'tests/resources/run.acp')
    print(pl.Run(path).__repr__())


def test_event():
    path = pkg_resources.resource_filename(
        'plenopy', 
        'tests/resources/run.acp')
    event = pl.Run(path)[0]
    print(event.__repr__())