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
    print(event.light_field.__repr__())
    print(event.sensor_plane2imaging_system.__repr__())
    print(event.simulation_truth.__repr__())
    print(event.simulation_truth.event.__repr__())
    print(event.simulation_truth.air_shower_photon_bunches.__repr__())
    print(event.simulation_truth.detector.__repr__())