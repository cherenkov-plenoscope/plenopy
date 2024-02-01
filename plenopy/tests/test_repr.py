import plenopy as pl
import os


def test_light_field_geometry():
    path = os.path.join(
        pl.testing.pkg_dir(),
        "tests",
        "resources",
        "run.acp",
        "input",
        "plenoscope",
    )
    print(pl.LightFieldGeometry(path).__repr__())


def test_run():
    path = os.path.join(pl.testing.pkg_dir(), "tests", "resources", "run.acp")
    print(pl.Run(path).__repr__())


def test_event():
    path = os.path.join(pl.testing.pkg_dir(), "tests", "resources", "run.acp")
    event = pl.Run(path)[0]
    print(event.__repr__())
    print(event.light_field_geometry.__repr__())
    print(event.light_field_geometry.sensor_plane2imaging_system.__repr__())
    print(event.simulation_truth.__repr__())
    print(event.simulation_truth.event.__repr__())
    print(event.simulation_truth.air_shower_photon_bunches.__repr__())
    print(event.simulation_truth.detector.__repr__())
