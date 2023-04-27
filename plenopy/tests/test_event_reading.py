import pytest
import numpy as np
import plenopy as pl
import pkg_resources
import os


def test_read_photon_stream():
    run_path = pkg_resources.resource_filename(
        "plenopy", "tests/resources/run.acp"
    )
    run = pl.Run(run_path)

    raw_path = os.path.join(
        run.path, "1", "raw_light_field_sensor_response.phs"
    )
    with open(raw_path, "rb") as f:
        raw = pl.raw_light_field_sensor_response.read(f=f)

    assert raw["number_lixel"] == 1039 * 19
    assert np.abs(raw["time_slice_duration"] - 0.5e-9) < 1e-12
    assert raw["number_time_slices"] == 100

    sequence_from_pure_python = np.zeros(
        shape=(raw["number_time_slices"], raw["number_lixel"]), dtype=np.uint16
    )
    sequence_from_cython = np.zeros(
        shape=(raw["number_time_slices"], raw["number_lixel"]), dtype=np.uint16
    )

    time_delay_mean = run.light_field_geometry.time_delay_mean.copy()

    pl.photon_stream.py_stream2sequence(
        photon_stream=raw["photon_stream"],
        time_slice_duration=raw["time_slice_duration"],
        NEXT_READOUT_CHANNEL_MARKER=pl.raw_light_field_sensor_response.NEXT_READOUT_CHANNEL_MARKER,
        sequence=sequence_from_pure_python,
        time_delay_mean=time_delay_mean,
    )

    pl.photon_stream.cython_reader.stream2sequence(
        photon_stream=raw["photon_stream"],
        time_slice_duration=raw["time_slice_duration"],
        NEXT_READOUT_CHANNEL_MARKER=pl.raw_light_field_sensor_response.NEXT_READOUT_CHANNEL_MARKER,
        sequence=sequence_from_cython,
        time_delay_mean=time_delay_mean,
    )

    np.testing.assert_equal(sequence_from_pure_python, sequence_from_cython)


def test_open_event():
    run_path = pkg_resources.resource_filename(
        "plenopy", "tests/resources/run.acp"
    )
    run = pl.Run(run_path)
