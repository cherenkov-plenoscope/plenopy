import pytest
import numpy as np
import plenopy as pl
import pkg_resources
import tempfile
import os
import shutil
import glob

run_path = pkg_resources.resource_filename(
    "plenopy", "tests/resources/run.acp"
)


def test_transparent_gzip():
    run = pl.Run(run_path)
    event = run[0]

    with tempfile.TemporaryDirectory(prefix="plenopy_") as tmp:
        tmp_run_path = os.path.join(tmp, "run.acp")
        shutil.copytree(run_path, tmp_run_path)

        event_numbers = (
            pl.tools.acp_format.all_folders_with_digit_names_in_path(
                tmp_run_path
            )
        )
        for event_number in event_numbers:
            pl.tools.acp_format.compress_event_in_place(
                os.path.join(tmp_run_path, "{:d}".format(event_number))
            )

        assert pl.tools.acp_format.is_gzipped(
            os.path.join(
                tmp_run_path, "1", "raw_light_field_sensor_response.phs.gz"
            )
        )

        assert pl.tools.acp_format.is_gzipped(
            os.path.join(
                tmp_run_path,
                "1",
                "simulation_truth",
                "detector_pulse_origins.bin.gz",
            )
        )

        run_gz = pl.Run(tmp_run_path)
        event_gz = run_gz[0]
        assert event_gz.number == 1
        assert event_gz.number == event.number

        np.testing.assert_array_equal(
            event_gz.simulation_truth.detector.pulse_origins,
            event.simulation_truth.detector.pulse_origins,
        )

        np.testing.assert_array_equal(
            event_gz.raw_sensor_response["photon_stream"],
            event.raw_sensor_response["photon_stream"],
        )
