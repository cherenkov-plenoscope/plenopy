import pytest
import os
import numpy as np
import plenopy as pl
import pkg_resources
import tempfile


def test_open_event_in_run():
    run_path = pkg_resources.resource_filename(
        "plenopy", "tests/resources/run.acp"
    )
    run = pl.Run(run_path)

    with tempfile.TemporaryDirectory(prefix="plenopy_test_") as tmp_dir:
        for n, event in enumerate(run):
            with open(
                os.path.join(
                    event._path, "raw_light_field_sensor_response.phs"
                ),
                "rb",
            ) as f:
                raw = pl.raw_light_field_sensor_response.read(f)

            tmp_path = os.path.join(
                tmp_dir, "raw_light_field_sensor_response.phs"
            )

            with open(tmp_path, "wb") as f:
                pl.raw_light_field_sensor_response.write(f, raw)

            with open(tmp_path, "rb") as f:
                raw_back = pl.raw_light_field_sensor_response.read(f)

            assert pl.raw_light_field_sensor_response.is_euqal(
                a=raw, b=raw_back
            )
