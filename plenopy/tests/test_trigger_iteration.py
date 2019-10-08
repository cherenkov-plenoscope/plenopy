import pytest
import numpy as np
import plenopy as pl
import pkg_resources


def test_trigger_iteration():
    run_path = pkg_resources.resource_filename(
        'plenopy',
        'tests/resources/run.acp')
    run = pl.Run(run_path)

    trigger_preparation = pl.trigger.prepare_refocus_sum_trigger(
        light_field_geometry=run.light_field_geometry,
        object_distances=[7.5e3, 15e3, 22.5e3])

    trigger_response_binary = []
    for n, event in enumerate(run):
        trigger_response = pl.trigger.apply_refocus_sum_trigger(
            event=event,
            trigger_preparation=trigger_preparation,
            min_number_neighbors=3,
            integration_time_in_slices=5,)
        trigger_response_binary.append(trigger_response)
