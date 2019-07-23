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

    trigger_response_linear = []
    trigger_response_binary = []
    for n, event in enumerate(run):
        trigger_response = pl.trigger.__apply_refocus_sum_trigger(
            event=event,
            trigger_preparation=trigger_preparation,
            min_number_neighbors=3,
            integration_time_in_slices=5,
            max_iterations=10000,)
        trigger_response_linear.append(trigger_response)

        trigger_response = pl.trigger.apply_refocus_sum_trigger(
            event=event,
            trigger_preparation=trigger_preparation,
            min_number_neighbors=3,
            integration_time_in_slices=5,)
        trigger_response_binary.append(trigger_response)

    for t in range(len(trigger_response_linear)):
        for obj in range(3):
            trb = trigger_response_binary[t][obj]
            trl = trigger_response_linear[t][obj]
            assert trl['iteration_converged'] == True
            assert trl['patch_threshold'] == trb['patch_threshold']