import pytest
import numpy as np
import plenopy as pl
import pkg_resources


def test_open_event_in_run():
    run_path = pkg_resources.resource_filename(
        'plenopy', 
        'tests/resources/run.acp')
    run = pl.Run(run_path)

    for n, event in enumerate(run):
        
        trigger_summary = pl.trigger_study.export_trigger_information(event)

        assert 'id' in trigger_summary
        assert 'run' in trigger_summary['id']
        assert 'event' in trigger_summary['id']

        assert 'trigger_type' in trigger_summary
        assert 'simulation_truth' in trigger_summary

        assert 'primary_particle' in trigger_summary['simulation_truth']
        assert 'energy' in trigger_summary['simulation_truth']
        assert 'zenith' in trigger_summary['simulation_truth']
        assert 'azimuth' in trigger_summary['simulation_truth']
        assert 'core_position' in trigger_summary['simulation_truth']
        assert 'scatter_radius' in trigger_summary['simulation_truth']
        assert 'first_interaction_height' in trigger_summary['simulation_truth']
        assert 'observation_level_altitude_asl' in trigger_summary['simulation_truth']
        
        assert 'acp' in trigger_summary
        assert 'response' in trigger_summary['acp']