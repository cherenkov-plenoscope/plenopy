import numpy as np
import os
import json

def write_dict_to_file(dictionary, path):
    with open(path, 'w') as outfile:
        json.dump(config, outfile)    

def collect_trigger_relevant_information(event):
    info = {
        'nice_indicator_1': 1337,
        'nice_indicator_1': 42}
    return info

def export_trigger_information(event):
    info = {}
    evth = event.simulation_truth.event.corsika_event_header
    runh = event.simulation_truth.event.corsika_run_header

    info['id'] = {'run': runh.number, 'event': evth.number}
    info['truth'] = {
        'primary_particle': {'name': evth.primary_particle, 'id': evth.raw[3-1]}
        'energy': evth.raw[4-1],
        'zenith': evth.raw[11-1],
        'azimuth': evth.raw[12-1],
        'core_position': {
            'x': evth.raw[98-1]/100, 
            'y': evth.raw[118-1]/100},
        'scatter_radius': runh.raw[248-1]/100,
        'first_interaction_height': evth.raw[7-1]/100}
    info['acp'] = {
        'response': collect_trigger_relevant_information(event),
        'observation_level_altitude_asl': runh[6-1]/100,
        'light_field_sensor': {
            'expected_focal_length': event.light_field.expected_focal_length_of_imaging_system,
            'expected_aperture_radius': event.light_field.expected_aperture_radius_of_imaging_system,
            'number_pixel': event.light_field.number_pixel,
            'number_paxel': event.light_field.number_paxel,
        }
    }

    return info
