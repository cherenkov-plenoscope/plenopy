import numpy as np
import os
import json

def write_dict_to_file(dictionary, path):
    with open(path, 'w') as outfile:
        json.dump(config, outfile)    

def collect_trigger_relevant_information(event):
    info = {}

    pixel_i = np.sum(event.light_field.intensity, axis=1)
    info['pixel'] = {
        'min': np.min(pixel_i), 
        'med': np.median(pixel_i), 
        'max': np.max(pixel_i)}

    paxel_i = np.sum(event.light_field.intensity, axis=0)
    info['paxel'] = {
        'min': np.min(paxel_i), 
        'med': np.median(paxel_i), 
        'max': np.max(paxel_i)}

    lixel_i = event.light_field.intensity.flatten()
    info['lixel'] = {
        'min': np.min(lixel_i), 
        'med': np.median(lixel_i), 
        'max': np.max(lixel_i),
        'sum': np.sum(lixel_i)}

    # arrival time distribution
    lixel_t = event.light_field.arrival_time.flatten()
    start_time = 0.0
    end_time = 50e-9
    steps = 50
    bins = np.linspace(start_time, end_time, steps)
    info['arrival_time_histogram'] = {
        'start_time': start_time, 
        'end_time': end_time,
        'histogram': np.histogram(lixel_t, weights=lixel_i, bins=bins)[0]}

    # light field refocus trigger
    f = event.light_field.expected_focal_length_of_imaging_system
    start_obj_dist = 10*f
    end_obj_dist = 200*f
    number_refocuses = int(np.ceil(0.25*event.light_field.number_lixel**(1/3)))

    object_distances = np.logspace(
        np.log10(start_obj_dist),
        np.log10(end_obj_dist),
        number_refocuses)

    refocus_nodes = []
    for object_distance in object_distances:
        image = event.light_field.refocus(object_distance)

        refocus_nodes.append({
            'object_distance': object_distance,
            'max': np.max(image.intensity),
            'med': np.median(image.intensity),
            'min': np.min(image.intensity)})

    info['refocus_pixel'] = refocus_nodes

    return info

def export_trigger_information(event):
    info = {}
    evth = event.simulation_truth.event.corsika_event_header
    runh = event.simulation_truth.event.corsika_run_header

    info['id'] = {'run': runh.number, 'event': evth.number}
    info['truth'] = {
        'primary_particle': {'name': evth.primary_particle, 'id': evth.raw[3-1]},
        'energy': evth.raw[4-1],
        'zenith': evth.raw[11-1],
        'azimuth': evth.raw[12-1],
        'core_position': {
            'x': evth.raw[98-1]/100, 
            'y': evth.raw[118-1]/100},
        'scatter_radius': runh.raw[248-1]/100,
        'first_interaction_height': np.abs(evth.raw[7-1]/100)}
    info['acp'] = {
        'response': collect_trigger_relevant_information(event),
        'observation_level_altitude_asl': runh.raw[6-1]/100,
        'light_field_sensor': {
            'expected_focal_length': event.light_field.expected_focal_length_of_imaging_system,
            'expected_aperture_radius': event.light_field.expected_aperture_radius_of_imaging_system,
            'number_pixel': event.light_field.number_pixel,
            'number_paxel': event.light_field.number_paxel,
        }
    }

    return info
