import numpy as np
import os
import json
from . import ImageRays
import matplotlib.pyplot as plt

def write_dict_to_file(dictionary, path):
    with open(path, 'w') as outfile:
        json.dump(dictionary, outfile)

def to_std_float_and_integer(dic):
    ret = {}
    for k, v in list(dic.items()):
        if isinstance(v, dict):
            ret[k] = to_std_float_and_integer(v)
        elif isinstance(v, np.ndarray):
            if v.dtype == np.float32:
                v = v.astype(np.float64)
            ret[k] = v.tolist()
        elif isinstance(v, np.floating):
            ret[k] = float(v)
        elif isinstance(v, np.integer):
            ret[k] = int(v)
        else:
            ret[k] = v
    return ret

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

    valid = event.light_field.valid_lixel.flatten()
    intensity = event.light_field.intensity.flatten()
    
    fov_radius_pixel = event.plenoscope_geometry.pixel_FoV_hex_flat2flat/2.0
    fov_radius = event.plenoscope_geometry.max_FoV_diameter

    number_pixel_on_diagonal = int(np.ceil(fov_radius/fov_radius_pixel))
    bins = np.linspace(-fov_radius, fov_radius, number_pixel_on_diagonal)

    in_round_fov = np.zeros(
        shape=(number_pixel_on_diagonal-1,number_pixel_on_diagonal-1), 
        dtype=np.bool)
    for x in range(number_pixel_on_diagonal-1):
        for y in range(number_pixel_on_diagonal-1):
            if np.hypot(bins[x], bins[y]) < fov_radius:
                in_round_fov[x,y] = True

    image_rays = ImageRays(event.light_field)
    refocus_nodes = []
    for object_distance in object_distances:
        #image = event.light_field.refocus(object_distance)

        cx, cy = image_rays.cx_cy_in_object_distance(object_distance)

        image = np.histogram2d(
            cx[valid], 
            cy[valid], 
            weights=intensity[valid],
            bins=(bins,bins))[0]

        """plt.imshow(
            image, 
            cmap='viridis', 
            interpolation='none',
            extent=[-fov_radius,fov_radius,-fov_radius,fov_radius])
        plt.show()"""

        refocus_nodes.append({
            'object_distance': object_distance,
            'max': np.max(image[in_round_fov]),
            'med': np.median(image[in_round_fov]),
            'min': np.min(image[in_round_fov])})

    info['refocussing'] = {
        'square_pixel_fov': 2*fov_radius_pixel,
        'fov': 2*fov_radius,
        'image_intensity': refocus_nodes}

    return info

def export_trigger_information(event):
    info = {}
    
    assert event.type == "SIMULATION"
    if event.trigger_type == "EXTERNAL_TRIGGER_BASED_ON_AIR_SHOWER_SIMULATION_TRUTH":
        info['trigger_type'] = "EXTERNAL_TRIGGER_BASED_ON_AIR_SHOWER_SIMULATION_TRUTH"
        evth = event.simulation_truth.event.corsika_event_header
        runh = event.simulation_truth.event.corsika_run_header
        info['id'] = {'run': runh.number, 'event': evth.number}
        info['simulation_truth'] = {
            'primary_particle': {'name': evth.primary_particle, 'id': evth.raw[3-1]},
            'energy': evth.raw[4-1],
            'zenith': evth.raw[11-1],
            'azimuth': evth.raw[12-1],
            'core_position': {
                'x': evth.raw[98-1]/100, 
                'y': evth.raw[118-1]/100},
            'scatter_radius': runh.raw[248-1]/100,
            'first_interaction_height': np.abs(evth.raw[7-1]/100),
            'observation_level_altitude_asl': runh.raw[6-1]/100}    

    elif  event.trigger_type == "EXTERNAL_RANDOM_TRIGGER":
        info['trigger_type'] = "EXTERNAL_RANDOM_TRIGGER"

    info['acp'] = {
        'response': collect_trigger_relevant_information(event),
        'light_field_sensor': {
            'expected_focal_length': event.light_field.expected_focal_length_of_imaging_system,
            'expected_aperture_radius': event.light_field.expected_aperture_radius_of_imaging_system,
            'number_pixel': event.light_field.number_pixel,
            'number_paxel': event.light_field.number_paxel,
        }
    }

    return to_std_float_and_integer(info)