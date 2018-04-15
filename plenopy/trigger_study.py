import numpy as np
import os
import gzip
import json
from .photon_stream import cython_reader as phscr
from . import sequence as lfs


def write_dict_to_file(dictionary, path):
    """
    Writes a dictionary into a json file. If the path extension is 'gz', then
    a gzipped json file is written.

    Parameters
    ----------
    dictionary      A dictionary to be written.

    path            The output path of the JSON file (or gzipped) JSON file
    """
    if os.path.splitext(path)[1] == '.gz':
        with gzip.open(path, mode="wt") as outfile:
            json.dump(dictionary, outfile)
    else:
        with open(path, 'w') as outfile:
            json.dump(dictionary, outfile)


def un_numpyify(s):
    if isinstance(s, list):
        return un_numpyify_list(s)
    elif isinstance(s, dict):
        return un_numpyify_dictionary(s)
    else:
        return un_numpyify_item(s)


def un_numpyify_dictionary(dic):
    ret = {}
    for k, v in list(dic.items()):
        ret[k] = un_numpyify_item(v)
    return ret


def un_numpyify_list(lis):
    ret = []
    for item in lis:
        ret.append(un_numpyify_item(item))
    return ret


def un_numpyify_item(v):
    ret = None
    if isinstance(v, dict):
        ret = un_numpyify_dictionary(v)
    elif isinstance(v, list):
        ret = un_numpyify_list(v)
    elif isinstance(v, np.ndarray):
        ret = v.tolist()
    elif isinstance(v, np.floating):
        ret = float(v)
    elif isinstance(v, np.integer):
        ret = int(v)
    else:
        ret = v
    return ret


def collect_trigger_relevant_information(event):
    info = {}

    lixel_sequence = event.light_field_sequence_raw()

    pixel_sequence = lfs.pixel_sequence(
        lixel_sequence=lixel_sequence,
        number_pixel=event.light_field_geometry.number_pixel,
        number_paxel=event.light_field_geometry.number_paxel)

    paxel_sequence = lfs.paxel_sequence(
        lixel_sequence=lixel_sequence,
        number_pixel=event.light_field_geometry.number_pixel,
        number_paxel=event.light_field_geometry.number_paxel)

    pixel_i = np.sum(pixel_sequence, axis=0)
    info['raw_pixel'] = {
        'min': np.min(pixel_i),
        'med': np.median(pixel_i),
        'max': np.max(pixel_i)}

    paxel_i = np.sum(paxel_sequence, axis=0)
    info['raw_paxel'] = {
        'min': np.min(paxel_i),
        'med': np.median(paxel_i),
        'max': np.max(paxel_i)}

    lixel_i = np.sum(lixel_sequence, axis=0)
    info['raw_lixel'] = {
        'min': np.min(lixel_i),
        'med': np.median(lixel_i),
        'max': np.max(lixel_i),
        'sum': np.sum(lixel_i)}

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
                'x': evth.raw[98-1+1]/100,
                'y': evth.raw[118-1+1]/100},
            'scatter_radius': runh.raw[248-1]/100,
            'first_interaction_height': np.abs(evth.raw[7-1]/100),
            'observation_level_altitude_asl': runh.raw[6-1]/100}

    elif event.trigger_type == "EXTERNAL_RANDOM_TRIGGER":
        info['trigger_type'] = "EXTERNAL_RANDOM_TRIGGER"

    info['acp'] = {
        'response': collect_trigger_relevant_information(event),
        'light_field_sensor': {
            'expected_focal_length': event.light_field_geometry.expected_focal_length_of_imaging_system,
            'expected_aperture_radius': event.light_field_geometry.expected_aperture_radius_of_imaging_system,
            'number_pixel': event.light_field_geometry.number_pixel,
            'number_paxel': event.light_field_geometry.number_paxel,
            'time_slice_duration': event.raw_sensor_response.time_slice_duration,
            'number_time_slices': event.raw_sensor_response.number_time_slices,
        }
    }

    return un_numpyify(info)
