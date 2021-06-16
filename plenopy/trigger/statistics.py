import numpy as np
from .prepare import arrays_to_list_of_lists
from .prepare import invert_projection_matrix


def gather_summation_statistics(trigger_geometry):
    tg = trigger_geometry

    stats = {}
    stats['number_foci'] = int(tg['number_foci'])
    stats['number_pixel'] = int(tg['image']['number_pixel'])
    stats['number_lixel'] = int(tg['number_lixel'])

    stats['foci'] = []
    for focus in range(tg['number_foci']):
        lixel_to_pixel = arrays_to_list_of_lists(
            starts=tg['foci'][focus]['starts'],
            lengths=tg['foci'][focus]['lengths'],
            links=tg['foci'][focus]['links']
        )

        pixel_to_lixel = invert_projection_matrix(
            lixel_to_pixel=lixel_to_pixel,
            number_pixel=tg['image']['number_pixel'],
            number_lixel=tg['number_lixel'],
        )

        stat = {}
        stat['number_lixel_in_pixel'] = [len(pix) for pix in pixel_to_lixel]
        stat['number_pixel_in_lixel'] = [len(lix) for lix in lixel_to_pixel]

        stats['foci'].append(stat)
    return stats
