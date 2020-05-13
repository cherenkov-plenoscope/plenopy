import numpy as np
from .image import ImageRays
import scipy.spatial
import array


def make_hexagonal_grid(
    outer_radius,
    spacing,
    inner_radius=0.0
):
    hex_a = np.array([np.sqrt(3)/2, 0.5])*spacing
    hex_b = np.array([0, 1])*spacing

    grid = []
    sample_radius = 2.0*np.floor(outer_radius/spacing);
    for a in np.arange(-sample_radius, sample_radius+1):
        for b in np.arange(-sample_radius, sample_radius+1):
            cell_ab = hex_a*a + hex_b*b
            cell_norm = np.linalg.norm(cell_ab)
            if cell_norm <= outer_radius and cell_norm >= inner_radius:
                grid.append(cell_ab);
    return np.array(grid)


def estimate_projection_of_light_field_to_image(
    light_field_geometry,
    object_distance,
    image_pixel_cx_rad,
    image_pixel_cy_rad,
    image_pixel_radius_rad,
    number_nearest_neighbors=100,
):
    image_rays = ImageRays(light_field_geometry)
    lixel_cx, lixel_cy = image_rays.cx_cy_in_object_distance(object_distance)
    trigger_pixel_tree = scipy.spatial.cKDTree(
        np.array([
            image_pixel_cx_rad,
            image_pixel_cy_rad
        ]).T
    )
    search = trigger_pixel_tree.query(
        x=np.vstack((lixel_cx, lixel_cy)).T,
        k=number_nearest_neighbors,
    )
    projection = []
    lixel_to_pixel_distances_rad = search[0]
    lixel_to_pixel_ids = search[1]
    for lix in range(light_field_geometry.number_lixel):
        lixel_to_pixel = []
        for pix in range(number_nearest_neighbors):
            dd = lixel_to_pixel_distances_rad[lix, pix]
            if dd <= image_pixel_radius_rad:
                lixel_to_pixel.append(lixel_to_pixel_ids[lix, pix])
        projection.append(lixel_to_pixel)
    return projection


def list_of_lists_to_arrays(list_of_lists):
    starts = array.array('l')
    lengths = array.array('l')
    stream = array.array('l')
    i = 0
    for _list in list_of_lists:
        starts.append(i)
        l = 0
        for symbol in _list:
            stream.append(symbol)
            i += 1
            l += 1
        lengths.append(l)
    return {
        "starts": np.array(starts),
        "lengths": np.array(lengths),
        "links": np.array(stream),
    }


def project_light_firld_sequence_onto_image_sequence(
    raw_photon_stream,
    raw_photon_stream_next_channel_marker,
    projection_links,
    projection_starts,
    projection_lengths,
    number_pixel,
    number_time_slices,
    time_slice_duration,
    lixel_time_delays
):
    out_image_sequence = np.zeros(
        shape=(number_time_slices, number_pixel),
        dtype=np.uint16
    )

    num_phs_symbols = raw_photon_stream.shape[0]

    phs_lixel = 0
    for phs_i in range(num_phs_symbols):
        phs_symbol = raw_photon_stream[phs_i]

        if phs_symbol == raw_photon_stream_next_channel_marker:
            phs_lixel += 1
        else:
            raw_arrival_time_slice = phs_symbol

            arrival_time = raw_arrival_time_slice*time_slice_duration
            arrival_time -= lixel_time_delays[phs_lixel]

            arrival_slice = int(np.round(arrival_time/time_slice_duration))

            if arrival_slice < number_time_slices and arrival_slice >= 0:

                for p in range(projection_lengths[phs_lixel]):
                    pp = projection_starts[phs_lixel] + p
                    pixel = projection_links[pp]
                    out_image_sequence[arrival_slice, pixel] += 1

    return out_image_sequence
