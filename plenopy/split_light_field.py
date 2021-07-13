import numpy as np


def make_split_light_field(loph_record, light_field_geometry):
    """
    The light-field is split into image-sequences, each taken at a different
    principal-aperture-cell (paxel).
    In natural units, not in detector specific units.

    parameter
    ---------
    loph_record
            A light-field recorded by the plenoscope in 'List-of-Photons'
            (LOPH) representation.
    light_field_geometry
             The plenoscope's light-firld-geometry.

    Fields
    ------
    image_sequences : list
            A list of image-sequences. Each image-sequence is an array
            of photons. Each photon has three coordinates cx, cy and t.
    number_photons : int
            Number of photons in the entire light-field.
    number_paxel : int
            Number of principal-aperture-cells (paxel), thus
            number of image-sequences.
    paxel_pos_x /
    paxel_pos_y : list of floats
            The positions of the paxels on the aperture-plane.
    median_cx /
    median_cy : float
            The median direction of all photons in the
            light-field.
    """

    out = {}
    out["number_photons"] = loph_record["photons"]["arrival_time_slices"].shape[0]
    out["number_paxel"] = light_field_geometry.number_paxel
    out["paxel_pos_x"] = light_field_geometry.paxel_pos_x
    out["paxel_pos_y"] = light_field_geometry.paxel_pos_y

    out["image_sequences"] = [[] for pax in range(out["number_paxel"])]
    _, ph_paxel = light_field_geometry.pixel_and_paxel_of_lixel(
        lixel=loph_record["photons"]["channels"]
    )
    ph_cx = light_field_geometry.cx_mean[loph_record["photons"]["channels"]]
    ph_cy = light_field_geometry.cy_mean[loph_record["photons"]["channels"]]
    ph_t = (
        loph_record["sensor"]["time_slice_duration"]
        * loph_record["photons"]["arrival_time_slices"]
    )

    for ph in range(out["number_photons"]):
        pax = ph_paxel[ph]
        out["image_sequences"][pax].append([ph_cx[ph], ph_cy[ph], ph_t[ph]])

    for pax in range(out["number_paxel"]):
        if len(out["image_sequences"][pax]) > 0:
            out["image_sequences"][pax] = np.array(out["image_sequences"][pax])
        else:
            out["image_sequences"][pax] = np.zeros(
                shape=(0, 3),
                dtype=np.float32
            )
    return out


def median_cx_cy(loph_record, light_field_geometry):
    ph_cx = light_field_geometry.cx_mean[loph_record["photons"]["channels"]]
    ph_cy = light_field_geometry.cy_mean[loph_record["photons"]["channels"]]
    return np.median(ph_cx), np.median(ph_cy)
