import numpy as np


def from_trigger_response(
    trigger_response, trigger_geometry, time_slice_duration,
):
    """
    Export the position of the largest trigger-response in absolute
    directions cx/rad, cy/rad, time/s, and object_distance/m.
    """
    tg = trigger_geometry
    foci_responses = [_focus["response_pe"] for _focus in trigger_response]
    focus = np.argmax(foci_responses)

    time_slice = trigger_response[focus]["time_slice"]
    pixel = trigger_response[focus]["pixel"]

    return {
        "time_center_roi": time_slice * time_slice_duration,
        "cx_center_roi": tg["image"]["pixel_cx_rad"][pixel],
        "cy_center_roi": tg["image"]["pixel_cy_rad"][pixel],
        "object_distance": tg["foci"][focus]["object_distance_m"],
    }
