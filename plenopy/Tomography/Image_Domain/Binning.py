import numpy as np
import json
from ...Thin_Lens import object_distance_2_image_distance as g2b
from ...Thin_Lens import image_distance_2_object_distance as b2g


def init(
    focal_length,
    cx_min=np.deg2rad(-3.5),
    cx_max=np.deg2rad(+3.5),
    number_cx_bins=96,
    cy_min=np.deg2rad(-3.5),
    cy_max=np.deg2rad(3.5),
    number_cy_bins=96,
    obj_min=5e3,
    obj_max=25e3,
    number_obj_bins=32,
):
    b = {}
    b["focal_length"] = focal_length

    b["cx_min"] = cx_min
    b["cx_max"] = cx_max
    b["number_cx_bins"] = number_cx_bins
    (
        b["cx_bin_edges"],
        b["cx_bin_centers"],
        b["cx_width"],
        b["cx_bin_radius"],
    ) = linspace_edges_centers(start=cx_min, stop=cx_max, num=number_cx_bins)
    b["sen_x_min"] = np.tan(cx_min) * focal_length
    b["sen_x_max"] = np.tan(cx_max) * focal_length
    b["number_sen_x_bins"] = number_cx_bins
    b["sen_x_width"] = b["sen_x_max"] - b["sen_x_min"]
    b["sen_x_bin_radius"] = np.tan(b["cx_bin_radius"]) * focal_length
    b["sen_x_bin_edges"] = np.tan(b["cx_bin_edges"]) * focal_length
    b["sen_x_bin_centers"] = np.tan(b["cx_bin_centers"]) * focal_length

    b["cy_min"] = cy_min
    b["cy_max"] = cy_max
    b["number_cy_bins"] = number_cy_bins
    (
        b["cy_bin_edges"],
        b["cy_bin_centers"],
        b["cy_width"],
        b["cy_bin_radius"],
    ) = linspace_edges_centers(start=cy_min, stop=cy_max, num=number_cy_bins)
    b["sen_y_min"] = np.tan(cy_min) * focal_length
    b["sen_y_max"] = np.tan(cy_max) * focal_length
    b["number_sen_y_bins"] = number_cy_bins
    b["sen_y_width"] = b["sen_y_max"] - b["sen_y_min"]
    b["sen_y_bin_radius"] = np.tan(b["cy_bin_radius"]) * focal_length
    b["sen_y_bin_edges"] = np.tan(b["cy_bin_edges"]) * focal_length
    b["sen_y_bin_centers"] = np.tan(b["cy_bin_centers"]) * focal_length

    b["number_bins"] = number_cx_bins * number_cy_bins * number_obj_bins

    b["sen_z_min"] = g2b(obj_max, focal_length)
    b["sen_z_max"] = g2b(obj_min, focal_length)
    b["number_sen_z_bins"] = number_obj_bins
    (
        b["sen_z_bin_edges"],
        b["sen_z_bin_centers"],
        b["sen_z_width"],
        b["sen_z_bin_radius"],
    ) = linspace_edges_centers(
        start=b["sen_z_min"], stop=b["sen_z_max"], num=b["number_sen_z_bins"]
    )

    b["obj_min"] = obj_min
    b["obj_max"] = obj_max
    b["number_obj_bins"] = number_obj_bins
    b["obj_bin_edges"] = b2g(b["sen_z_bin_edges"], focal_length)
    b["obj_bin_centers"] = b2g(b["sen_z_bin_centers"], focal_length)

    return b


BINNING_CONSTRUCTORS = {
    "focal_length": float,
    "cx_min": float,
    "cx_max": float,
    "number_cx_bins": int,
    "cy_min": float,
    "cy_max": float,
    "number_cy_bins": int,
    "obj_min": float,
    "obj_max": float,
    "number_obj_bins": int,
}


def write(binning, path):
    binning_ctor_dict = {}
    for key in BINNING_CONSTRUCTORS:
        _ctor = BINNING_CONSTRUCTORS[key]
        binning_ctor_dict[key] = _ctor(binning[key])
    with open(path, "wt") as f:
        f.write(json.dumps(binning_ctor_dict, indent=4))


def read(path):
    with open(path, "rt") as f:
        binning_ctor_dict = json.loads(f.read())
    return init(**binning_ctor_dict)


def is_equal(binning_a, binning_b):
    for key in BINNING_CONSTRUCTORS:
        if binning_a[key] != binning_b[key]:
            return False
    return True


def linspace_edges_centers(start, stop, num):
    bin_edges = np.linspace(start, stop, num + 1)
    width = stop - start
    bin_radius = 0.5 * width / num
    bin_centers = bin_edges[:-1] + bin_radius
    return bin_edges, bin_centers, width, bin_radius


def volume_intensity_as_cube(volume_intensity, binning):
    return volume_intensity.reshape(
        (
            binning["number_sen_x_bins"],
            binning["number_sen_y_bins"],
            binning["number_sen_z_bins"],
        ),
        order="C",
    )
