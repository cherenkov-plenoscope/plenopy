import os
import numpy as np
import scipy.spatial
import zipfile
import gzip
import posixpath
from . import utils
from .. import tools
from .. import image


def init_trigger_image_geometry(
    image_outer_radius_rad,
    pixel_spacing_rad,
    pixel_radius_rad,
    max_number_nearest_lixel_in_pixel,
):
    """
    Returns a dict with the basic geometry of the image where the
    trigger-decision is made. This is independent of the instrument's
    hardware.

    Parameter
    ---------
    image_outer_radius_rad : float
            The outer radius, (opening-angle), of the disc-shaped image.
    pixel_spacing_rad : float
            The angle between the centers of neighboring pixels in the image.
    pixel_radius_rad : float
            The outer radius (opening-angle) of a pixel.
    max_number_nearest_lixel_in_pixel : int
            An upper limit for the maximum number of light-field-cells (lixels)
            to be added into a pixel of the image.
    """
    assert image_outer_radius_rad > 0.0
    assert pixel_spacing_rad > 0.0
    assert pixel_radius_rad > 0.0
    grid_cx_cy = tools.hexagonal_grid.make_hexagonal_grid(
        outer_radius=image_outer_radius_rad,
        spacing=pixel_spacing_rad,
        inner_radius=0.0,
    )
    trg_img = {}
    trg_img["pixel_cx_rad"] = grid_cx_cy[:, 0]
    trg_img["pixel_cy_rad"] = grid_cx_cy[:, 1]
    trg_img["pixel_radius_rad"] = pixel_radius_rad
    trg_img["number_pixel"] = grid_cx_cy.shape[0]
    trg_img[
        "max_number_nearest_lixel_in_pixel"
    ] = max_number_nearest_lixel_in_pixel
    return trg_img


def init_trigger_geometry(
    light_field_geometry,
    trigger_image_geometry,
    object_distances=[7.5e3, 15e3, 22.5e3],
):
    """
    Returns a dict with the trigger's geometry.

    For each object_distance there is the projection-matrix describing
    which light-field-cells need to be added up into a pixel of the
    trigger-image.

    Parameters
    ----------
    light_field_geometry : class
            The light-field's geometry. I.e. the viewing-directions
            and support-positions of each light-field-cell.
    image_geometry : dict
            The geometry of the image where the trigger-decision is made.
            We use the same image_geometry for all foci/object-distances.
            This ensures that the actual hardware will be easily able to
            compare the responses in two, or more, different images.
    object_distances : list of floats
            The object-distances to focus the trigger-images to.
    """
    tg = {}
    tg["image"] = trigger_image_geometry
    tg["number_foci"] = len(object_distances)
    tg["number_lixel"] = np.uint32(light_field_geometry.number_lixel)
    tg["foci"] = []

    for object_distance in object_distances:
        lixel_to_pixel = estimate_projection_of_light_field_to_image(
            light_field_geometry=light_field_geometry,
            object_distance=object_distance,
            image_pixel_cx_rad=tg["image"]["pixel_cx_rad"],
            image_pixel_cy_rad=tg["image"]["pixel_cy_rad"],
            image_pixel_radius_rad=tg["image"]["pixel_radius_rad"],
            max_number_nearest_lixel_in_pixel=tg["image"][
                "max_number_nearest_lixel_in_pixel"
            ],
        )

        focus = utils.list_of_lists_to_arrays(list_of_lists=lixel_to_pixel)
        focus["object_distance_m"] = object_distance
        tg["foci"].append(focus)
    return tg


def estimate_projection_of_light_field_to_image(
    light_field_geometry,
    object_distance,
    image_pixel_cx_rad,
    image_pixel_cy_rad,
    image_pixel_radius_rad,
    max_number_nearest_lixel_in_pixel=7,
):
    """
    Returns a list over lixels of lists of pixels.
    For each lixel there is a list of pixels where the lixel has to be added
    to.
    [
        [pixel A, pixel B, pixel C, ... ],    lixel 0
        [pixel B, pixel C, pixel D, ... ],    lixel 1
        [pixel D, pixel E, pixel F, ... ],    lixel 2
        .
        .
        .
        [...],    lixel N
    ]

    Parameters
    ----------
    light_field_geometry

    object_distance             The object-distance the image is focused on.
                                float

    image_pixel_cx_rad          Array of floats

    image_pixel_cy_rad          Array of floats

    image_pixel_radius_rad      Array of floats
    """
    image_rays = image.ImageRays(light_field_geometry)
    lixel_cx, lixel_cy = image_rays.cx_cy_in_object_distance(object_distance)
    trigger_pixel_tree = scipy.spatial.cKDTree(
        np.array([image_pixel_cx_rad, image_pixel_cy_rad]).T
    )
    search = trigger_pixel_tree.query(
        x=np.vstack((lixel_cx, lixel_cy)).T,
        k=max_number_nearest_lixel_in_pixel,
    )
    projection_lixel_to_pixel = []
    lixel_to_pixel_distances_rad = search[0]
    lixel_to_pixel_ids = search[1]
    for lix in range(light_field_geometry.number_lixel):
        lixel_to_pixel = []
        for pix in range(max_number_nearest_lixel_in_pixel):
            dd = lixel_to_pixel_distances_rad[lix, pix]
            if dd <= image_pixel_radius_rad:
                lixel_to_pixel.append(lixel_to_pixel_ids[lix, pix])
        projection_lixel_to_pixel.append(lixel_to_pixel)
    return projection_lixel_to_pixel


def invert_projection_matrix(lixel_to_pixel, number_pixel, number_lixel):
    assert number_lixel == len(lixel_to_pixel)
    pixel_to_lixel = [[] for p in range(number_pixel)]
    for lixel in range(number_lixel):
        for pixel in lixel_to_pixel[lixel]:
            pixel_to_lixel[pixel].append(lixel)
    return pixel_to_lixel


def init_summation_statistics(trigger_geometry):
    """
    Returns a dict of count-statistics.
    Counting how many light-field-cells are combined into one picture-cell.
    """
    tg = trigger_geometry

    stats = {}
    stats["number_foci"] = int(tg["number_foci"])
    stats["number_pixel"] = int(tg["image"]["number_pixel"])
    stats["number_lixel"] = int(tg["number_lixel"])

    stats["foci"] = []
    for focus in range(tg["number_foci"]):
        lixel_to_pixel = utils.arrays_to_list_of_lists(
            starts=tg["foci"][focus]["starts"],
            lengths=tg["foci"][focus]["lengths"],
            links=tg["foci"][focus]["links"],
        )

        pixel_to_lixel = invert_projection_matrix(
            lixel_to_pixel=lixel_to_pixel,
            number_pixel=tg["image"]["number_pixel"],
            number_lixel=tg["number_lixel"],
        )

        stat = {}
        stat["number_lixel_in_pixel"] = [len(pix) for pix in pixel_to_lixel]
        stat["number_pixel_in_lixel"] = [len(lix) for lix in lixel_to_pixel]

        stats["foci"].append(stat)
    return stats


def suggested_filename_extension():
    return ".trigger_geometry.zip"


def write(trigger_geometry, path):
    assert_trigger_geometry_consistent(trigger_geometry=trigger_geometry)
    tg = trigger_geometry
    join = posixpath.join

    with zipfile.ZipFile(file=path, mode="w") as zout:
        _zwrite(zout, "number_lixel", tg["number_lixel"], "u4")

        _zwrite(zout, "image.number_pixel", tg["image"]["number_pixel"], "u4")
        _zwrite(
            zout,
            "image.max_number_nearest_lixel_in_pixel",
            tg["image"]["max_number_nearest_lixel_in_pixel"],
            "u4",
        )
        _zwrite(zout, "image.pixel_cx_rad", tg["image"]["pixel_cx_rad"], "f4")
        _zwrite(zout, "image.pixel_cy_rad", tg["image"]["pixel_cy_rad"], "f4")
        _zwrite(
            zout,
            "image.pixel_radius_rad",
            tg["image"]["pixel_radius_rad"],
            "f4",
        )

        _zwrite(zout, "number_foci", tg["number_foci"], "u4")
        for focus in range(tg["number_foci"]):
            name = "foci.{:06d}".format(focus)
            _zwrite(
                zout,
                name + ".object_distance_m",
                tg["foci"][focus]["object_distance_m"],
                "f4",
            )
            _zwrite(zout, name + ".starts", tg["foci"][focus]["starts"], "u4")
            _zwrite(
                zout, name + ".lengths", tg["foci"][focus]["lengths"], "u4"
            )
            _zwrite(zout, name + ".links", tg["foci"][focus]["links"], "u4")


def read(path):
    join = posixpath.join
    tg = {}
    with zipfile.ZipFile(file=path, mode="r") as zin:
        tg["number_lixel"] = _zread(zin, "number_lixel", "u4")[0]

        tg["image"] = {}
        tg["image"]["number_pixel"] = _zread(zin, "image.number_pixel", "u4")[
            0
        ]
        tg["image"]["max_number_nearest_lixel_in_pixel"] = _zread(
            zin, "image.max_number_nearest_lixel_in_pixel", "u4"
        )[0]
        tg["image"]["pixel_cx_rad"] = _zread(zin, "image.pixel_cx_rad", "f4")
        tg["image"]["pixel_cy_rad"] = _zread(zin, "image.pixel_cy_rad", "f4")
        tg["image"]["pixel_radius_rad"] = _zread(
            zin, "image.pixel_radius_rad", "f4"
        )[0]

        tg["number_foci"] = _zread(zin, "number_foci", "u4")[0]
        tg["foci"] = [{} for i in range(tg["number_foci"])]
        for focus in range(tg["number_foci"]):
            name = "foci.{:06d}".format(focus)
            tg["foci"][focus]["object_distance_m"] = _zread(
                zin, name + ".object_distance_m", "f4"
            )[0]
            tg["foci"][focus]["starts"] = _zread(zin, name + ".starts", "u4")
            tg["foci"][focus]["lengths"] = _zread(zin, name + ".lengths", "u4")
            tg["foci"][focus]["links"] = _zread(zin, name + ".links", "u4")

        assert_trigger_geometry_consistent(trigger_geometry=tg)
        return tg


def _zwrite(zout, name_without_extension, value, dtype):
    with zout.open(name_without_extension + "." + dtype + ".gz", "w") as f:
        payload = np.float32(value).astype(dtype).tobytes()
        payload_gz = gzip.compress(payload)
        f.write(payload_gz)


def _zread(zin, name_without_extension, dtype):
    with zin.open(name_without_extension + "." + dtype + ".gz", "r") as f:
        payload_gz = f.read()
        payload = gzip.decompress(payload_gz)
        return np.frombuffer(payload, dtype=dtype)


def plot(trigger_geometry_path, out_dir):
    """
    Try to plot the geometry of the sum-trigger.

    Parameters
    ----------
    trigger_geometry_path : str
        Zipfile written by plenopy.trigger.geometry.write().
    out_dir : str
        Write figures to this directory.
    """
    try:
        import subprocess
        import importlib
        from importlib import resources

        plenopy_trigger_script_plot_path = os.path.join(
            str(importlib.resources.files("plenopy")),
            "trigger",
            "scripts",
            "plot.py",
        )
        subprocess.call(
            [
                "python",
                plenopy_trigger_script_plot_path,
                trigger_geometry_path,
                out_dir,
            ]
        )
    except:
        pass


def assert_trigger_geometry_consistent(trigger_geometry):
    tg = trigger_geometry
    assert tg["image"]["number_pixel"] == tg["image"]["pixel_cx_rad"].shape[0]
    assert tg["image"]["number_pixel"] == tg["image"]["pixel_cy_rad"].shape[0]
    assert tg["image"]["pixel_radius_rad"] >= 0.0

    for focus in range(tg["number_foci"]):
        assert tg["number_lixel"] == tg["foci"][focus]["starts"].shape[0]
        assert tg["number_lixel"] == tg["foci"][focus]["lengths"].shape[0]
