from sklearn.cluster import DBSCAN
import numpy as np
import array
from .photon_stream.cython_reader import stream2_cx_cy_arrivaltime_point_cloud
from .photon_stream.cython_reader import arrival_slices_and_lixel_ids
from .image.ImageRays import ImageRays
import os
import gzip
import json


def cluster_air_shower_photons_based_on_density(
    cx_cy_arrival_time_point_cloud,
    epsilon_cx_cy_radius=np.deg2rad(0.1),
    min_number_photons=20,
    deg_over_s=0.35e9,
):
    if cx_cy_arrival_time_point_cloud.shape[0] == 0:
        return np.array([])
    xyt = cx_cy_arrival_time_point_cloud.copy()
    xyt[:, 2] *= np.deg2rad(deg_over_s)

    dbscan = DBSCAN(
        eps=epsilon_cx_cy_radius, min_samples=min_number_photons
    ).fit(xyt)

    return dbscan.labels_


class RawPhotons:
    def __init__(
        self,
        photon_ids,
        arrival_slices,
        lixel_ids,
        light_field_geometry,
        t_pap,
        t_img,
    ):
        self.photon_ids = photon_ids
        self.arrival_slices = arrival_slices
        self.lixel_ids = lixel_ids
        self.t_pap = t_pap
        self.t_img = t_img
        self._light_field_geometry = light_field_geometry
        self._image_rays = ImageRays(self._light_field_geometry)

    @classmethod
    def from_event(cls, event):
        arrival_slices, lixel_ids = arrival_slices_and_lixel_ids(
            event.raw_sensor_response
        )
        return cls(
            photon_ids=np.arange(arrival_slices.shape[0], dtype=np.int),
            arrival_slices=arrival_slices,
            lixel_ids=lixel_ids,
            light_field_geometry=event.light_field_geometry,
            t_pap=(
                arrival_slices.astype(np.float)
                * event.raw_sensor_response["time_slice_duration"]
                + event.light_field_geometry.time_delay_mean[lixel_ids]
            ),
            t_img=(
                arrival_slices.astype(np.float)
                * event.raw_sensor_response["time_slice_duration"]
                + event.light_field_geometry.time_delay_image_mean[lixel_ids]
            ),
        )

    def cx_cy_in_object_distance(self, object_distance):
        cx, cy = self._image_rays.cx_cy_in_object_distance(object_distance)
        return cx[self.lixel_ids], cy[self.lixel_ids]

    @property
    def x(self):
        return self._light_field_geometry.x_mean[self.lixel_ids]

    @property
    def y(self):
        return self._light_field_geometry.y_mean[self.lixel_ids]

    @property
    def cx(self):
        return self._light_field_geometry.cx_mean[self.lixel_ids]

    @property
    def cy(self):
        return self._light_field_geometry.cy_mean[self.lixel_ids]

    @property
    def number(self):
        return len(self.photon_ids)

    def __repr__(self):
        return "{:s}({:d} photons)".format(
            self.__class__.__name__, self.number
        )

    def cut(self, mask):
        return RawPhotons(
            photon_ids=self.photon_ids[mask],
            arrival_slices=self.arrival_slices[mask],
            lixel_ids=self.lixel_ids[mask],
            light_field_geometry=self._light_field_geometry,
            t_pap=self.t_pap[mask],
            t_img=self.t_img[mask],
        )


def benchmark(pulse_origins, photon_ids_cherenkov):
    """
    Parameters
    ----------
    pulse_origins           Array 1D. The origins of the pulses in the raw
                            response of the light-field-sensor. Positive
                            integers encode the ids of the photon from the
                            input file used in the merlict photon-propagator.
                            The input files are usually Cherenkov-photons from
                            the KIT-CORSIKA air-shower-simulation.
                            Negative integers encode night-sky-background and
                            artifacts of the sensors.

    photon_ids_cherenkov    Array 1D. Ids of the photons which got classified
                            to be Cherenkov-photons from the air-shower.

    Returns
    -------
    number of matches       The number of true positive, false positive, true
                            negative, and false negative matches.

    """
    photon_ids_nsb = np.setdiff1d(
        np.arange(pulse_origins.shape[0]), photon_ids_cherenkov
    )

    is_cherenkov = pulse_origins >= 0
    is_nsb = pulse_origins < 0

    return {
        # is Cherenkov AND classified as Cherenkov
        # correctly identified
        "num_true_positives": int(is_cherenkov[photon_ids_cherenkov].sum()),
        # is Cherenkov AND classified as NSB
        # incorrectly rejected
        "num_false_negatives": int(is_cherenkov[photon_ids_nsb].sum()),
        # is NSB AND classified as Cherenkov
        # incorrectly identified
        "num_false_positives": int(is_nsb[photon_ids_cherenkov].sum()),
        # is NSB AND classified as NSB
        # correctly rejected
        "num_true_negatives": int(is_nsb[photon_ids_nsb].sum()),
    }


def cherenkov_photons_in_roi_in_image(
    roi,
    photons,
    roi_time_offset_start=-10e-9,
    roi_time_offset_stop=10e-9,
    roi_cx_cy_radius=np.deg2rad(2.0),
    roi_object_distance_offsets=np.linspace(4e3, -2e3, 4),
    dbscan_epsilon_cx_cy_radius=np.deg2rad(0.075),
    dbscan_min_number_photons=17,
    dbscan_deg_over_s=0.375e9,
):
    """
    Classify Cherenkov and night-sky-background-photons based on density in
    refocused image-sequences.
    For more performance, only photons within a certain region-of-interest
    (roi) are taken into account.

    Parameters
    ----------
    roi: dictionary, region-of-interest
        Can be taken from the trigger.
        - time_center_roi
        - cx_center_roi
        - cy_center_roi
        - object_distance
    photns: RawPhotons, all the photons recorded by the plenoscope.


    Radii for roi are chosen based on 0.25GeV to > 380GeV airshowers recorded
    with the 71m Portal Cherenkov-plenoscope. The defaults will give good
    results over all energies. Smaller radii will speed up the classification,
    but will also result in less performance for energies > 100GeV where often
    low surface-brightness is found over large areas.
    """
    start_time = roi["time_center_roi"] + roi_time_offset_start
    stop_time = roi["time_center_roi"] + roi_time_offset_stop
    roi_mask_time = (photons.t_img >= start_time) & (photons.t_img < stop_time)

    cx_roi = roi["cx_center_roi"]
    cy_roi = roi["cy_center_roi"]
    roi_cx_cy_radius_square = roi_cx_cy_radius ** 2
    cxs, cys = photons.cx_cy_in_object_distance(roi["object_distance"])
    cx_cy_square = (cx_roi - cxs) ** 2 + (cy_roi - cys) ** 2
    roi_mask_cx_cy = cx_cy_square <= roi_cx_cy_radius_square

    roi_mask = roi_mask_cx_cy & roi_mask_time
    photons_roi = photons.cut(roi_mask)
    num_photons_roi = photons_roi.t_img.shape[0]
    refocus_masks = []
    object_distances = roi["object_distance"] + roi_object_distance_offsets
    for object_distance in object_distances:
        assert object_distance > 0.0
        cxs, cys = photons_roi.cx_cy_in_object_distance(object_distance)
        photon_labels = cluster_air_shower_photons_based_on_density(
            cx_cy_arrival_time_point_cloud=np.c_[cxs, cys, photons_roi.t_img],
            epsilon_cx_cy_radius=dbscan_epsilon_cx_cy_radius,
            min_number_photons=dbscan_min_number_photons,
            deg_over_s=dbscan_deg_over_s,
        )
        refocus_masks.append(photon_labels >= 0)
    refocus_masks = np.array(refocus_masks)
    cherenkov_mask = np.sum(refocus_masks, axis=0) > 0

    settings = {
        "roi": {
            "time_center_roi": float(roi["time_center_roi"]),
            "cx_center_roi": float(roi["cx_center_roi"]),
            "cy_center_roi": float(roi["cy_center_roi"]),
            "object_distance": float(roi["object_distance"]),
        },
        "roi_time_offset_start": float(roi_time_offset_start),
        "roi_time_offset_stop": float(roi_time_offset_stop),
        "roi_cx_cy_radius": float(roi_cx_cy_radius),
        "roi_object_distance_offsets": list(roi_object_distance_offsets),
        "dbscan_epsilon_cx_cy_radius": float(dbscan_epsilon_cx_cy_radius),
        "dbscan_min_number_photons": int(dbscan_min_number_photons),
        "dbscan_deg_over_s": float(dbscan_deg_over_s),
    }

    return photons_roi.cut(cherenkov_mask), settings


def write_dense_photon_ids_to_event(event_path, photon_ids, settings):
    """
    Parameters
    ----------
    event_path, string
        Path to write photon_ids to.

    photon_ids, array uint

    settings, dictionary
        Returned by cherenkov_photons_in_roi_in_image().
    """
    assert os.path.exists(event_path)
    assert np.sum(photon_ids < 0) == 0
    assert np.sum(photon_ids >= 2 ** 32) == 0
    photon_ids_uint32 = photon_ids.astype(dtype=np.uint32)

    filename = "dense_photon_ids"

    settings_path = os.path.join(event_path, filename + ".settings.json")
    with open(settings_path, "wt") as fout:
        fout.write(json.dumps(settings, indent=4))

    photon_ids_path = os.path.join(event_path, filename + ".uint32.gz")
    with gzip.open(photon_ids_path, "wb") as fout:
        fout.write(photon_ids_uint32.tobytes())


def read_dense_photon_ids(path):
    with gzip.open(path, "rb") as fin:
        dense_photon_ids = np.frombuffer(fin.read(), dtype=np.uint32)
    return dense_photon_ids
