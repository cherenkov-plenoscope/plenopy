from sklearn.cluster import DBSCAN
import numpy as np
import array
from .photon_stream.cython_reader import stream2_cx_cy_arrivaltime_point_cloud
from .photon_stream.cython_reader import arrival_slices_and_lixel_ids
from .image.ImageRays import ImageRays
from . import trigger


def cluster_air_shower_photons_based_on_density(
    cx_cy_arrival_time_point_cloud,
    epsilon_cx_cy_radius=np.deg2rad(0.1),
    min_number_photons=20,
    deg_over_s=0.35e9
):
    if cx_cy_arrival_time_point_cloud.shape[0] == 0:
        return np.array([])
    xyt = cx_cy_arrival_time_point_cloud.copy()
    xyt[:, 2] *= np.deg2rad(deg_over_s)

    dbscan = DBSCAN(
        eps=epsilon_cx_cy_radius,
        min_samples=min_number_photons
    ).fit(xyt)

    return dbscan.labels_


def center_for_region_of_interest(event):
    trigger_response = trigger.read_trigger_response_of_event(event)
    return trigger.region_of_interest_from_trigger_response(
        trigger_response=trigger_response,
        time_slice_duration=event.raw_sensor_response.time_slice_duration,
        pixel_pos_cx=event.light_field_geometry.pixel_pos_cx,
        pixel_pos_cy=event.light_field_geometry.pixel_pos_cy)


class RawPhotons():
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
            event.raw_sensor_response)
        return cls(
            photon_ids=np.arange(arrival_slices.shape[0], dtype=np.int),
            arrival_slices=arrival_slices,
            lixel_ids=lixel_ids,
            light_field_geometry=event.light_field_geometry,
            t_pap= (
                arrival_slices.astype(np.float)*
                event.raw_sensor_response.time_slice_duration +
                event.light_field_geometry.time_delay_mean[lixel_ids]),
            t_img= (
                arrival_slices.astype(np.float)*
                event.raw_sensor_response.time_slice_duration +
                event.light_field_geometry.time_delay_image_mean[lixel_ids])
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

    def __repr__(self):
        out = 'RawPhotons(' + str(len(self.photon_ids)) +' photons)'
        return out

    def cut(self, mask):
        return RawPhotons(
            photon_ids=self.photon_ids[mask],
            arrival_slices=self.arrival_slices[mask],
            lixel_ids=self.lixel_ids[mask],
            light_field_geometry=self._light_field_geometry,
            t_pap=self.t_pap[mask],
            t_img=self.t_img[mask])


def benchmark(
    pulse_origins,
    photon_ids_cherenkov
):
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
        np.arange(pulse_origins.shape[0]),
        photon_ids_cherenkov)

    is_cherenkov = pulse_origins >= 0
    is_nsb = pulse_origins < 0

    return {
        # is Cherenkov AND classified as Cherenkov
        'number_true_positives': is_cherenkov[photon_ids_cherenkov].sum(),
        # is Cherenkov AND classified as NSB
        'number_false_positives': is_cherenkov[photon_ids_nsb].sum(),
        # is NSB AND classified as Cherenkov
        'number_false_negatives': is_nsb[photon_ids_cherenkov].sum(),
        # is NSB AND classified as NSB
        'number_true_negatives': is_nsb[photon_ids_nsb].sum()}


def cherenkov_photons_in_roi_in_image(
    roi,
    photons,
    time_radius_roi=5e-9,
    c_radius=np.deg2rad(0.3),
    epsilon_cx_cy_radius=np.deg2rad(0.075),
    min_number_photons=17,
    deg_over_s=0.375e9,
    number_refocuses=5,
    object_distance_radius=2.5e3,
    object_distances=None
):
    start_time = roi['time_center_roi'] - time_radius_roi
    end_time = roi['time_center_roi'] + time_radius_roi
    roi_mask_time = (
        (photons.t_img >= start_time) & (photons.t_img < end_time))
    ph_cx, ph_cy = photons.cx_cy_in_object_distance(roi['object_distance'])
    ph_c_distance_square = (
        (roi['cx_center_roi'] - ph_cx)**2 +
        (roi['cy_center_roi'] - ph_cy)**2)
    c_radius_square = c_radius**2
    roi_mask_c = ph_c_distance_square <= c_radius_square
    roi_mask = roi_mask_time & roi_mask_c
    photons_in_roi = photons.cut(roi_mask)

    if object_distances is None:
        object_distances = np.linspace(
            roi['object_distance'] - object_distance_radius,
            roi['object_distance'] + object_distance_radius,
            number_refocuses)
    cherenkov_mask = np.zeros(
        photons_in_roi.photon_ids.shape[0], dtype=np.bool)
    for object_distance in object_distances:
        photon_labels = cluster_air_shower_photons_based_on_density(
            cx_cy_arrival_time_point_cloud=np.c_[
                photons_in_roi.cx_cy_in_object_distance(object_distance)[0],
                photons_in_roi.cx_cy_in_object_distance(object_distance)[1],
                photons_in_roi.t_img],
            epsilon_cx_cy_radius=epsilon_cx_cy_radius,
            min_number_photons=min_number_photons,
            deg_over_s=deg_over_s)
        refocus_mask = photon_labels >= 0
        cherenkov_mask = np.logical_or(cherenkov_mask, refocus_mask)
    return photons_in_roi.cut(cherenkov_mask)


import scipy


def cherenkov_photons_in_image(
    photons,
    light_field_geometry,
    object_distances=np.geomspace(2.5e3, 25e3, 5),
    pixel_seed_threshold=99 + 6*10,
    min_num_neighbors=2
):
    FOV_RADIUS = 0.5*light_field_geometry.\
        sensor_plane2imaging_system.max_FoV_diameter
    PIXEL_FOV_DIAMETER = light_field_geometry.\
        sensor_plane2imaging_system.pixel_FoV_hex_flat2flat

    neighborhood = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]])

    c_bin_edges = np.arange(-FOV_RADIUS, FOV_RADIUS, PIXEL_FOV_DIAMETER)
    num_pixel_diagonal = c_bin_edges.shape[0] - 1
    num_photons = photons.cx.shape[0]

    mask = np.zeros(num_photons, dtype=np.bool)
    for object_distance in object_distances:
        cx, cy = photons.cx_cy_in_object_distance(object_distance)
        cx_pixel_idx = np.digitize(cx, bins=c_bin_edges)
        cy_pixel_idx = np.digitize(cy, bins=c_bin_edges)
        image = np.histogram2d(cx, cy, bins=[c_bin_edges, c_bin_edges])[0]
        seeds = image > pixel_seed_threshold
        neighbors_above_seed_threshold = (
            scipy.signal.convolve2d(
                seeds,
                neighborhood,
                mode='same') > min_num_neighbors)
        all_pixel_ids = np.arange(num_pixel_diagonal**2)
        pixel_ids_above_threshold = all_pixel_ids[
            neighbors_above_seed_threshold.flatten()]

        cx_pixel_idx_above_threshold =\
            pixel_ids_above_threshold//num_pixel_diagonal
        cy_pixel_idx_above_threshold =\
            np.mod(pixel_ids_above_threshold, num_pixel_diagonal)

        for i in range(cx_pixel_idx_above_threshold.shape[0]):
            cxi = cx_pixel_idx_above_threshold[i]
            cyi = cy_pixel_idx_above_threshold[i]
            match = (cx_pixel_idx == cxi)*(cy_pixel_idx == cyi)
            mask = np.logical_or(mask, match)

    cherenkov_photons = photons.cut(mask)
    return cherenkov_photons
