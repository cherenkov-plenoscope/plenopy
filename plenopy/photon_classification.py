from sklearn.cluster import DBSCAN
import numpy as np
import array
from .photon_stream.cython_reader import stream2_cx_cy_arrivaltime_point_cloud
from .image.ImageRays import ImageRays
from . import trigger


def classify_air_shower_photons(
    light_field_geometry,
    raw_sensor_response,
    start_time_roi,
    end_time_roi,
    cx_center_roi,
    cy_center_roi,
    cx_cy_radius_roi,
    object_distances,
    deg_over_s,
    epsilon_cx_cy_radius,
    min_number_photons,
):
    """
    Classifiy air-shower-photons and night-sky-background photons based on the
    higher density of air-shower-photons in the refocussed image-sequences.

    To speed up the classification, you have to restrict to a Region of Interest
    (roi) in both directions cx, and cy, and time.
    """
    imrays = ImageRays(light_field_geometry)
    air_shower_photon_ids = []
    for i, object_distance in enumerate(object_distances):
        cx, cy = imrays.cx_cy_in_object_distance(object_distance)
        cxcyt, lixel_ids = stream2_cx_cy_arrivaltime_point_cloud(
            photon_stream=raw_sensor_response.photon_stream,
            time_slice_duration=raw_sensor_response.time_slice_duration,
            NEXT_READOUT_CHANNEL_MARKER=raw_sensor_response.NEXT_READOUT_CHANNEL_MARKER,
            cx=cx,
            cy=cy,
            time_delay=light_field_geometry.time_delay_mean)

        photon_ids = np.arange(
            raw_sensor_response.number_photons,
            dtype=np.uint32)

        mask_time_roi = mask_time_in_point_cloud(
            pcl=cxcyt,
            start_time=start_time_roi,
            end_time=end_time_roi)

        photon_ids_t = photon_ids[mask_time_roi]
        lixel_ids_t = lixel_ids[mask_time_roi]
        cxcyt_t = cxcyt[mask_time_roi]

        mask_cx_cy_roi = mask_cxcy_in_point_cloud(
            cx=cxcyt_t[:, 0],
            cy=cxcyt_t[:, 1],
            cx_center=cx_center_roi,
            cy_center=cy_center_roi,
            c_radius=cx_cy_radius_roi)

        photon_ids_tc = photon_ids_t[mask_cx_cy_roi]
        lixel_ids_tc = lixel_ids_t[mask_cx_cy_roi]
        cxcyt_tc = cxcyt_t[mask_cx_cy_roi]


        photon_labels = cluster_air_shower_photons_based_on_density(
            cx_cy_arrival_time_point_cloud=cxcyt_tc,
            epsilon_cx_cy_radius=epsilon_cx_cy_radius,
            min_number_photons=min_number_photons,
            deg_over_s=deg_over_s)
        mask_air_shower = photon_labels >= 0

        photon_ids_air_shower = photon_ids_tc[mask_air_shower]
        cxcyt_air_shower = cxcyt_tc[mask_air_shower]
        lixel_ids_air_shower = lixel_ids_tc[mask_air_shower]

        air_shower_photon_ids.append(photon_ids_air_shower)

    w = []
    for l in air_shower_photon_ids:
        for s in l:
            w.append(s)
    air_shower_photon_ids = np.array(list(set(w)))
    return air_shower_photon_ids, lixel_ids


def mask_time_in_point_cloud(pcl, start_time, end_time):
    before_end = pcl[:, 2] <= end_time
    after_start = pcl[:, 2] > start_time
    return before_end*after_start


def mask_cxcy_in_point_cloud(cx, cy, cx_center, cy_center, c_radius):
    distance_to_center_2 = (cx - cx_center)**2 + (cy - cy_center)**2
    return distance_to_center_2 <= c_radius**2


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


def classify_air_shower_photons_from_trigger_response(
    event,
    trigger_region_of_interest,
    roi_time_radius=5e-9,
    roi_cx_cy_radius=np.deg2rad(0.3),
    roi_object_distance_radius=5e3,
    deg_over_s=0.20e9,
    refocusses_for_classification=7,
    epsilon_cx_cy_radius=np.deg2rad(0.055),
    min_number_photons=9
):
    """
    These defaults give for the 71m ACP 61-paxel x 8433-pixel:
    true air-shower-photons over nsb-photons: 0.62
    median number of photons in cluster: 131
    fratction of all true air-shower-photons: 0.65
    """
    roi = trigger_region_of_interest
    return classify_air_shower_photons(
        light_field_geometry=event.light_field_geometry,
        raw_sensor_response=event.raw_sensor_response,
        start_time_roi=roi['time_center_roi'] - roi_time_radius,
        end_time_roi=roi['time_center_roi'] + roi_time_radius,
        cx_center_roi=roi['cx_center_roi'],
        cy_center_roi=roi['cy_center_roi'],
        cx_cy_radius_roi=roi_cx_cy_radius,
        object_distances=np.logspace(
            np.log10(roi['object_distance'] - roi_object_distance_radius),
            np.log10(roi['object_distance'] + roi_object_distance_radius),
            refocusses_for_classification),
        deg_over_s=deg_over_s,
        epsilon_cx_cy_radius=epsilon_cx_cy_radius,
        min_number_photons=min_number_photons)