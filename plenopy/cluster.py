from sklearn.cluster import DBSCAN
import numpy as np


def air_shower_photons(light_field):
    pass


def raw_phs_2_point_cloud(
    cx_mean,
    cy_mean,
    time_delay_mean,
    raw_sensor_response
):
    num_photons = (
        raw_sensor_response.photon_stream.shape[0] -
        (raw_sensor_response.number_lixel - 1))
    cxcyat = np.zeros(shape=(num_photons, 3))
    lixel_ids = np.zeros(num_photons, dtype=np.uint32)
    photon = 0
    lixel = 0
    for symbol in raw_sensor_response.photon_stream:
        if symbol == event.raw_sensor_response.NEXT_READOUT_CHANNEL_MARKER:
            lixel += 1
        else:
            time_slice = symbol
            arrival_time = time_slice*raw_sensor_response.time_slice_duration
            arrival_time -= time_delay_mean[lixel]

            cxcyat[photon, 0] = cx_mean[lixel]
            cxcyat[photon, 1] = cy_mean[lixel]
            cxcyat[photon, 2] = arrival_time
            lixel_ids[photon] = lixel
            photon += 1
    return cxcyat, lixel_ids


def mask_time_in_point_cloud(pcl, start_time, end_time):
    before_end = pcl[:, 2] <= end_time
    after_start = pcl[:, 2] > start_time
    return before_end*after_start


def mask_cxcy_in_point_cloud(pcl, cx_center, cy_center, c_radius):
    distance_to_center = np.sqrt(
        (pcl[:, 0] - cx_center)**2 +
        (pcl[:, 1] - cy_center)**2)
    return distance_to_center <= c_radius



class PhotonStreamCluster(object):
    def __init__(
        self,
        pcl,
        epsilon=np.deg2rad(0.1),
        min_samples=20,
        deg_over_s=0.35e9
    ):
        self.point_cloud = pcl
        self.xyt = self.point_cloud.copy()
        self.xyt[:, 2] *= np.deg2rad(deg_over_s)

        if self.xyt.shape[0] == 0:
            self.labels = np.array([])
            self.number = 0
            return

        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(self.xyt)
        self.labels = dbscan.labels_

        # Number of clusters in labels, ignoring background if present.
        self.number = len(set(self.labels)) - (1 if -1 in self.labels else 0)

    def __repr__(self):
        out = '{}('.format(self.__class__.__name__)
        out += 'number of clusters '+str(self.number)
        out += ')\n'
        return out

import photon_stream.plot as psplot

light_field = event.light_field
raw_sensor_response = event.raw_sensor_response
start_time = t[0]['time_slice_with_most_active_neighboring_patches']*light_field.time_slice_duration - 5e-9
end_time = t[0]['time_slice_with_most_active_neighboring_patches']*light_field.time_slice_duration + 5e-9

trigger_pixel = t[0]['patches'][0]

cx_center = light_field.pixel_pos_cx[trigger_pixel]
cy_center = light_field.pixel_pos_cy[trigger_pixel]
c_radius = np.deg2rad(0.5)

object_distances = np.linspace(5e3, 20e3, 15)

imrays = pl.image.ImageRays(light_field)

for i, object_distance in enumerate(object_distances):

    cx, cy = imrays.cx_cy_in_object_distance(object_distance)
    cxcyt, lixel_ids = raw_phs_2_point_cloud(
        cx_mean=cx,
        cy_mean=cy,
        time_delay_mean=light_field.time_delay_mean,
        raw_sensor_response=raw_sensor_response)

    mt = mask_time_in_point_cloud(
        pcl=cxcyt,
        start_time=start_time,
        end_time=end_time)

    lixel_ids_t = lixel_ids[mt]
    cxcyt_t = cxcyt[mt]

    mc = mask_cxcy_in_point_cloud(
        pcl=cxcyt_t,
        cx_center=cx_center,
        cy_center=cy_center,
        c_radius=c_radius)

    lixel_ids_tc = lixel_ids_t[mc]
    cxcyt_tc = cxcyt_t[mc]

    cl = PhotonStreamCluster(pcl=cxcyt_tc)
    m_as = cl.labels >= 0

    cxcyt_air_shower = cxcyt_tc[m_as]
    lixel_ids_air_shower = lixel_ids_tc[m_as]
    #psplot.point_cloud(cxcyt_air_shower)
    #plt.show()

    plt.hist2d(
        x=np.rad2deg(cxcyt_air_shower[:, 0]),
        y=np.rad2deg(cxcyt_air_shower[:, 1]),
        bins=np.rad2deg([
            np.linspace(cx_center - c_radius, cx_center + c_radius, 40),
            np.linspace(cy_center - c_radius, cy_center + c_radius, 40),
        ]))
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title('object distance {:0.1f}m'.format(object_distance))
    plt.savefig('./{:06d}.png'.format(i))