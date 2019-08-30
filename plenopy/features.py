import numpy as np
from collections import namedtuple
from sklearn.cluster import DBSCAN
from . import trigger
from . import tools
from .event.Event import Event
from .light_field_geometry import LightFieldGeometry
import matplotlib.pyplot as plt


def estimate_nearest_neighbors(x, y, epsilon, itself=False):
    nn = trigger.neighborhood(x, y, epsilon, itself=itself)
    ns = []
    num_points = len(x)
    for i in range(num_points):
        ns.append(np.arange(num_points)[nn[i, :]])
    return ns


def islands_greater_equal_threshold(intensities, neighborhood, threshold):
    num_points = len(intensities)
    intensities = np.array(intensities)
    abv_thr = set(np.arange(num_points)[intensities >= threshold])
    to_do = abv_thr.copy()
    islands = []
    while len(to_do) > 0:
        i = next(iter(to_do))
        to_do.remove(i)
        island = [i]
        i_neighbors = set(neighborhood[i])
        candidates = to_do.intersection(i_neighbors)
        while len(candidates) > 0:
            c = next(iter(candidates))
            if c in abv_thr:
                island.append(c)
                candidates.remove(c)
                to_do.remove(c)
                c_neighbors = set(neighborhood[c])
                c_neighbors_to_do = to_do.intersection(c_neighbors)
                candidates = candidates.union(c_neighbors_to_do)
        islands.append(island)
    return islands


def estimate_light_front_surface_normal(
    xs,
    ys,
    zs,
    max_number_itarations=100,
    min_number_points_for_plane_fit=10,
    max_orthogonal_distance_of_inlier=0.025):
    B, inlier = tools.ransac_3d_plane.fit(
        xyz_point_cloud=np.c_[xs, ys, zs],
        max_number_itarations=max_number_itarations,
        min_number_points_for_plane_fit=min_number_points_for_plane_fit,
        max_orthogonal_distance_of_inlier=max_orthogonal_distance_of_inlier,)
    c_pap_time = np.array([B[0], B[1], B[2]])
    if c_pap_time[2] > 0:
        c_pap_time *= -1
    c_pap_time = c_pap_time/np.linalg.norm(c_pap_time)
    return c_pap_time


HillasEllipse = namedtuple(
    'HillasEllipse', [
    'cx_mean',
    'cy_mean',
    'cx_major',
    'cy_major',
    'cx_minor',
    'cy_minor',
    'std_major',
    'std_minor'])


def estimate_hillas_ellipse(cxs, cys):
    cx_mean = np.mean(cxs)
    cy_mean = np.mean(cys)
    cov_matrix = np.cov(np.c_[cxs, cys].T)
    eigen_vals, eigen_vecs= np.linalg.eig(cov_matrix)
    major_idx = np.argmax(eigen_vals)
    if major_idx == 0:
        minor_idx = 1
    else:
        minor_idx = 0
    major_axis = eigen_vecs[:, major_idx]
    major_std = np.sqrt(eigen_vals[major_idx])
    minor_axis = eigen_vecs[:, minor_idx]
    minor_std = np.sqrt(eigen_vals[minor_idx])
    return HillasEllipse(
        cx_mean=cx_mean,
        cy_mean=cy_mean,
        cx_major=major_axis[0],
        cy_major=major_axis[1],
        cx_minor=minor_axis[0],
        cy_minor=minor_axis[1],
        std_major=major_std,
        std_minor=minor_std)


def interpolate_x(x0, x1, y0, y1, yarg):
    # f(x) = m*x + b
    # m = (y1 - y0)/(x1 - x0)
    # y1 = m*x1 + b
    m = (y1 - y0)/(x1 - x0)
    b = -m*x1 + y1
    # (y - b)/m = x
    x = (yarg - b)/m
    return x


def extract_features(
    cherenkov_photons,
    light_field_geometry,
    light_field_geometry_addon,
    debug=False):
    cp = cherenkov_photons
    lfg = light_field_geometry
    lfg_addon = light_field_geometry_addon

    f = {}
    f['num_photons'] = int(cp.number)
    if f['num_photons'] < 35:
        msg = "Expected at least {:d} photons, got {:d}.".format(
                35,
                f['num_photons'])
        raise RuntimeError(msg)
    #=======================
    # aperture
    #=======================
    if debug: print("aperture")
    paxelidx_bin_edges = np.arange(lfg.number_paxel + 1)
    paxel_intensity = np.histogram(
        lfg.paxel_pos_tree.query(
        np.c_[cp.x, cp.y])[1],
        bins=paxelidx_bin_edges)[0]

    f['paxel_intensity_peakness_std_over_mean'] = \
        float(np.std(paxel_intensity)/np.mean(paxel_intensity))

    f['paxel_intensity_peakness_max_over_mean'] = \
        float(np.max(paxel_intensity)/np.mean(paxel_intensity))

    # position
    # ----------
    f['paxel_intensity_median_x'] = float(np.median(cp.x))
    f['paxel_intensity_median_y'] = float(np.median(cp.y))

    if debug: print("aperture, num islands")
    # num islands on aperture
    #------------------------
    expected_paxel_intensity = np.sum(paxel_intensity)/lfg.number_paxel
    for thr in [2, 4, 8]:
        key = 'aperture_num_islands_watershed_rel_thr_{:d}'.format(thr)
        islands = islands_greater_equal_threshold(
            intensities=paxel_intensity,
            neighborhood=lfg_addon["paxel_neighborhood"],
            threshold=thr*expected_paxel_intensity)
        f[key] = len(islands)

    #=======================
    # light-front
    #=======================
    if debug: print("light-front")
    light_front_normal = estimate_light_front_surface_normal(
        xs=cp.x,
        ys=cp.y,
        zs=cp.t_pap*3e8)
    f['light_front_cx'] = light_front_normal[0]
    f['light_front_cy'] = light_front_normal[1]

    #=======================
    # image
    #=======================
    if debug: print("image, focus infinity")
    f['image_infinity_cx_mean'] = float(np.mean(cp.cx))
    f['image_infinity_cy_mean'] = float(np.mean(cp.cy))
    f['image_infinity_cx_std'] = float(np.std(cp.cx))
    f['image_infinity_cy_std'] = float(np.std(cp.cy))
    leakage_mask = np.hypot(cp.cx, cp.cy) >= lfg_addon["fov_radius_leakage"]
    f['image_infinity_num_photons_on_edge_field_of_view'] = int(
        np.sum(leakage_mask))


    # quick refocus scan
    #-------------------
    if debug: print("refocus-stack")
    def ellipse_solid_angle(object_distance, cherenkov_photons):
        obj = object_distance
        cp = cherenkov_photons
        cxs, cys = cp.cx_cy_in_object_distance(obj)
        ellipse = estimate_hillas_ellipse(cxs=cxs, cys=cys)
        return np.pi*ellipse.std_major*ellipse.std_minor

    def ellipse_solid_angle_slope(
        object_distance,
        cherenkov_photons,
        relative_delta=0.05):
        obj_upper = object_distance *(1 + relative_delta)
        obj_lower = object_distance *(1 - relative_delta)
        obj_delta = obj_upper - obj_lower
        esa_upper = ellipse_solid_angle(obj_upper, cherenkov_photons)
        esa_lower = ellipse_solid_angle(obj_lower, cherenkov_photons)
        esa_delta = esa_upper - esa_lower
        return esa_delta/obj_delta

    if debug: print("refocus-stack, sharpest obj-dist")
    NUM_REFOCUS_SLICES = 24
    OBJECT_DISTANCE_MIN = 2.5e3
    OBJECT_DISTANCE_MAX = 250e3
    OBJECT_DISTANCES = np.geomspace(
        OBJECT_DISTANCE_MIN,
        OBJECT_DISTANCE_MAX,
        NUM_REFOCUS_SLICES)
    ellipse_solid_angles = np.zeros(NUM_REFOCUS_SLICES)
    for i, obj in enumerate(OBJECT_DISTANCES):
        ellipse_solid_angles[i] = ellipse_solid_angle(
            object_distance=obj,
            cherenkov_photons=cp)
    obj = OBJECT_DISTANCES[np.argmin(ellipse_solid_angles)]
    obj_delta = 1e3
    for i in range(10):
        sa_plus = ellipse_solid_angle(
            object_distance=obj + obj_delta,
            cherenkov_photons=cp)
        sa_minus = ellipse_solid_angle(
            object_distance=obj - obj_delta,
            cherenkov_photons=cp)
        if sa_plus < sa_minus:
            obj += obj_delta
        else:
            obj -= obj_delta
        obj_delta /= 2
    f['image_smallest_ellipse_object_distance'] = float(obj)
    f['image_smallest_ellipse_solid_angle'] = float(sa_plus)

    # half-depth min
    sa_twice = 2.*f['image_smallest_ellipse_solid_angle']
    for i in range(NUM_REFOCUS_SLICES):
        if ellipse_solid_angles[i] <= sa_twice:
            if i == 0:
                obj_lower = OBJECT_DISTANCE_MIN
            else:
                obj_lower = interpolate_x(
                    x0=OBJECT_DISTANCES[i - 1],
                    x1=OBJECT_DISTANCES[i],
                    y0=ellipse_solid_angles[i - 1],
                    y1=ellipse_solid_angles[i],
                    yarg=sa_twice)
            break
    for i in range(NUM_REFOCUS_SLICES):
        idx = NUM_REFOCUS_SLICES - i - 1
        if ellipse_solid_angles[idx] <= sa_twice:
            if idx == NUM_REFOCUS_SLICES - 1:
                obj_upper = OBJECT_DISTANCE_MAX
            else:
                obj_upper = interpolate_x(
                    x0=OBJECT_DISTANCES[idx],
                    x1=OBJECT_DISTANCES[idx + 1],
                    y0=ellipse_solid_angles[idx],
                    y1=ellipse_solid_angles[idx + 1],
                    yarg=sa_twice)
            break

    if obj_lower == OBJECT_DISTANCE_MIN:
        raise RuntimeError("Lower focus out of range.")
    """
    if obj_upper == OBJECT_DISTANCE_MAX:
        plt.figure()
        plt.plot(OBJECT_DISTANCES, ellipse_solid_angles)
        plt.plot(obj, sa_plus, 'ro')
        plt.plot([obj_lower, obj_upper], [sa_twice, sa_twice], 'g')
        plt.ylim([0, 1e-3])
        plt.show()
        plt.close("all")
        raise RuntimeError("Upper focus out of range.")
    """

    f['image_smallest_ellipse_half_depth'] = float(obj_upper - obj_lower)

    # shift of mean in image while refocusing
    cxs_u, cys_u = cp.cx_cy_in_object_distance(obj_upper)
    cxs_l, cys_l = cp.cx_cy_in_object_distance(obj_lower)
    f['image_half_depth_shift_cx'] = float(np.median(cxs_u) - np.median(cxs_l))
    f['image_half_depth_shift_cy'] = float(np.median(cys_u) - np.median(cys_l))

    if debug: print('image_smallest_ellipse_half_depth', f['image_smallest_ellipse_half_depth'])

    """
    plt.figure()
    plt.plot(OBJECT_DISTANCES, ellipse_solid_angles)
    plt.plot(obj, sa_plus, 'ro')
    plt.plot([obj_lower, obj_upper], [sa_twice, sa_twice], 'g')
    plt.ylim([0, 1e-3])
    plt.show()
    plt.close("all")
    """

    cx, cy = cp.cx_cy_in_object_distance(
        f['image_smallest_ellipse_object_distance'])
    leakage_mask = np.hypot(cx, cy) >= lfg_addon["fov_radius_leakage"]
    f['image_smallest_ellipse_num_photons_on_edge_field_of_view'] = int(
        np.sum(leakage_mask))

    """
    plt.plot(cx, cy, 'x')
    plt.gca().set_aspect('equal')
    plt.show()
    """

    if debug: print("image_smallest_ellipse, num islands")
    epsilon_cx_cy_radius = np.deg2rad(0.05)
    min_number_photons = 17
    dbscan = DBSCAN(
        eps=epsilon_cx_cy_radius,
        min_samples=min_number_photons
    ).fit(np.c_[cx, cy])

    if debug: print("image_smallest_ellipse, num islands 2")
    label_ids = np.array(list(set(dbscan.labels_)))
    f['image_num_islands'] = int(np.sum(label_ids >= 0))
    cluster_masks = []
    cluster_sizes = []
    cluster_ids = label_ids[label_ids >= 0]
    for cluster_id in cluster_ids:
        cluster_masks.append(dbscan.labels_ == cluster_id)
        cluster_sizes.append(np.sum(cluster_masks[-1]))

    """
    for label in label_ids:
        mask = dbscan.labels_ == label
        plt.plot(cx[mask], cy[mask], 'x')
    plt.gca().set_aspect('equal')
    plt.show()
    """
    return f


def extract_features_from_events(
    event_paths,
    light_field_geometry_path,
):
    lfg = LightFieldGeometry(light_field_geometry_path)

    lfg_addon = {}
    lfg_addon["paxel_radius"] = \
        lfg.sensor_plane2imaging_system.\
            expected_imaging_system_max_aperture_radius/\
        lfg.sensor_plane2imaging_system.number_of_paxel_on_pixel_diagonal
    lfg_addon["nearest_neighbor_paxel_enclosure_radius"] = \
        3*lfg_addon["paxel_radius"]
    lfg_addon["paxel_neighborhood"] = estimate_nearest_neighbors(
        x=lfg.paxel_pos_x,
        y=lfg.paxel_pos_y,
        epsilon=lfg_addon["nearest_neighbor_paxel_enclosure_radius"])
    lfg_addon["fov_radius"] = \
        .5*lfg.sensor_plane2imaging_system.max_FoV_diameter
    lfg_addon["fov_radius_leakage"] = 0.9*lfg_addon["fov_radius"]
    lfg_addon["num_pixel_on_diagonal"] = \
        np.floor(2*np.sqrt(lfg.number_pixel/np.pi))

    features = []
    for event_path in event_paths:
        event = Event(event_path, light_field_geometry=lfg)

        run_id = event.simulation_truth.event.corsika_run_header.number
        event_id = np.mod(event.number, 1000000)

        try:
            cp = event.cherenkov_photons
            if cp is None:
                raise RuntimeError("No Cherenkov-photons classified yet.")
            f = extract_features(
                cherenkov_photons=cp,
                light_field_geometry=lfg,
                light_field_geometry_addon=lfg_addon)
            f["run"] = int(run_id)
            f["event"] = int(event_id)
            features.append(f)
        except Exception as e:
            print("Run {:d}, Event: {:d} :".format(run_id, event_id), e)
    return features;
