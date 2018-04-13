from skimage.measure import LineModelND, ransac
from mpl_toolkits.mplot3d import Axes3D
import plenopy as pl
import json
import os
import scipy
import shutil
from astroML.stats import fit_bivariate_normal
from matplotlib.patches import Ellipse

def plot_plane(a, b, c, d):
    xx, yy = np.mgrid[-40:40, -40:40]
    return xx, yy, (-d - a * xx - b * yy) / c


run = pl.Run('/home/sebastian/Desktop/past_trigger')

imrays = pl.image.ImageRays(run.light_field_geometry)

trigger_offsets = []
pap_time_offsets = []
energies = []
number_reconstructed_air_shower_photons = []
a = 0
for event in run:
    primary_momentum = event.simulation_truth.event.corsika_event_header.momentum()
    primary_direction = primary_momentum/np.linalg.norm(primary_momentum)

    core_cx = primary_direction[0]
    core_cy = primary_direction[1]
    core_x = event.simulation_truth.event.corsika_event_header.core_position_x_meter()
    core_y = event.simulation_truth.event.corsika_event_header.core_position_y_meter()
    energy = event.simulation_truth.event.corsika_event_header.total_energy_GeV
    energies.append(energy)

    trigger_response = pl.tomography.image_domain.image_domain_tomography.read_trigger_response(event)
    roi = pl.trigger.region_of_interest_from_trigger_response(
        trigger_response=trigger_response,
        time_slice_duration=event.light_field.time_slice_duration,
        pixel_pos_cx=event.light_field.pixel_pos_cx,
        pixel_pos_cy=event.light_field.pixel_pos_cy)

    trigger_offset = np.sqrt(
        (roi['cx_center_roi'] - core_cx)**2 +
        (roi['cy_center_roi'] - core_cy)**2)
    trigger_offsets.append(trigger_offset)


    (air_shower_photon_ids, lixel_ids
    ) = pl.photon_classification.classify_air_shower_photons_from_trigger_response(
            event=event,
            trigger_region_of_interest=roi)
    number_reconstructed_air_shower_photons.append(air_shower_photon_ids.shape[0])

    # Arrival times

    cxcyt, lixel_ids = pl.photon_stream.cython_reader.stream2_cx_cy_arrivaltime_point_cloud(
        photon_stream=event.raw_sensor_response.photon_stream,
        time_slice_duration=event.raw_sensor_response.time_slice_duration,
        NEXT_READOUT_CHANNEL_MARKER=event.raw_sensor_response.NEXT_READOUT_CHANNEL_MARKER,
        cx=event.light_field.cx_mean,
        cy=event.light_field.cy_mean,
        time_delay=event.light_field.time_delay_mean)

    arrival_times = cxcyt[:, 2][air_shower_photon_ids]

    # Array Images
    # ------------
    number_refocusses = 6
    object_distances = np.logspace(np.log10(7e3), np.log10(30e3), number_refocusses)
    cx_cy_radius_roi=np.deg2rad(1.0)
    cx_center_roi = roi['cx_center_roi']
    cy_center_roi = roi['cy_center_roi']

    number_telescopes = 6
    tel_az = np.linspace(0, 2*np.pi, number_telescopes, endpoint=False)
    tel_r = run.light_field_geometry.sensor_plane2imaging_system.expected_imaging_system_max_aperture_radius
    fov_diameter = run.light_field_geometry.sensor_plane2imaging_system.max_FoV_diameter
    pos_x = tel_r*np.cos(tel_az)
    pos_y = tel_r*np.sin(tel_az)


    fig, axs = plt.subplots(number_refocusses+1, number_telescopes, figsize=(28,28))

    if air_shower_photon_ids.shape[0] > 100:


        for i, object_distance in enumerate(object_distances):
            cx, cy = imrays.cx_cy_in_object_distance(object_distance)

            cx_air_shower = cx[lixel_ids[air_shower_photon_ids]]
            cy_air_shower = cy[lixel_ids[air_shower_photon_ids]]

            x = event.light_field.x_mean[lixel_ids[air_shower_photon_ids]]
            y = event.light_field.y_mean[lixel_ids[air_shower_photon_ids]]
            """
            for tel in range(number_telescopes):
                dist = np.sqrt((x - pos_x[tel])**2 + (y - pos_y[tel])**2)
                v = dist <= tel_r/1.5

                axs[i, tel].hist2d(
                    x=np.rad2deg(cx_air_shower[v]),
                    y=np.rad2deg(cy_air_shower[v]),
                    bins=np.rad2deg([
                        np.linspace(cx_center_roi - cx_cy_radius_roi, cx_center_roi + cx_cy_radius_roi, 80),
                        np.linspace(cy_center_roi - cx_cy_radius_roi, cy_center_roi + cx_cy_radius_roi, 80),
                    ]))

                if np.sum(v) >= 3:
                    (mu, sigma1, sigma2, alpha) = fit_bivariate_normal(
                        x=cx_air_shower[v],
                        y=cy_air_shower[v],
                        robust=False)

                    Nsig=5
                    E = Ellipse(
                        np.rad2deg(mu),
                        np.rad2deg(sigma1 * Nsig),
                        np.rad2deg(sigma2 * Nsig),
                        np.rad2deg(alpha),
                        edgecolor='r',
                        facecolor='none',
                        linestyle='dashed')
                    axs[i, tel].add_patch(E)


                    fov = Ellipse(
                        [0,0],
                        np.rad2deg(fov_diameter),
                        np.rad2deg(fov_diameter),
                        0.0,
                        edgecolor='white',
                        facecolor='none',)
                    axs[i, tel].add_patch(fov)


                axs[i, tel].set_title('i: {i:d}, tel:{tel:d}'.format(i=i, tel=tel))
                axs[i, tel].plot(
                    np.rad2deg(core_cx),
                    np.rad2deg(core_cy), 'xb')
                axs[i, tel].set_xlabel('cx/deg')
                axs[i, tel].set_ylabel('cy/deg')
                axs[i, tel].set_aspect('equal')

        for tel in range(number_telescopes):
            dist = np.sqrt((x - pos_x[tel])**2 + (y - pos_y[tel])**2)
            v = dist <= tel_r/1.5
            axs[i+1, tel].hist2d(
                x=x[v],
                y=y[v],
                bins=[
                    np.linspace(-200, 200, 80),
                    np.linspace(-200, 200, 80),
                ])
            axs[i+1, tel].plot(core_x, core_y, 'xb')
            axs[i+1, tel].set_xlabel('x/m')
            axs[i+1, tel].set_ylabel('y/m')
            axs[i+1, tel].set_aspect('equal')

            A = Ellipse(
                [0,0],
                tel_r*2,
                tel_r*2,
                0.0,
                edgecolor='white',
                facecolor='none',)
            axs[i+1, tel].add_patch(A)
        plt.savefig('{:09d}.png'.format(event.number))
        """
        plt.close('all')

        c0 = 3e8
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, c0*arrival_times, c='b', marker='o')

        sample = np.c_[x,y,c0*arrival_times]

        B, inlier = pl.tools.ransac_3d_plane.fit(
            xyz_point_cloud=sample,
            max_number_itarations=500,
            min_number_points_for_plane_fit=10,
            max_orthogonal_distance_of_inlier=0.025,)

        c_pap_time = np.array([B[0], B[1], B[2]])
        if c_pap_time[2] > 0:
            c_pap_time *= -1
        c_pap_time = c_pap_time/np.linalg.norm(c_pap_time)
        #print(B, np.sum(inlier)/sample.shape[0], np.linalg.norm(c_pap_time))
        pap_time_offset = np.sqrt(
            (c_pap_time[0] - core_cx)**2 +
            (c_pap_time[1] - core_cy)**2)
        pap_time_offsets.append(pap_time_offset)

        a, b, c, d = B
        xx, yy, zz = plot_plane(a, b, c, d)
        ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))

        print(
            '{o:.1f}deg, {p:.1f}deg, {e:.1f}GeV'.format(
                o=np.rad2deg(trigger_offset), p=np.rad2deg(pap_time_offset), e=energy))

        plt.figure()
        plt.plot([0.0, core_cx], [0.0, core_cy], 'b')
        plt.plot(core_cx, core_cy, 'ob')
        plt.plot([0.0, roi['cx_center_roi']], [0.0, roi['cy_center_roi']], 'r')
        plt.plot(roi['cx_center_roi'], roi['cy_center_roi'], 'or')
        plt.plot([0.0, c_pap_time[0]], [0.0, c_pap_time[1]], 'g')
        plt.plot(c_pap_time[0], c_pap_time[1], 'og')
        plt.gca().set_aspect('equal')
        plt.show()



