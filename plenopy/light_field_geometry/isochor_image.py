"""
Isochorus-imaging

On the ideal imaging-system an isochorus fron of light coming from
direction (cx, cy) shall arrive in the pixel-cx-cy in the same moment
of time. However, on large aperture telescopes this is not the case for
pixels which are in the outer regions of the field-of-view. Using e.g.
paraboic imaging-mirrors, the central pixels in the field-of-view do
fulfill the demand for isochorus arrivals of fronts of light. But not the
outer pixels.

The plenoscope can overcome this limitation and implement giant apertures
without the spread of the arrival-times in the pixels in the outer regions
of the field-of-view.

        A              B
        |              |
       -|--------------|- isochorus front of light from (cx=0, cy=0)
        |              |  The plane of the light-front is orthogonal to the
        |              |  optical-axis.
        |      .       |  Photons A and B arrive in the pixel in the same
        |      |       |  moment in time.
        |      .       |
        |  [sensors]   |
        |      /\      |
        |     /. \     |
        |    / |  \    |
        |   /  .   \   |
        |  /   |    \  |
        | /    .     \ |
        |/     |      \|
    ------------------------ principal-aperture-plan
               |
               .
               optical-axis


            A
            |              B
           -|---________   |
            |           ---|- isochorus front of light from (cx!=0, or cy!=0)
           |              |   The plane of the light-front is inclined relative
           |   .          |   to the optical-axis.
           |   |          |   The photons A and B arrive in the pixel at
          |    .         |    different moments in time.
          |[sensors]     |    The way of A is longer than the way of B.
          | |\ |         |
         | |  \|        |
         | |   |\       |
         ||    . \      |
        | |    |   \   |
        ||     .    \  |
        ||     |      \|
    ------------------------ principal-aperture-plan
               |
               .
               optical-axis

"""
import numpy as np


def relative_path_length_for_isochor_image(
    cx_mean,
    cx_std,
    cy_mean,
    cy_std,
    x_mean,
    x_std,
    y_mean,
    y_std
):
    """
    # Making plane models [cx, cy, cz, d]
    # normal-vector: [cx, cy, cz]
    # distance of plane to origin: d
    cz = np.sqrt(1.0 - cx**2 - cy**2)
    plane_models = np.c_[cx, cy, cz, np.zeros(number_lixels)]

    # Support-positions of the rays of the light-field (lixels) on the
    # principal-aperture-palne (pap).
    lixel_supports_on_pap = np.c_[
        x,
        y,
        np.zeros(number_lixels),
        np.ones(number_lixels)]

    # The relative distances between the support-positions of the lixels and
    # the plane of isochorusly incoming photons.
    relative_path_lengths = np.zeros(number_lixels)
    for lixel in range(number_lixels):
        relative_distances[lixel] = (
            plane_models[lixel].dot(lixel_supports_on_pap[lixel]))
    """

    # It turns out that the relative-path-lengths can be computed like this:
    d_mean = cx_mean*x_mean + cy_mean*y_mean

    # der(d_mean)/der(cx) = x
    # der(d_mean)/der(cy) = y
    # der(d_mean)/der(x) = cx
    # der(d_mean)/der(y) = cy

    d_std = np.sqrt(
        (x_mean**2)*(cx_std**2) +
        (y_mean**2)*(cy_std**2) +
        (cx_mean**2)*(x_std**2) +
        (cy_mean**2)*(y_std**2))
    return d_mean, d_std


def time_delay_from_sensors_to_image(
    time_delay_from_sensors_to_principal_aperture_plane,
    cx_mean,
    cx_std,
    cy_mean,
    cy_std,
    x_mean,
    x_std,
    y_mean,
    y_std,
    speed_of_light=299792458,
):
    rel_dists_mean, rel_dists_std = (
        estimate_relative_path_length_and_uncertainty_for_isochor_image(
            cx_mean=cx_mean,
            cx_std=cx_std,
            cy_mean=cy_mean,
            cy_std=cy_std,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,))

    time_delay_principal_aperture_plane_2_image = rel_dists_mean/speed_of_light

    time_delay_sesnsor_2_image = (
        time_delay_from_sensors_to_principal_aperture_plane -
        time_delay_principal_aperture_plane_2_image)

    return time_delay_sesnsor_2_image