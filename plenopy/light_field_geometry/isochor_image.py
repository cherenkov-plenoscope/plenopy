r"""
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
    cx_mean, cx_std, cy_mean, cy_std, x_mean, x_std, y_mean, y_std
):
    """
    The ray r(d) = (x,y,0)^T + d*(cx,cy,cz)^T has a point closest to the
    aperture's principal plane's origin (0,0,0)^T.
    The 'd' to reach this point on the ray r(d) is the path-length we are
    looking for.
    """
    d_mean = cx_mean * x_mean + cy_mean * y_mean

    # del d_mean / del cx = x
    # del d_mean / del cy = y
    # del d_mean / del x = cx
    # del d_mean / del y = cy

    d_std = np.sqrt(
        (x_mean**2) * (cx_std**2)
        + (y_mean**2) * (cy_std**2)
        + (cx_mean**2) * (x_std**2)
        + (cy_mean**2) * (y_std**2)
    )
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
    rel_dists_mean, rel_dists_std = relative_path_length_for_isochor_image(
        cx_mean=cx_mean,
        cx_std=cx_std,
        cy_mean=cy_mean,
        cy_std=cy_std,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )
    time_delay_principal_aperture_plane_2_image = (
        rel_dists_mean / speed_of_light
    )
    time_delay_sesnsor_2_image = (
        time_delay_from_sensors_to_principal_aperture_plane
        - time_delay_principal_aperture_plane_2_image
    )
    return time_delay_sesnsor_2_image
