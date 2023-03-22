import numpy as np


def object_distance_2_image_distance(object_distance, focal_length):
    # 1/f = 1/g + 1/b
    return 1.0 / (1.0 / focal_length - 1.0 / object_distance)


def image_distance_2_object_distance(image_distance, focal_length):
    # 1/f = 1/g + 1/b
    return 1.0 / (1.0 / focal_length - 1.0 / image_distance)


def cxcyb2xyz(cx, cy, image_distance, focal_length):
    # 1/focal_length = 1/object_distance + 1/image_distance
    object_distance = image_distance_2_object_distance(
        image_distance=image_distance, focal_length=focal_length
    )
    x = np.tan(cx) * object_distance
    y = np.tan(cy) * object_distance
    return np.array([x, y, object_distance])


def xyz2cxcyb(x, y, z, focal_length):
    object_distance = z
    # 1/focal_length = 1/object_distance + 1/image_distance
    image_distance = object_distance_2_image_distance(
        object_distance=object_distance, focal_length=focal_length
    )
    cx = np.arctan(x / object_distance)
    cy = np.arctan(y / object_distance)
    return np.array([cx, cy, image_distance])


def resolution_of_depth(
    object_distance_m,
    focal_length_m,
    mirror_diameter_m,
    diameter_of_pixel_projected_on_sensor_plane_m,
):
    """
    Estimate and return the upper (g_p) and lower (g_m) object-distance which
    mark the range in object-distance where a telescope sees a sharp picture of
    when its focus is set to object_distance_m.

    reference
    ---------
    @article{bernlohr2013monte,
        author = {
            Bernlohr, K and Barnacka, A and Becherini, Yvonne and Bigas, O
            Blanch and Carmona, E and Colin, P and Decerprit, G and Di Pierro, F
            and Dubois, F and Farnier, Christian and others
        },
        journal = {Astroparticle Physics},
        pages = {171--188},
        publisher = {Elsevier},
        title = {
            {Monte} {Carlo} design studies for the {Cherenkov Telescope Array}
        },
        volume = {43},
        year = {2013},
    }
    """
    f = focal_length_m
    D = mirror_diameter_m
    p = diameter_of_pixel_projected_on_sensor_plane_m
    g = object_distance_m

    g_p = g * (1 + p * g / (2 * f * D))
    g_m = g * (1 - p * g / (2 * f * D))

    return g_p, g_m
