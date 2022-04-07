import numpy as np


def object_distance_2_image_distance(object_distance, focal_length):
    # 1/f = 1/g + 1/b
    return 1.0/(1.0/focal_length - 1.0/object_distance)


def image_distance_2_object_distance(image_distance, focal_length):
    # 1/f = 1/g + 1/b
    return 1.0/(1.0/focal_length - 1.0/image_distance)


def cxcyb2xyz(cx, cy, image_distance, focal_length):
    # 1/focal_length = 1/object_distance + 1/image_distance
    object_distance = image_distance_2_object_distance(
        image_distance=image_distance,
        focal_length=focal_length)
    x = np.tan(cx)*object_distance
    y = np.tan(cy)*object_distance
    return np.array([x, y, object_distance])


def xyz2cxcyb(x, y, z, focal_length):
    object_distance = z
    # 1/focal_length = 1/object_distance + 1/image_distance
    image_distance = object_distance_2_image_distance(
        object_distance=object_distance,
        focal_length=focal_length)
    cx = np.arctan(x/object_distance)
    cy = np.arctan(y/object_distance)
    return np.array([cx, cy, image_distance])
