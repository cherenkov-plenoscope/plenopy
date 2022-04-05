import numpy as np
from .. import system_matrix


def init(sparse_system_matrix):
    sysmatcsr = system_matrix.to_numpy_csr_matrix(sparse_system_matrix)
    psf = {}
    psf["csr"] = sysmatcsr
    # Total length of ray
    psf["image_ray_integral"] = psf["csr"].sum(axis=0).T

    # Total distance of all rays in this voxel
    psf["voxel_integral"] = psf["csr"].sum(axis=1)

    # The sum of the length of all rays hiting this voxel weighted with the
    # overlap of the ray and this voxel
    voxel_cross_psf = psf["csr"].dot(psf["image_ray_integral"])
    voxel_cross_psf = np.array(voxel_cross_psf).reshape(
        (voxel_cross_psf.shape[0],)
    )

    image_ray_cross_psf = psf["csr"].T.dot(psf["voxel_integral"])
    image_ray_cross_psf = np.array(image_ray_cross_psf).reshape(
        (image_ray_cross_psf.shape[0],)
    )

    psf["voxel_cross_psf"] = voxel_cross_psf
    psf["image_ray_cross_psf"] = image_ray_cross_psf
    return psf
