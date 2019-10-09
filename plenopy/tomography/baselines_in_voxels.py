import numpy as np
from .Rays import Rays
from .narrow_angle.deconvolution import make_cached_tomographic_system_matrix


def baselines_in_voxels(
    light_field_geometry,
    x_bin_edges,
    y_bin_edges,
    z_bin_edges,
):
    '''
    Estimate the 3D reconstruction power of the Plenoscope by counting the
    number of possible 3D reconstruction baselines between different principal
    aperture support cells (paxels) that participate to a single volume cell
    (voxel).

    The number of baselines n_b is nb = (n_s**2 - n_s)/2, where n_s is the
    number of different support positions on the aperture plane.

    Parameter
    ---------

    light_field_geometry    The geometry of the light field captured by
                            plenoscope.

    x,y,z_bin_edges         The edges of the volume cells (voxels).
    '''
    n_x_bins = x_bin_edges.shape[0]-1
    n_y_bins = y_bin_edges.shape[0]-1
    n_z_bins = z_bin_edges.shape[0]-1

    rays = Rays.from_light_field_geometry(light_field_geometry)

    system_matrix = make_cached_tomographic_system_matrix(
        supports=-rays.support,
        directions=rays.direction,
        x_bin_edges=x_bin_edges,
        y_bin_edges=y_bin_edges,
        z_bin_edges=z_bin_edges,
    )

    n_paxels_in_voxel = np.zeros(n_x_bins*n_y_bins*n_z_bins)
    for paxel in range(light_field_geometry.number_paxel):
        rays_in_this_paxel = np.zeros(
            light_field_geometry.number_paxel,
            dtype=np.bool
        )
        rays_in_this_paxel[paxel] = True
        rays_in_this_paxel = np.tile(
            rays_in_this_paxel,
            light_field_geometry.number_pixel
        )

        paxel_system_matrix_integral = system_matrix[
            :, rays_in_this_paxel
        ].sum(axis=1)
        paxel_system_matrix_integral = np.array(
            paxel_system_matrix_integral
        ).reshape((paxel_system_matrix_integral.shape[0],))
        n_paxels_in_voxel += paxel_system_matrix_integral > 0

    number_baselines_in_voxel = (
        n_paxels_in_voxel**2 -
        n_paxels_in_voxel
    )/2.0

    return number_baselines_in_voxel.reshape(
        (n_x_bins, n_y_bins, n_z_bins,),
        order='C'
    )
