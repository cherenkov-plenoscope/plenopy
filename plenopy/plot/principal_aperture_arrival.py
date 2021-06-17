import numpy as np
import sebastians_matplotlib_addons as splt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import os
import tempfile

from .FigureSize import FigureSize
from ..tomography import Rays

SPEED_OF_LIGHT = 299792458


def add2ax(
    light_field_geometry,
    photon_lixel_ids,
    photon_arrival_times,
    ax,
    alpha=1.0,
    size=35.
):
    xs = light_field_geometry.x_mean[photon_lixel_ids]
    ys = light_field_geometry.y_mean[photon_lixel_ids]
    zs = photon_arrival_times*SPEED_OF_LIGHT
    return ax.scatter(xs, ys, zs, linewidth=0, s=size, alpha=alpha)


def save_principal_aperture_arrival_stack(
    light_field_geometry,
    photon_lixel_ids,
    photon_arrival_times,
    out_dir,
    elev=15,
    steps=7,
    alpha=0.3,
    size=35.,
    figure_size=None
):
    if figure_size is None:
        fsz = FigureSize(16, 9, pixel_rows=1080, dpi=240)
    else:
        fsz = figure_size

    plt.rcParams.update({'font.size': 12})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(fsz.width, fsz.hight))
    ax = fig.gca(projection='3d')

    time_start = np.min(photon_arrival_times)
    time_stop = np.max(photon_arrival_times)

    add2ax(
        light_field_geometry=light_field_geometry,
        photon_lixel_ids=photon_lixel_ids,
        photon_arrival_times=photon_arrival_times - time_start,
        ax=ax,
        alpha=alpha,
        size=size)

    aperture_radius = (
        light_field_geometry.expected_aperture_radius_of_imaging_system)
    p = Circle(
        (0, 0),
        aperture_radius,
        edgecolor='k',
        facecolor='none',
        linewidth=1.)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(
        p,
        z=0,
        zdir="z")

    ax.set_xlim(-aperture_radius, aperture_radius)
    ax.set_ylim(-aperture_radius, aperture_radius)
    ax.set_zlim(0, (time_stop - time_start)*SPEED_OF_LIGHT)
    ax.set_xlabel('x/m')
    ax.set_ylabel('y/m')
    ax.set_zlabel('z/m')

    azimuths = np.linspace(0., 360., steps, endpoint=False)
    for i, azimuth in enumerate(azimuths):
        ax.view_init(elev=elev, azim=azimuth)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig(
            os.path.join(out_dir, 'aperture3D_{:06d}.png'.format(i)),
            dpi=fsz.dpi)
    plt.close()


def save_principal_aperture_arrival_video(
    light_field_geometry,
    photon_lixel_ids,
    photon_arrival_times,
    output_path,
    elev=15,
    steps=73,
    frames_per_second=12,
    figure_size=None
):
    with tempfile.TemporaryDirectory(prefix='plenopy_video') as tmp:

        save_principal_aperture_arrival_stack(
            light_field_geometry=light_field_geometry,
            photon_lixel_ids=photon_lixel_ids,
            photon_arrival_times=photon_arrival_times,
            out_dir=tmp,
            elev=elev,
            steps=steps,
            figure_size=None)

        splt.write_video_from_image_slices(
            image_sequence_wildcard_path=os.path.join(tmp, 'aperture3D_%06d.png'),
            output_path=output_path,
            frames_per_second=frames_per_second)
