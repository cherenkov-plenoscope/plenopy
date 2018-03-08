import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import os
import tempfile
import subprocess
import tqdm
from ..tools import FigureSize
from ..tomography import Rays
from . import images2video

c_vacuum = 3e8

def save_principal_aperture_arrival_stack(
    light_field,
    out_dir,
    steps=7,
    threshold=1,
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

    rays = Rays.from_light_field_geometry(light_field)
    intensity_max = light_field.sequence.max()

    for t in tqdm.tqdm(range(light_field.number_time_slices)):
        for i in range(light_field.number_lixel):
            intensity = light_field.sequence[t,i]
            if intensity > threshold:
                distance = c_vacuum*t*light_field.time_slice_duration
                pos = rays.support[i] + rays.direction[i]*distance
                I = intensity/intensity_max
                ax.scatter(pos[0], pos[1], pos[2], lw=0, s=35., alpha=I**2)

    aperture_radius = light_field.expected_aperture_radius_of_imaging_system
    p = Circle((0, 0), aperture_radius, edgecolor='k', facecolor='none', lw=1.)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

    ax.set_xlim(-aperture_radius, aperture_radius)
    ax.set_ylim(-aperture_radius, aperture_radius)
    ax.set_zlim(0,
        c_vacuum*light_field.number_time_slices*light_field.time_slice_duration)
    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_zlabel('Z/m')

    azimuths = np.linspace(0., 360., steps, endpoint=False)
    for i, azimuth in enumerate(azimuths):
        ax.view_init(elev=15., azim=azimuth)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig(
            os.path.join(out_dir, 'aperture3D_'+str(i).zfill(6) + ".png"),
            dpi=fsz.dpi)
    plt.close()


def save_principal_aperture_arrival_video(
    light_field,
    output_path,
    steps=73,
    threshold=1,
    frames_per_second=12,
    figure_size=None
):
    with tempfile.TemporaryDirectory(prefix='plenopy_video') as work_dir:

        save_principal_aperture_arrival_stack(
            light_field=light_field,
            steps=steps,
            threshold=threshold,
            out_dir=work_dir,
            figure_size=figure_size)

        images2video.images2video(
            image_path=os.path.join(work_dir, 'aperture3D_%06d.png'),
            output_path=output_path,
            frames_per_second=frames_per_second)
