import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import os
import tempfile
import subprocess

from PlotTools import FigureSize


class FigureSize():

    def __init__(self, cols, rows, dpi):
        self.cols = cols
        self.rows = rows
        self.dpi = dpi
        self.hight = self.rows / self.dpi
        self.width = self.cols / self.dpi


def save_principal_aperture_arrival_stack(lf, path, steps=7, threshold=100):

    plt.rcParams.update({'font.size': 12})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fsz = FigureSize(1920, 1080, dpi=240)
    fig = plt.figure(figsize=(fsz.width, fsz.hight))
    ax = fig.gca(projection='3d')

    above_threshold = lf.I > threshold

    min_t = lf.t[above_threshold].min()
    dur_t = lf.t[above_threshold].max() - min_t
    max_I = lf.I[above_threshold].max()

    for pax in range(lf.lixel_statistics.number_paxel):
        for pix in range(lf.lixel_statistics.number_pixel):
            if above_threshold[pix, pax]:
                d = (lf.t[pix, pax] - min_t)

                xpix = lf.lixel_statistics.pixel_pos_cx[pix]
                ypix = lf.lixel_statistics.pixel_pos_cy[pix]

                incident = np.array([
                    xpix,
                    ypix,
                    np.sqrt(1. - xpix * xpix - ypix * ypix)
                ])

                x = np.array([
                    lf.lixel_statistics.paxel_pos_x[pax],
                    lf.lixel_statistics.paxel_pos_x[pax] + d * incident[0]
                ])
                y = np.array([
                    lf.lixel_statistics.paxel_pos_y[pax],
                    lf.lixel_statistics.paxel_pos_y[pax] + d * incident[1]
                ])
                z = np.array([
                    0.,
                    d * incident[2]
                ])
                #ax.plot(x, y, z, 'b')
                I = lf.I[pix, pax] / max_I
                ax.scatter(x[1], y[1], z[1], lw=0, s=35., alpha=I**2)

    aperture_radius = lf.lixel_statistics.expected_aperture_radius_of_imaging_system
    p = Circle((0, 0), aperture_radius, edgecolor='k', facecolor='none', lw=1.)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

    ax.set_xlim(-aperture_radius, aperture_radius)
    ax.set_ylim(-aperture_radius, aperture_radius)
    ax.set_zlim(0, dur_t)
    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_zlabel('t/s')

    azimuths = np.linspace(0., 360., steps, endpoint=False)
    for i, azimuth in enumerate(azimuths):
        ax.view_init(elev=5., azim=azimuth)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig(os.path.join(path, 'aperture3D_' +
                                 str(i).zfill(6) + ".png"), dpi=fsz.dpi)

    plt.close()


def save_principal_aperture_arrival_video(lf, path, steps=73, threshold=100, fps=12):
    with tempfile.TemporaryDirectory() as work_dir:
        save_principal_aperture_arrival_stack(
            lf,
            steps=steps,
            threshold=threshold,
            path=work_dir)
        avconv_command = [
            'avconv',
            '-y',  # force overwriting of existing output file
            '-framerate', str(int(fps)),  # Frames per second
            '-f', 'image2',
            '-i', os.path.join(work_dir, 'aperture3D_%06d.png'),
            '-c:v', 'h264',
            '-s', '1920x1080',  # sample images down to FullHD 1080p
            '-crf', '23',  # high quality 0 (best) to 53 (worst)
            '-crf_max', '25',  # worst quality allowed
            os.path.splitext(path)[0] + '.mp4'
        ]
        subprocess.call(avconv_command)
