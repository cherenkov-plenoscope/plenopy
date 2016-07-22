#!/usr/bin/env python
from __future__ import absolute_import, print_function, division
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import subprocess

from PlotTools import plot_pixel
from PlotTools import plot_paxel


def save_pixel_sum_stereo_image_sequence(evt, path, steps=25):
    aperture_radius = evt.light_field.lixel_statistics.expected_aperture_radius_of_imaging_system

    r = aperture_radius / 3.0
    phis = np.linspace(0.0, 2 * np.pi, steps)
    l_imgs = []
    r_imgs = []
    for i, phi in enumerate(phis):
        l_imgs.append(evt.light_field.pixel_sum_sub_aperture(
            sub_x=r * np.cos(phi) - r, sub_y=r * np.sin(phi), sub_r=r, smear=0.25 * r
        )
        )
        r_imgs.append(evt.light_field.pixel_sum_sub_aperture(
            sub_x=r * np.cos(phi) + r, sub_y=r * np.sin(phi), sub_r=r, smear=0.25 * r
        )
        )

    I_min = 9.9e9
    I_max = 0.0
    for i in range(len(l_imgs)):
        I_min_ = np.min([l_imgs[i].intensity.min(), r_imgs[i].intensity.min()])
        if I_min_ < I_min:
            I_min = I_min_

        I_max_ = np.max([l_imgs[i].intensity.max(), r_imgs[i].intensity.max()])
        if I_max_ > I_max:
            I_max = I_max_

    plt.rcParams.update({'font.size': 20})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for i in range(len(l_imgs)):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 9))
        ax_l.set_title('left eye')
        ax_r.set_title('right eye')
        plt.suptitle(evt.mc_truth.short_event_info())

        plot_pixel(l_imgs[i], ax_l, vmin=I_min, vmax=I_max)
        plot_pixel(r_imgs[i], ax_r, vmin=I_min, vmax=I_max)

        out = os.path.join(path, str(i).zfill(6) + '.png')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.savefig(out, dpi=120)
        plt.close()


def save_pixel_sum_stereo_video(evt, path, steps=25, fps=25):
    with tempfile.TemporaryDirectory() as work_dir:
        save_dual_stereo_stack(
            evt,
            path=work_dir,
            steps=steps
        )

        avconv_command = [
            'avconv',
            '-y',  # force overwriting of existing output file
            '-framerate', str(int(fps)),  # Frames per second
            '-f', 'image2',
            '-i', os.path.join(work_dir, '%06d.png'),
            '-c:v', 'h264',
            #'-s', '1260x1080', # sample images down to FullHD 1080p
            '-crf', '23',  # high quality 0 (best) to 53 (worst)
            '-crf_max', '25',  # worst quality allowed
            os.path.splitext(path)[0] + '.mp4'
        ]
        subprocess.call(avconv_command)
