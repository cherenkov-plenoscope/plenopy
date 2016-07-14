#!/usr/bin/env python
from __future__ import absolute_import, print_function, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import gridspec
import os
import tempfile
import subprocess
import shutil

from .PlotTools import plot_pixel

def save_refocus_stack(event, path, obj_dist_min=2e3, obj_dist_max=12e3, steps=10, use_absolute_scale=True):

    plt.rcParams.update({'font.size': 12})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    object_distances = np.logspace(
        np.log10(obj_dist_min),
        np.log10(obj_dist_max),
        steps
    )

    images = [event.light_field.refocus(object_distance) for object_distance in object_distances]
    intensities = [i.intensity for i in images]
    if use_absolute_scale:
        vmin=np.array(intensities).min()
        vmax=np.array(intensities).max()
    else:
        vmin, vmax = None, None

    for i in range(len(images)):
        image = images[i]
        object_distance = object_distances[i]

        fig = plt.figure(figsize=(7, 6)) 
        fig, (ax_dir, ax_pap) = plt.subplots(1,2)
        plt.suptitle(event.mc_truth.short_event_info())

        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 6]) 
        ax0 = plt.subplot(gs[0])
        ax0.set_xlim([0, 1])
        ax0.set_ylim([0, obj_dist_max/1e3])
        ax0.yaxis.tick_left()
        ax0.set_ylabel('object distance/km')
        ax0.spines['right'].set_visible(False)
        ax0.spines['top'].set_visible(False)
        ax0.spines['bottom'].set_visible(False)
        ax0.xaxis.set_visible(False)
        ax0.plot([0, .5], [object_distance/1e3, object_distance/1e3], linewidth=5.0)
        ax0.text(0.0, -1.0, format(object_distance/1e3, '.2f')+r'\,km')

        ax1 = plt.subplot(gs[1])
        ax1.set_aspect('equal')  

        plot_pixel(image, ax1, vmin=vmin, vmax=vmax)
        plt.savefig(os.path.join(path, 'refocus_'+str(i).zfill(6)+".png"), dpi=180)
        plt.close()

def save_refocus_video(event, path, obj_dist_min=2e3, obj_dist_max=12e3, steps=25, fps=25, use_absolute_scale=True):
    with tempfile.TemporaryDirectory() as work_dir:
        save_refocus_stack(
            event,
            obj_dist_min=obj_dist_min,
            obj_dist_max=obj_dist_max,
            steps=steps,
            path=work_dir,
            use_absolute_scale=use_absolute_scale)

        #duplicate the images and use them again in reverse order
        i=0
        for o in range(5):
            shutil.copy(
                os.path.join(work_dir, 'refocus_'+str(0).zfill(6)+".png"), 
                os.path.join(work_dir, 'video_'+str(i).zfill(6)+".png"))
            i+=1

        for o in range(steps):
            shutil.copy(
                os.path.join(work_dir, 'refocus_'+str(o).zfill(6)+".png"), 
                os.path.join(work_dir, 'video_'+str(i).zfill(6)+".png"))
            i+=1

        for o in range(5):
            shutil.copy(
                os.path.join(work_dir, 'refocus_'+str(steps-1).zfill(6)+".png"), 
                os.path.join(work_dir, 'video_'+str(i).zfill(6)+".png"))
            i+=1

        for o in range(steps-1, -1, -1):
            shutil.copy(
                os.path.join(work_dir, 'refocus_'+str(o).zfill(6)+".png"), 
                os.path.join(work_dir, 'video_'+str(i).zfill(6)+".png"))
            i+=1

        avconv_command = [
                'avconv',
                '-y', # force overwriting of existing output file 
                '-framerate', str(int(fps)), # Frames per second
                '-f', 'image2', 
                '-i', os.path.join(work_dir, 'video_%06d.png'), 
                '-c:v', 'h264',
                #'-s', '1260x1080', # sample images down to FullHD 1080p
                '-crf', '23', # high quality 0 (best) to 53 (worst)
                '-crf_max', '25', # worst quality allowed
                os.path.splitext(path)[0]+'.mp4'
            ]
        subprocess.call(avconv_command)