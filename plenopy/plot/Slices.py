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

def save_slice_stack(event, path, obj_dist_min=2e3, obj_dist_max=12e3, steps=10, use_absolute_scale=True, restrict_to_plenoscope_aperture=True):

    plt.rcParams.update({'font.size': 12})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    object_distances = np.logspace(
        np.log10(obj_dist_min),
        np.log10(obj_dist_max),
        steps
    )

    n_lixel = event.light_field.valid_lixel.sum()
    xs = np.zeros([steps, n_lixel])
    ys = np.zeros([steps, n_lixel])
    intensity = event.light_field.intensity.flatten()[event.light_field.valid_lixel.flatten()]

    for i, object_distance in enumerate(object_distances):
        pos_xy = event.light_field.rays.slice_intersections_in_object_distance(
            object_distance)
        xs[i,:] = pos_xy[:,0][event.light_field.valid_lixel.flatten()]
        ys[i,:] = pos_xy[:,1][event.light_field.valid_lixel.flatten()]

    if restrict_to_plenoscope_aperture:
        r = 5.0*event.light_field.expected_aperture_radius_of_imaging_system
        xmax = r
        xmin = -r
        ymax = r
        ymin = -r      
    else:
        xmax = xs.max()
        xmin = xs.min()
        ymax = ys.max()
        ymin = ys.min()

    n_bins = int(0.25*np.sqrt(n_lixel))

    bins = np.zeros([steps, n_bins, n_bins])
    xedges = np.zeros(n_bins+1)
    yedges = np.zeros(n_bins+1)
    for i in range(steps):
        bins[i,:,:], xedges, yedges = np.histogram2d(
            x=xs[i],
            y=ys[i], 
            weights=intensity,
            bins=[n_bins, n_bins],
            range=[[xmin, xmax], [ymin, ymax]])

    Imax = bins.max()

    for i in range(len(xs)):
        object_distance = object_distances[i]

        fig = plt.figure(figsize=(7, 6)) 
        fig, (ax0, ax1) = plt.subplots(1,2)
        plt.suptitle(event.simulation_truth.short_event_info())

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
        im = ax1.imshow(
            bins[i,:,:], 
            vmin=0.0,
            vmax=Imax,
            interpolation='none', 
            origin='low', 
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
            aspect='equal')    
        im.set_cmap('viridis')
        ax1.set_xlabel('x/m')  
        ax1.set_ylabel('y/m')

        plt.savefig(os.path.join(path, 'refocus_'+str(i).zfill(6)+".png"), dpi=180)
        plt.close()

def save_slices_video(event, path, obj_dist_min=2e3, obj_dist_max=12e3, steps=25, fps=25, use_absolute_scale=True):
    with tempfile.TemporaryDirectory() as work_dir:
        save_slice_stack(
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