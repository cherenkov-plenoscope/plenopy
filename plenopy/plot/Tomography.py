import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import tempfile
import shutil
from .FigSize import FigSize
from .ObjectDistanceRuler import add2ax_object_distance_ruler
from .images2video import images2video

def save_slice_stack(
    event, 
    binning, 
    intensity_volume, 
    output_path, 
    image_prefix='slice_'):

    fig_size = FigSize(dpi=200)
    fig = plt.figure(
        figsize=(fig_size.width, fig_size.hight), 
        dpi=fig_size.dpi)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3]) 
    ax_object_distance_ruler = plt.subplot(gs[0])
    ax_histogram = plt.subplot(gs[1])

    intensity_max = intensity_volume.max()
    intensity_min = intensity_volume.min()
    for z_slice in range(binning.number_z_bins):

        fig.suptitle(event.simulation_truth.event.short_event_info())

        add2ax_z_slice(
            ax=ax_histogram,
            z_slice=z_slice,
            binning=binning, 
            intensity_volume=intensity_volume,
            intensity_min=intensity_min,
            intensity_max=intensity_max)

        add2ax_object_distance_ruler(
            ax=ax_object_distance_ruler,
            object_distance=binning.z_bin_centers[z_slice],
            object_distance_min=binning.z_min,
            object_distance_max=binning.z_max)

        plt.savefig(
            os.path.join(output_path, image_prefix+str(z_slice).zfill(6)+".jpg"), 
            dpi=fig_size.dpi)

        ax_object_distance_ruler.clear()
        ax_histogram.clear()
    plt.close(fig)


def add2ax_z_slice(
    ax, 
    z_slice, 
    binning, 
    intensity_volume,
    intensity_min=None, 
    intensity_max=None):

    xy_lim = binning.xy_bin_centers.max()
    img = ax.imshow(
        intensity_volume[:,:,z_slice], 
        cmap='viridis', 
        extent=[-xy_lim, xy_lim, -xy_lim, xy_lim], 
        interpolation='None')
    img.set_clim(intensity_min, intensity_max)
    ax.set_xlabel('x/m')
    ax.set_ylabel('y/m')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)


def save_slice_video(
    event, 
    binning, 
    intensity_volume, 
    output_path,
    fps=25):

    with tempfile.TemporaryDirectory() as work_dir:
        image_prefix = 'slice_'
        save_slice_stack(
            event=event,
            binning=binning,
            intensity_volume=intensity_volume,
            output_path=work_dir,
            image_prefix=image_prefix)
        steps=binning.number_z_bins

        # duplicate the images and use them again in reverse order
        i = 0
        for o in range(5):
            shutil.copy(
                os.path.join(work_dir, image_prefix+str(0).zfill(6)+'.jpg'),
                os.path.join(work_dir, 'video_'+str(i).zfill(6)+'.jpg'))
            i += 1

        for o in range(steps):
            shutil.copy(
                os.path.join(work_dir, image_prefix+str(o).zfill(6)+'.jpg'),
                os.path.join(work_dir, 'video_'+str(i).zfill(6)+'.jpg'))
            i += 1

        for o in range(5):
            shutil.copy(
                os.path.join(work_dir, image_prefix+str(steps-1).zfill(6)+'.jpg'),
                os.path.join(work_dir, 'video_'+str(i).zfill(6)+'.jpg'))
            i += 1

        for o in range(steps - 1, -1, -1):
            shutil.copy(
                os.path.join(work_dir, image_prefix+str(o).zfill(6)+'.jpg'),
                os.path.join(work_dir, 'video_'+str(i).zfill(6)+'.jpg'))
            i += 1

        images2video(
            image_path=os.path.join(work_dir, 'video_%06d.jpg'),
            output_path=output_path,
            frames_per_second=fps)