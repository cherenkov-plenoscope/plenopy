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
    image_prefix='slice_',
    intensity_volume_2=None):

    fig_size = FigSize(dpi=200)
    fig = plt.figure(
        figsize=(fig_size.width, fig_size.hight), 
        dpi=fig_size.dpi)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3]) 
    ax_object_distance_ruler = plt.subplot(gs[0])
    ax_histogram = plt.subplot(gs[1])

    intensity_max = intensity_volume.max()
    intensity_min = intensity_volume.min()

    if intensity_volume_2 is not None:
        intensity_max_2 = intensity_volume_2.max()
        intensity_min_2 = intensity_volume_2.min()
    else:
        intensity_max_2 = None
        intensity_min_2 = None

    for z_slice in range(binning.number_z_bins):

        fig.suptitle(event.simulation_truth.event.short_event_info())

        add2ax_z_slice(
            ax=ax_histogram,
            z_slice=z_slice,
            binning=binning, 
            intensity_volume=intensity_volume,
            intensity_min=intensity_min,
            intensity_max=intensity_max,
            intensity_volume_2=intensity_volume_2,
            intensity_min_2=intensity_min_2,
            intensity_max_2=intensity_max_2,
            )

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
    intensity_max=None,
    intensity_volume_2=None,
    intensity_min_2=None, 
    intensity_max_2=None):
    
    xy_lim = binning.xy_bin_centers.max()
    ax.set_xlabel('x/m')
    ax.set_ylabel('y/m')

    if intensity_volume_2 is None:
        img = ax.imshow(
            intensity_volume[:,:,z_slice], 
            cmap='viridis', 
            extent=[-xy_lim, xy_lim, -xy_lim, xy_lim], 
            interpolation='None')
        img.set_clim(intensity_min, intensity_max)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img, cax=cax)
    else:
        image = matrix_2_rgb_image(
            intensity_volume[:,:,z_slice], 
            color_channel=2,
            intensity_min=intensity_min,
            intensity_max=intensity_max) + matrix_2_rgb_image(
            intensity_volume_2[:,:,z_slice], 
            color_channel=0,
            intensity_min=intensity_min_2,
            intensity_max=intensity_max_2) 
        img = ax.imshow(
            image, 
            extent=[-xy_lim, xy_lim, -xy_lim, xy_lim], 
            interpolation='None')


def matrix_2_rgb_image(
    matrix,
    color_channel=0,
    intensity_min=None, 
    intensity_max=None):

    if intensity_min is None:
        intensity_min = matrix.min()

    if intensity_max is None:
        intensity_max = matrix.max()

    image = np.zeros(shape=(matrix.shape[0], matrix.shape[1], 3))

    inensity = (matrix - intensity_min)/(intensity_max - intensity_min)
    image[:,:,color_channel] = inensity
    return image


def save_slice_video(
    event, 
    binning, 
    intensity_volume, 
    output_path,
    fps=25,
    intensity_volume_2=None):

    with tempfile.TemporaryDirectory() as work_dir:
        image_prefix = 'slice_'
        save_slice_stack(
            event=event,
            binning=binning,
            intensity_volume=intensity_volume,
            output_path=work_dir,
            image_prefix=image_prefix,
            intensity_volume_2=intensity_volume_2)
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


def add2ax_flat(ax, xyzIs, color='b', alpha_max=1.0, steps=32, ball_size=50.0):
    intensities = xyzIs[:,3]
    xyzIs_sorted = xyzIs[np.argsort(intensities)]

    chunk_suize = np.ceil(xyzIs.shape[0]/steps)

    for step, alpha in enumerate(np.linspace(0.0025, alpha_max, steps)):

        start = step*chunk_suize
        end = start+chunk_suize
        if end >= xyzIs.shape[0]:
            end = xyzIs.shape[0]-1

        ax.scatter(
            xyzIs_sorted[start:end,0], xyzIs_sorted[start:end,1], xyzIs_sorted[start:end,2],
            s=ball_size,
            depthshade=False,
            alpha=alpha,
            c=color,
            lw=0)           


def plot_flat(xyzIs, xyzIs2=None, alpha_max=1.0, ball_size=25, steps=32):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    add2ax_flat(ax, xyzIs, color='b', steps=steps, alpha_max=alpha_max,ball_size=ball_size)

    if xyzIs2 is not None:
        add2ax_flat(ax, xyzIs2, color='r', steps=steps, alpha_max=alpha_max,ball_size=ball_size)

    plt.show()