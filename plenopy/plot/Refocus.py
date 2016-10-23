import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import os
import tempfile
import shutil
from .ObjectDistanceRuler import add2ax_object_distance_ruler
from .FigSize import FigSize
from .images2video import images2video
from ..plot import Image as plt_Image


def save_refocus_stack(
    event, 
    output_path, 
    obj_dist_min=2e3, 
    obj_dist_max=12e3, 
    steps=10, 
    use_absolute_scale=True,
    image_prefix='refocus_'):

    plt.rcParams.update({'font.size': 12})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    object_distances = np.logspace(
        np.log10(obj_dist_min),
        np.log10(obj_dist_max),
        steps)

    images = [event.light_field.refocus(
        object_distance) for object_distance in tqdm(object_distances)]
    intensities = [i.intensity for i in images]
    if use_absolute_scale:
        vmin = np.array(intensities).min()
        vmax = np.array(intensities).max()
    else:
        vmin, vmax = None, None

    fig_size = FigSize(dpi=200)

    fig = plt.figure(figsize=(fig_size.width, fig_size.hight))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 6])
    ax_ruler = plt.subplot(gs[0])
    ax_image = plt.subplot(gs[1])

    for i in tqdm(range(len(images))):
        plt.suptitle(event.simulation_truth.event.short_event_info())

        add2ax_object_distance_ruler(
            ax=ax_ruler,
            object_distance=object_distances[i],
            object_distance_min=obj_dist_min,
            object_distance_max=obj_dist_max)
        
        ax_image.set_aspect('equal')
        plt_Image.add_pixel_image_to_ax(images[i], ax_image, vmin=vmin, vmax=vmax)

        plt.savefig(
            os.path.join(output_path, image_prefix+str(i).zfill(6)+'.jpg'),
            dpi=fig_size.dpi)
        ax_ruler.clear()
        ax_image.clear()    
    plt.close(fig)


def save_refocus_video(
    event, 
    output_path, 
    obj_dist_min=2e3, 
    obj_dist_max=12e3, 
    steps=25, 
    fps=25, 
    use_absolute_scale=True):

    with tempfile.TemporaryDirectory() as work_dir:
        image_prefix = 'refocus_'
        save_refocus_stack(
            event,
            obj_dist_min=obj_dist_min,
            obj_dist_max=obj_dist_max,
            steps=steps,
            path=work_dir,
            use_absolute_scale=use_absolute_scale,
            image_prefix=image_prefix)

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