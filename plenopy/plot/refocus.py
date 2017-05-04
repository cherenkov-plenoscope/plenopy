import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import os
import tempfile
import shutil
from ..tools import add2ax_object_distance_ruler
from ..tools import FigureSize
from .images2video import images2video
from .. import image
from ..image import Image
from ..image import ImageRays
from ..image.plot import add_pixel_image_to_ax
from ..light_field import sequence


def refocus_images(
    light_field, 
    object_distances, 
    tims_slice_start, 
    time_slice_end):

    image_rays = ImageRays(light_field)
    images = []    
    for object_distance in object_distances:

        lisel2pixel = image_rays.pixel_ids_of_lixels_in_object_distance(
            object_distance)

        raw_img_sequence = light_field.pixel_sequence_refocus(lisel2pixel)
        raw_img = raw_img_sequence[tims_slice_start:time_slice_end].sum(axis=0)

        img = Image(
            raw_img, 
            light_field.pixel_pos_cx,
            light_field.pixel_pos_cy)
        images.append(img)
    return images


def save_side_by_side(
    event,
    object_distances, 
    output_path,
    tims_slice_range,
    cx_range=None,
    cy_range=None,
    use_absolute_scale=True,
    dpi=250,
    pixel_rows_per_image=1920):

    fig_size = FigureSize(
        dpi=dpi, 
        pixel_rows=pixel_rows_per_image*len(object_distances),
        relative_hight=1*len(object_distances),
        relative_width=1)
    fig = plt.figure(figsize=(fig_size.width, fig_size.hight))
    N = 12
    gs = gridspec.GridSpec(N*len(object_distances)+1, N//2)
    images = refocus_images(
        light_field=event.light_field, 
        object_distances=object_distances, 
        tims_slice_start=tims_slice_range[0], 
        time_slice_end=tims_slice_range[1],)

    intensities = [i.intensity for i in images]

    if use_absolute_scale:
        vmin = np.array(intensities).min()
        vmax = np.array(intensities).max()
    else:
        vmin, vmax = None, None

    for i, img in enumerate(images):
        st = i*N
        en = (i+1)*N

        ax_ruler = plt.subplot(gs[st:en,0])
        ax_image = plt.subplot(gs[st:en,1:])

        add2ax_object_distance_ruler(
            ax=ax_ruler,
            object_distance=object_distances[i],
            object_distance_min=np.min(object_distances),
            object_distance_max=np.max(object_distances),
            label='',
            print_value=False)
        
        ax_ruler.set_aspect('equal')
        ax_image.set_aspect('equal')
        patch_collection = add_pixel_image_to_ax(
            images[i], 
            ax_image, 
            vmin=vmin, 
            vmax=vmax, 
            colorbar=False)

    plt.colorbar(
        patch_collection, 
        cax=plt.subplot(gs[en,1:]), 
        orientation='horizontal',)

    obj_dist_label_ax = plt.subplot(gs[en,0])
    obj_dist_label_ax.axis('off')                                     
    obj_dist_label_ax.text(0.0, 0.0, 'object\ndistance\nkm')

    plt.tight_layout()
    plt.savefig(output_path, dpi=fig_size.dpi)
    return fig

def save_refocus_stack(
    event, 
    output_path, 
    obj_dist_min=2e3, 
    obj_dist_max=27e3,
    time_slices_window_radius=1,
    steps=16, 
    use_absolute_scale=True,
    image_prefix='refocus_'):

    plt.rcParams.update({'font.size': 12})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    object_distances = np.logspace(
        np.log10(obj_dist_min),
        np.log10(obj_dist_max),
        steps)

    pix_img_seq = event.light_field.pixel_sequence()
    time_max = sequence.time_slice_with_max_intensity(pix_img_seq)
    time_start = np.max([time_max-time_slices_window_radius, 0])
    time_end = np.min([time_max+time_slices_window_radius, pix_img_seq.shape[0]-1])

    images = refocus_images(
        light_field=event.light_field, 
        object_distances=object_distances,
        tims_slice_start=time_start,
        time_slice_end=time_end)

    intensities = [i.intensity for i in images]
    
    if use_absolute_scale:
        vmin = np.array(intensities).min()
        vmax = np.array(intensities).max()
    else:
        vmin, vmax = None, None

    fig_size = FigureSize(dpi=200)

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
        add_pixel_image_to_ax(images[i], ax_image, vmin=vmin, vmax=vmax)

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
    obj_dist_max=27e3, 
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
            output_path=work_dir,
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