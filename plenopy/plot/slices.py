import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import tempfile
import shutil
from ..tools import FigureSize
from ..tools import add2ax_object_distance_ruler
from ..plot.images2video import images2video

def save_slice_stack(
    intensity_volume,
    event_info_repr,
    xy_extent,
    z_bin_centers,
    output_path,
    image_prefix='slice_',
    intensity_volume_2=None,
    xlabel='x/m',
    ylabel='y/m',
    sqrt_intensity=False,
):
    fig_size = FigureSize(dpi=200)
    fig = plt.figure(
        figsize=(fig_size.width, fig_size.hight),
        dpi=fig_size.dpi
    )
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])
    ax_object_distance_ruler = plt.subplot(gs[0])
    ax_histogram = plt.subplot(gs[1])


    vol_I_1 = intensity_volume.copy()
    if sqrt_intensity:
        vol_I_1 = np.sqrt(vol_I_1)

    I_vol_max1 = vol_I_1.max()
    I_vol_min1 = vol_I_1.min()

    if intensity_volume_2 is not None:
        vol_I_2 = intensity_volume_2.copy()
        if sqrt_intensity:
            vol_I_2 = np.sqrt(vol_I_2)

        I_vol_max2 = vol_I_2.max()
        I_vol_min2 = vol_I_2.min()
    else:
        I_vol_max2 = None
        I_vol_min2 = None

    for z_slice in range(vol_I_1.shape[2]):

        fig.suptitle(event_info_repr)

        ax_histogram.set_xlabel(xlabel)
        ax_histogram.set_ylabel(ylabel)

        image_2 = None
        if intensity_volume_2 is not None:
            image_2 = vol_I_2[:,:,z_slice]

        add2ax_image(
            ax=ax_histogram,
            xy_extent=xy_extent,
            image=vol_I_1[:,:,z_slice],
            I_vol_min1=I_vol_min1,
            I_vol_max1=I_vol_max1,
            image_2=image_2,
            I_vol_min2=I_vol_min2,
            I_vol_max2=I_vol_max2,
        )

        add2ax_object_distance_ruler(
            ax=ax_object_distance_ruler,
            object_distance=z_bin_centers[z_slice],
            object_distance_min=z_bin_centers.min(),
            object_distance_max=z_bin_centers.max(),
        )

        plt.savefig(
            os.path.join(
                output_path,
                image_prefix+str(z_slice).zfill(6)+".jpg"
            ),
            dpi=fig_size.dpi
        )

        ax_object_distance_ruler.clear()
        ax_histogram.clear()
    plt.close(fig)


def add2ax_image(
    ax,
    image,
    I_vol_min1=None,
    I_vol_max1=None,
    image_2=None,
    I_vol_min2=None,
    I_vol_max2=None,
    xy_extent=[-500,500,-500,500],
    cmap='viridis'
):
    if image_2 is None:
        img = ax.imshow(
            image,
            cmap=cmap,
            extent=xy_extent,
            interpolation='None'
        )
        img.set_clim(I_vol_min1, I_vol_max1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img, cax=cax)
    else:
        rgb_img1 = matrix_2_rgb_image(
            image,
            color_channel=1,
            I_vol_min1=I_vol_min1,
            I_vol_max1=I_vol_max1
        )
        rgb_img2 = matrix_2_rgb_image(
            image_2,
            color_channel=0,
            I_vol_min1=I_vol_min2,
            I_vol_max1=I_vol_max2
        )
        rgb_image = rgb_img1 + rgb_img2
        img = ax.imshow(
            rgb_image,
            extent=xy_extent,
            interpolation='None'
        )


def matrix_2_rgb_image(
    matrix,
    color_channel=0,
    I_vol_min1=None,
    I_vol_max1=None
):
    if I_vol_min1 is None:
        I_vol_min1 = matrix.min()
    if I_vol_max1 is None:
        I_vol_max1 = matrix.max()
    image = np.zeros(shape=(matrix.shape[0], matrix.shape[1], 3))
    inensity = (matrix - I_vol_min1)/(I_vol_max1 - I_vol_min1)
    image[:,:,color_channel] = inensity
    return image


def save_slice_video(
    event_info_repr,
    xy_extent,
    z_bin_centers,
    binning,
    intensity_volume,
    output_path,
    fps=25,
    intensity_volume_2=None,
    xlabel='x/m',
    ylabel='y/m',
):
    with tempfile.TemporaryDirectory() as work_dir:
        save_slice_stack(
            event_info_repr=event_info_repr,
            xy_extent=xy_extent,
            z_bin_centers=z_bin_centers,
            intensity_volume=intensity_volume,
            output_path=work_dir,
            intensity_volume_2=intensity_volume_2,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        steps = z_bin_centers.shape[0]

        # duplicate the images and use them again in reverse order
        i = 0
        for o in range(5):
            shutil.copy(
                os.path.join(work_dir, image_prefix+str(0).zfill(6)+'.jpg'),
                os.path.join(work_dir, 'video_'+str(i).zfill(6)+'.jpg')
            )
            i += 1

        for o in range(steps):
            shutil.copy(
                os.path.join(work_dir, image_prefix+str(o).zfill(6)+'.jpg'),
                os.path.join(work_dir, 'video_'+str(i).zfill(6)+'.jpg')
            )
            i += 1

        for o in range(5):
            shutil.copy(
                os.path.join(work_dir, image_prefix+str(steps-1).zfill(6)+'.jpg'),
                os.path.join(work_dir, 'video_'+str(i).zfill(6)+'.jpg')
            )
            i += 1

        for o in range(steps - 1, -1, -1):
            shutil.copy(
                os.path.join(work_dir, image_prefix+str(o).zfill(6)+'.jpg'),
                os.path.join(work_dir, 'video_'+str(i).zfill(6)+'.jpg')
            )
            i += 1

        images2video(
            image_path=os.path.join(work_dir, 'video_%06d.jpg'),
            output_path=output_path,
            frames_per_second=fps
        )
