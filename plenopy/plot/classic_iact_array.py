import numpy as np
import matplotlib.pyplot as plt
from .. import image
from ..light_field import sequence
from ..image import Image

def classic_view(
    event,
    output_path='.',
    time_slice_integration_radius=3,
    dpi=400,
    scale=5.0,
    l_margin=0.05,
    r_margin=0.05,
    t_margin=0.05,
    b_margin=0.05,
    colormap='viridis',
    add_fov_circles=False,
    fov_circles_line_width=1.0
):
    fov_radius = 0.5*event.light_field.sensor_plane2imaging_system.max_FoV_diameter

    light_field = sequence.integrate_around_arrival_peak(
        event.light_field.sequence, 
        integration_radius=time_slice_integration_radius
    )['integral']

    array_images = light_field.reshape(
        (event.light_field.number_pixel, event.light_field.number_paxel)
    )
    max_intensity = array_images.max()
    min_intensity = array_images.min()

    iact_img_h = 1.0*scale
    iact_img_w = 1.0*scale

    paxels_on_diagonal = event.light_field.sensor_plane2imaging_system.number_of_paxel_on_pixel_diagonal

    ap_x_width = iact_img_w*paxels_on_diagonal
    ap_y_width = iact_img_h*paxels_on_diagonal

    x = -event.light_field.lixel_positions_x[0:event.light_field.number_paxel]
    y = -event.light_field.lixel_positions_y[0:event.light_field.number_paxel]
    # its mirrored
    x -= x.mean()
    y -= y.mean()
    x -= x.min()
    y -= y.min()
    x_width = x.max()-x.min()
    y_width = y.max()-y.min()
    width = np.max([x_width, y_width])
    rel_x = x/width
    rel_y = y/width

    ap_x_pos = ap_x_width*rel_x + iact_img_w/2
    ap_y_pos = ap_y_width*rel_y + iact_img_h/2 + 1/np.sqrt(3)*iact_img_h/2

    fig_w = l_margin + ap_x_width + iact_img_w + r_margin
    fig_h = t_margin + ap_y_width + iact_img_h + b_margin
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    for i in range(number_paxel):

        l_anchor = l_margin + ap_x_pos[i] - iact_img_w/2
        b_anchor = b_margin + ap_y_pos[i] - iact_img_h/2

        ax_image = fig.add_axes(
            (
                l_anchor/fig_w, 
                b_anchor/fig_h, 
                iact_img_w/fig_w, 
                iact_img_h/fig_h
            )
        )

        ax_image.axis('off')
        ax_image.set_aspect('equal')

        patch_collection = image.plot.add2ax(
            ax=ax_image, 
            I=array_images[:,i], 
            px=np.rad2deg(event.light_field.pixel_pos_cx), 
            py=np.rad2deg(event.light_field.pixel_pos_cy), 
            colormap=colormap, 
            hexrotation=30, 
            vmin=min_intensity, 
            vmax=max_intensity, 
            colorbar=False,
        )

        if add_fov_circles:
            fov_d = event.light_field.sensor_plane2imaging_system.max_FoV_diameter
            fov_limit = plt.Circle(
                (0, 0), 
                np.rad2deg(fov_radius), 
                edgecolor='k', 
                lw=fov_circles_line_width, 
                facecolor='none',
                clip_on=False
            )
            ax_image.add_artist(fov_limit)


    plt.savefig(output_path, dpi=dpi)
    return fig