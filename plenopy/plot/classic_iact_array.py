import numpy as np
import matplotlib.pyplot as plt
from . import image
from .. import sequence


def classic_view(
    light_field_geometry,
    photon_lixel_ids,
    output_path='.',
    dpi=100,
    iact_image_diagonal=5.0,
    l_margin=0.05,
    r_margin=0.05,
    t_margin=0.05,
    b_margin=0.05,
    colormap='viridis',
    add_fov_circles=False,
    fov_circles_line_width=1.0,
    s=1.0
):
    iact_image_fov_radius = (
        0.5*light_field_geometry.sensor_plane2imaging_system.max_FoV_diameter)

    array_images = np.zeros(
        (light_field_geometry.number_pixel, light_field_geometry.number_paxel),
        dtype=np.int64)

    for lix in photon_lixel_ids:
        pix, pax = light_field_geometry.pixel_and_paxel_of_lixel(lix)
        array_images[pix, pax] += 1

    max_intensity = array_images.max()
    min_intensity = array_images.min()

    iact_img_h = 1.0*iact_image_diagonal
    iact_img_w = 1.0*iact_image_diagonal

    paxels_on_diagonal = (
        light_field_geometry.sensor_plane2imaging_system.number_of_paxel_on_pixel_diagonal)

    ap_x_width = iact_img_w*paxels_on_diagonal
    ap_y_width = iact_img_h*paxels_on_diagonal

    x = -light_field_geometry.lixel_positions_x[
        0: light_field_geometry.number_paxel]
    y = -light_field_geometry.lixel_positions_y[
        0: light_field_geometry.number_paxel]
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

    for i in range(light_field_geometry.number_paxel):
        l_anchor = l_margin + ap_x_pos[i] - (s*iact_img_w)/2
        b_anchor = b_margin + ap_y_pos[i] - (s*iact_img_h)/2

        ax_image = fig.add_axes(
            (
                l_anchor/fig_w,
                b_anchor/fig_h,
                (s*iact_img_w)/fig_w,
                (s*iact_img_h)/fig_h
            )
        )

        ax_image.axis('off')
        ax_image.set_aspect('equal')

        patch_collection = image.add2ax(
            ax=ax_image,
            I=array_images[:, i],
            px=np.rad2deg(light_field_geometry.pixel_pos_cx),
            py=np.rad2deg(light_field_geometry.pixel_pos_cy),
            colormap=colormap,
            hexrotation=30,
            vmin=min_intensity,
            vmax=max_intensity,
            colorbar=False,)

        if add_fov_circles:
            fov_d = light_field_geometry.sensor_plane2imaging_system.max_FoV_diameter
            fov_limit = plt.Circle(
                (0, 0),
                np.rad2deg(iact_image_fov_radius),
                edgecolor='k',
                lw=fov_circles_line_width,
                facecolor='none',
                clip_on=False)
            ax_image.add_artist(fov_limit)

    plt.savefig(output_path, dpi=dpi)
    return fig
