from ..light_field import sequence as lfs


def show(event):
    """
    Shows an overview figure of the Event.

    1)  Directional intensity distribution accross the field of view
        (the classic IACT image)

    2)  Positional intensity distribution on the principal aperture plane

    3)  The arrival time distribution of photo equivalents accross all
        lixels

    4)  The photo equivalent distribution accross all lixels
    """
    raw = event.raw_sensor_response
    lixel_sequence = event.lixel_sequence_raw()

    pix_img_seq = lfs.pixel_sequence(
        lixel_sequence=lixel_sequence,
        number_pixel=event.light_field_geometry.number_pixel,
        number_paxel=event.light_field_geometry.number_paxel)

    pax_img_seq = lfs.paxel_sequence(
        lixel_sequence=lixel_sequence,
        number_pixel=event.light_field_geometry.number_pixel,
        number_paxel=event.light_field_geometry.number_paxel)

    fig, axs = plt.subplots(2, 2)
    plt.suptitle(short_info(self))
    pix_int = lfs.integrate_around_arrival_peak(
        sequence=pix_img_seq,
        integration_radius=1)
    pixel_image = image.Image(
        pix_int['integral'],
        event.light_field_geometry.pixel_pos_cx,
        event.light_field_geometry.pixel_pos_cy)
    axs[0][0].set_title(
        'directional image at time slice '+str(pix_int['peak_slice']))
    image.plot.add_pixel_image_to_ax(pixel_image, axs[0][0])
    pax_int = lfs.integrate_around_arrival_peak(
        sequence=pax_img_seq,
        integration_radius=1)
    paxel_image = image.Image(
        pax_int['integral'],
        event.light_field_geometry.paxel_pos_x,
        event.light_field_geometry.paxel_pos_y)
    axs[0][1].set_title(
        'principal aperture at time slice '+str(pax_int['peak_slice']))
    image.plot.add_paxel_image_to_ax(paxel_image, axs[0][1])
    light_field.plot.add2ax_hist_arrival_time(
        sequence=lixel_sequence,
        time_slice_duration=raw.time_slice_duration,
        ax=axs[1][0])
    light_field.plot.add2ax_hist_intensity(lixel_sequence, axs[1][1])
    plt.show()
