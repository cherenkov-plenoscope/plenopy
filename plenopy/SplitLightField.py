import numpy as np


class SplitLightField:
    """
    The light-field is split into image-sequences, each taken at a different
    principal-aperture-cell (paxel).
    In natural units, not in detector specific units.

    Fields
    ======

    image_sequences         A list of image-sequences.
                            Each image-sequence is an array of photons.
                            Each photon has three coordinates cx, cy and t.

    number_photons          Number of photons in the entire light-field.

    number_paxel            Number of principal-aperture-cells (paxel), thus
                            number of image-sequences.

    paxel_pos_x /
    paxel_pos_y             The positions of the paxels on the aperture-plane.

    median_cx /
    median_cy               The median direction of all photons in the
                            light-field.
    """

    def __init__(self, loph_record, light_field_geometry):
        """
        parameter
        =========

        loph_record             A light-field recorded by the plenoscope in
                                'List-of-Photons' (LOPH) representation.

        light_field_geometry    The plenoscope's light-firld-geometry.
        """
        lr = loph_record
        lfg = light_field_geometry
        self.number_photons = lr["photons"]["arrival_time_slices"].shape[0]

        self.number_paxel = lfg.number_paxel
        self.paxel_pos_x = lfg.paxel_pos_x
        self.paxel_pos_y = lfg.paxel_pos_y

        self.image_sequences = [[] for pax in range(self.number_paxel)]
        _, ph_paxel = lfg.pixel_and_paxel_of_lixel(
            lixel=lr["photons"]["channels"]
        )
        ph_cx = lfg.cx_mean[lr["photons"]["channels"]]
        ph_cy = lfg.cy_mean[lr["photons"]["channels"]]
        self.median_cx = np.median(ph_cx)
        self.median_cy = np.median(ph_cy)
        ph_t = (
            lr["sensor"]["time_slice_duration"]
            * lr["photons"]["arrival_time_slices"]
        )

        for ph in range(self.number_photons):
            pax = ph_paxel[ph]
            self.image_sequences[pax].append([ph_cx[ph], ph_cy[ph], ph_t[ph]])

        for pax in range(self.number_paxel):
            if len(self.image_sequences[pax]) > 0:
                self.image_sequences[pax] = np.array(self.image_sequences[pax])
            else:
                self.image_sequences[pax] = np.zeros(
                    shape=(0, 3),
                    dtype=np.float32
                )
