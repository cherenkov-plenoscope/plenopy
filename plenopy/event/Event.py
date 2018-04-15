import numpy as np
import os
from os import path as op
import matplotlib.pyplot as plt
from . import plenoscope_event_header as peh
from ..RawLightFieldSensorResponse import RawLightFieldSensorResponse
from ..photon_stream import cython_reader as phs
from ..tools import HeaderRepresentation as hr
from ..light_field import sequence as lfs
from .. import image
from .. import corsika
from .. import simulation_truth
from .. import light_field


class Event(object):
    """
    Atmospheric Cherenkov Plenoscope (ACP) event.

    number                      The ID of this event in its run.

    type                        Event-type indicator (string)
                                "simulation" in case this event was simulated
                                "observation" in case this event was observed

    trigger_type                Indicates the trigger-mechanism.

    light_field_geometry        The light-field-geometry when this event was
                                recorded.

    simulation_truth            If type == "simulation"
                                Additional 'true' information known from the
                                simulation itself.

    raw_sensor_response         The raw response of the light-field-sensor
                                of the plenoscope.
    """
    def __init__(self, path, light_field_geometry):
        """
        Parameter
        ---------
        path                    The path of this event. Typically inside a Run
                                directory.

        light_field_geometry    The light-field-geometry to calibrate the raw
                                sensor-response of this event.
        """
        self.light_field_geometry = light_field_geometry
        self._path = op.abspath(path)
        self._read_event_header()
        self.raw_sensor_response = RawLightFieldSensorResponse(
            op.join(self._path, 'raw_light_field_sensor_response.phs'))
        if self.type == 'SIMULATION':
            self._read_simulation_truth()
        self.number = int(op.basename(self._path))

    def _read_event_header(self):
        header_path = op.join(self._path, 'event_header.bin')
        header = hr.read_float32_header(header_path)
        self.type = peh.event_type(header)
        self.trigger_type = peh.trigger_type(header)

    def _read_simulation_truth(self):
        truth_path = op.join(self._path, 'simulation_truth')
        if (self.trigger_type ==
            'EXTERNAL_TRIGGER_BASED_ON_AIR_SHOWER_SIMULATION_TRUTH'):
            simulation_truth_event = simulation_truth.Event(
                evth=corsika.EventHeader(
                    op.join(truth_path, 'corsika_event_header.bin')),
                runh=corsika.RunHeader(
                    op.join(truth_path, 'corsika_run_header.bin')))

            try:
                air_shower_photon_bunches = corsika.PhotonBunches(
                    op.join(truth_path, 'air_shower_photon_bunches.bin'))
            except(FileNotFoundError):
                air_shower_photon_bunches = None

            try:
                simulation_truth_detector = simulation_truth.Detector(
                    op.join(truth_path, 'detector_pulse_origins.bin'))
            except(FileNotFoundError):
                simulation_truth_detector = None

            self.simulation_truth = simulation_truth.SimulationTruth(
                event=simulation_truth_event,
                air_shower_photon_bunches=air_shower_photon_bunches,
                detector=simulation_truth_detector)

    def __repr__(self):
        out = self.__class__.__name__
        out += "("
        out += "number " + str(self.number) + ", "
        out += "type '" + self.type
        out += "')"
        return out

    def _plot_suptitle(self):
        if self.type == "SIMULATION":
            if (self.trigger_type ==
                "EXTERNAL_TRIGGER_BASED_ON_AIR_SHOWER_SIMULATION_TRUTH"):
                return self.simulation_truth.event.short_event_info()
            elif self.trigger_type == "EXTERNAL_RANDOM_TRIGGER":
                return 'Extrenal random trigger, no air shower'
            else:
                return 'Simulation, but trigger type is unknown: '+str(
                    self.trigger_type)
        elif self.type == "OBSERVATION":
            return 'Observation'
        else:
            return 'unknown event type: '+str(self.type)

    def _light_field_sequence(self, time_delays):
        raw = self.raw_sensor_response
        lixel_sequence = np.zeros(
            shape=(raw.number_time_slices, raw.number_lixel),
            dtype=np.uint16)
        phs.stream2sequence(
            photon_stream=raw.photon_stream,
            time_slice_duration=raw.time_slice_duration,
            NEXT_READOUT_CHANNEL_MARKER=raw.NEXT_READOUT_CHANNEL_MARKER,
            sequence=lixel_sequence,
            time_delay_mean=time_delays)
        return lixel_sequence

    def photon_arrival_times_and_lixel_ids(self):
        """
        Returns (arrival_slices, lixel_ids) of all recorded photons.
        """
        (arrival_slices, lixel_ids
            ) = phs.arrival_slices_and_lixel_ids(
                self.raw_sensor_response)
        return (
            arrival_slices*self.raw_sensor_response.time_slice_duration,
            lixel_ids)

    def light_field_sequence_for_isochor_image(self):
        return self._light_field_sequence(
            time_delays=self.light_field_geometry.time_delay_image_mean)

    def light_field_sequence_for_isochor_aperture(self):
        return self._light_field_sequence(
            time_delays=self.light_field_geometry.time_delay_mean)

    def light_field_sequence_raw(self):
        return self._light_field_sequence(
            time_delays=np.zeros(
                self.light_field_geometry.number_lixel, dtype=np.float32))

    def show(self):
        """
        Shows an overview figure of the Event.

        1)  Directional intensity distribution accross the field of view
            (the classic IACT image)

        2)  Positional intensity distribution on the principal aperture plane

        3)  The arrival time distribution of photo equivalents accross all
            lixels

        4)  The photo equivalent distribution accross all lixels
        """
        raw = self.raw_sensor_response
        lixel_sequence = self.lixel_sequence_raw()

        pix_img_seq = lfs.pixel_sequence(
            lixel_sequence=lixel_sequence,
            number_pixel=self.light_field_geometry.number_pixel,
            number_paxel=self.light_field_geometry.number_paxel)

        pax_img_seq = lfs.paxel_sequence(
            lixel_sequence=lixel_sequence,
            number_pixel=self.light_field_geometry.number_pixel,
            number_paxel=self.light_field_geometry.number_paxel)

        fig, axs = plt.subplots(2, 2)
        plt.suptitle(self._plot_suptitle())
        pix_int = lfs.integrate_around_arrival_peak(
            sequence=pix_img_seq,
            integration_radius=1)
        pixel_image = image.Image(
            pix_int['integral'],
            self.light_field_geometry.pixel_pos_cx,
            self.light_field_geometry.pixel_pos_cy)
        axs[0][0].set_title(
            'directional image at time slice '+str(pix_int['peak_slice']))
        image.plot.add_pixel_image_to_ax(pixel_image, axs[0][0])
        pax_int = lfs.integrate_around_arrival_peak(
            sequence=pax_img_seq,
            integration_radius=1)
        paxel_image = image.Image(
            pax_int['integral'],
            self.light_field_geometry.paxel_pos_x,
            self.light_field_geometry.paxel_pos_y)
        axs[0][1].set_title(
            'principal aperture at time slice '+str(pax_int['peak_slice']))
        image.plot.add_paxel_image_to_ax(paxel_image, axs[0][1])
        light_field.plot.add2ax_hist_arrival_time(
            sequence=lixel_sequence,
            time_slice_duration=raw.time_slice_duration,
            ax=axs[1][0])
        light_field.plot.add2ax_hist_intensity(lixel_sequence, axs[1][1])
        plt.show()
