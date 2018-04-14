import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from . import plenoscope_event_header
from ..light_field_geometry import PlenoscopeGeometry
from ..RawLightFieldSensorResponse import RawLightFieldSensorResponse
from ..light_field.LightField import LightField
from ..light_field import sequence
from ..tools.HeaderRepresentation import assert_marker_of_header_is
from ..tools.HeaderRepresentation import read_float32_header
from ..image.Image import Image
from .. import image
from .. import corsika
from .. import simulation_truth
from .. import light_field


class Event(object):
    """
    Atmospheric Cherenkov Plenoscope (ACP) event.

    number                      The ID of this event in its run.

    type                        Event type indicator (string)
                                "simulation" in case this event was simulated
                                "observation" in case this event was observed

    trigger_type                Indicates the trigger mechanism.

    light_field                 The light field recorded by the plenoscope.
                                Created using the raw light field sensor
                                response and the light field geometry.

    simulation_truth            If type == "simulation"
                                Additional 'true' information known from the
                                simulation itself.

    raw_sensor_response         The raw light field sensor response
                                of the plenoscope.
    """
    def __init__(self, path, light_field_geometry):
        """
        Parameter
        ---------
        path                    The path of this event. Typically inside a Run
                                directory.

        light_field_geometry    The light field geometry to calibrate the raw
                                sensor response of this event.
        """
        self.light_field_geometry = light_field_geometry
        self._path = os.path.abspath(path)
        self._read_event_header()
        raw_path = os.path.join(self._path, 'raw_light_field_sensor_response.phs')
        self.raw_sensor_response = RawLightFieldSensorResponse(raw_path)
        self.light_field = LightField(
            self.raw_sensor_response,
            self.light_field_geometry)
        if self.type == 'SIMULATION':
            self._read_simulation_truth()
        self.number = int(os.path.basename(self._path))

    def _read_event_header(self):
        header_path = os.path.join(self._path, 'event_header.bin')
        header = read_float32_header(header_path)
        self.type = plenoscope_event_header.event_type(header)
        self.trigger_type = plenoscope_event_header.trigger_type(header)

    def _read_simulation_truth(self):
        sim_truth_path = os.path.join(self._path, 'simulation_truth')
        if self.trigger_type == 'EXTERNAL_TRIGGER_BASED_ON_AIR_SHOWER_SIMULATION_TRUTH':
            simulation_truth_event = simulation_truth.Event(
                evth=corsika.EventHeader(os.path.join(sim_truth_path, 'corsika_event_header.bin')),
                runh=corsika.RunHeader(os.path.join(sim_truth_path, 'corsika_run_header.bin')))

            try:
                simulation_truth_air_shower_photon_bunches = corsika.PhotonBunches(
                    os.path.join(sim_truth_path, 'air_shower_photon_bunches.bin'))
            except(FileNotFoundError):
                simulation_truth_air_shower_photon_bunches = None

            try:
                simulation_truth_detector = simulation_truth.Detector(
                    os.path.join(sim_truth_path, 'detector_pulse_origins.bin'))
            except(FileNotFoundError):
                simulation_truth_detector = None

            self.simulation_truth = simulation_truth.SimulationTruth(
                event=simulation_truth_event,
                air_shower_photon_bunches=simulation_truth_air_shower_photon_bunches,
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
            if self.trigger_type == "EXTERNAL_TRIGGER_BASED_ON_AIR_SHOWER_SIMULATION_TRUTH":
                return self.simulation_truth.event.short_event_info()
            elif self.trigger_type == "EXTERNAL_RANDOM_TRIGGER":
                return 'Extrenal random trigger, no air shower'
            else:
                return 'Simulation, but trigger type is unknown: '+str(self.trigger_type)
        elif self.type == "OBSERVATION":
            return 'Observation'
        else:
            return 'unknown event type: '+str(self.type)

    def show(self):
        """
        Shows an overview figure of the Event.

        1   Directional intensity distribution accross the field of view
            (the classic IACT image)

        2   Positional intensity distribution on the principal aperture plane

        3   The arrival time distribution of photo equivalents accross all
            lixels

        4   The photo equivalent distribution accross all lixels
        """
        fig, axs = plt.subplots(2, 2)
        plt.suptitle(self._plot_suptitle())
        pix_img_seq = self.light_field.pixel_sequence()
        pix_int = light_field.sequence.integrate_around_arrival_peak(
            sequence=pix_img_seq,
            integration_radius=1
        )
        pixel_image = Image(
            pix_int['integral'],
            self.light_field.pixel_pos_cx,
            self.light_field.pixel_pos_cy
        )
        axs[0][0].set_title(
            'directional image at time slice '+str(pix_int['peak_slice'])
        )
        image.plot.add_pixel_image_to_ax(pixel_image, axs[0][0])
        pax_img_seq = self.light_field.paxel_sequence()
        pax_int = light_field.sequence.integrate_around_arrival_peak(
            sequence=pax_img_seq,
            integration_radius=1
        )
        paxel_image = Image(
            pax_int['integral'],
            self.light_field.paxel_pos_x,
            self.light_field.paxel_pos_y
        )
        axs[0][1].set_title(
            'principal aperture at time slice '+str(pax_int['peak_slice'])
        )
        image.plot.add_paxel_image_to_ax(paxel_image, axs[0][1])
        light_field.plot.add2ax_hist_arrival_time(self.light_field, axs[1][0])
        light_field.plot.add2ax_hist_intensity(self.light_field, axs[1][1])
        plt.show()
