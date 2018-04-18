import numpy as np
from os import path as op
from . import utils
from ..RawLightFieldSensorResponse import RawLightFieldSensorResponse
from ..photon_stream import cython_reader as phs
from ..tools import header273float32 as hr
from .. import corsika
from .. import simulation_truth


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
        self.type = utils.event_type_from_header(header)
        self.trigger_type = utils.trigger_type_from_header(header)

    def _read_simulation_truth(self):
        truth_path = op.join(self._path, 'simulation_truth')
        if (
            self.trigger_type ==
            'EXTERNAL_TRIGGER_BASED_ON_AIR_SHOWER_SIMULATION_TRUTH'
        ):
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

    def _light_field_sequence(self, time_delays_to_be_subtracted):
        raw = self.raw_sensor_response
        lixel_sequence = np.zeros(
            shape=(raw.number_time_slices, raw.number_lixel),
            dtype=np.uint16)
        phs.stream2sequence(
            photon_stream=raw.photon_stream,
            time_slice_duration=raw.time_slice_duration,
            NEXT_READOUT_CHANNEL_MARKER=raw.NEXT_READOUT_CHANNEL_MARKER,
            sequence=lixel_sequence,
            time_delay_mean=time_delays_to_be_subtracted)
        return lixel_sequence

    def photon_arrival_times_and_lixel_ids(self):
        """
        Returns (arrival_slices, lixel_ids) of all recorded photons.
        """
        (arrival_slices, lixel_ids) = phs.arrival_slices_and_lixel_ids(
            self.raw_sensor_response)
        return (
            arrival_slices*self.raw_sensor_response.time_slice_duration,
            lixel_ids)

    def light_field_sequence_for_isochor_image(self):
        return self._light_field_sequence(
            time_delays_to_be_subtracted=
                -self.light_field_geometry.time_delay_image_mean)

    def light_field_sequence_for_isochor_aperture(self):
        return self._light_field_sequence(
            time_delays_to_be_subtracted=
                +self.light_field_geometry.time_delay_mean)

    def light_field_sequence_raw(self):
        return self._light_field_sequence(
            time_delays_to_be_subtracted=np.zeros(
                self.light_field_geometry.number_lixel, dtype=np.float32))

    def __repr__(self):
        out = self.__class__.__name__
        out += "("
        out += "number " + str(self.number) + ", "
        out += "type '" + self.type
        out += "')"
        return out
