import numpy as np
from os import path as op
from . import utils
from .. import raw_light_field_sensor_response
from .. import corsika
from .. import simulation_truth
from .. import classify
from .. import tools


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

        _raw_path = op.join(self._path, "raw_light_field_sensor_response.phs")

        with tools.acp_format.gz_transparent_open(_raw_path, "rb") as f:
            self.raw_sensor_response = raw_light_field_sensor_response.read(f)

        if self.type == "SIMULATION":
            self._read_simulation_truth()
        self.number = int(op.basename(self._path))
        self._read_dense_photons()

    def _read_event_header(self):
        header_path = op.join(self._path, "event_header.bin")
        header = tools.header273float32.read_float32_header(header_path)
        self.type = utils.event_type_from_header(header)
        self.trigger_type = utils.trigger_type_from_header(header)

    def _read_simulation_truth(self):
        truth_path = op.join(self._path, "simulation_truth")
        if (
            self.trigger_type
            == "EXTERNAL_TRIGGER_BASED_ON_AIR_SHOWER_SIMULATION_TRUTH"
        ):
            simulation_truth_event = simulation_truth.Event(
                evth=corsika.EventHeader(
                    op.join(truth_path, "corsika_event_header.bin")
                ),
                runh=corsika.RunHeader(
                    op.join(truth_path, "corsika_run_header.bin")
                ),
            )

            try:
                air_shower_photon_bunches = corsika.PhotonBunches(
                    op.join(truth_path, "air_shower_photon_bunches.bin")
                )
            except FileNotFoundError:
                air_shower_photon_bunches = None

            try:
                simulation_truth_detector = simulation_truth.Detector(
                    op.join(truth_path, "detector_pulse_origins.bin")
                )
            except FileNotFoundError:
                simulation_truth_detector = None

            try:
                _raw_header = tools.header273float32.read_float32_header(
                    op.join(truth_path, "mctracer_event_header.bin")
                )
                photon_propagator = simulation_truth.PhotonPropagator(
                    raw_header=_raw_header
                )
            except FileNotFoundError:
                photon_propagator = None

            self.simulation_truth = simulation_truth.SimulationTruth(
                event=simulation_truth_event,
                air_shower_photon_bunches=air_shower_photon_bunches,
                detector=simulation_truth_detector,
                photon_propagator=photon_propagator,
            )

    def _read_dense_photons(self):
        path = op.join(self._path, "dense_photon_ids.uint32.gz")
        if op.exists(path):
            self.dense_photon_ids = classify.read_dense_photon_ids(path)
            photons = classify.RawPhotons.from_event(self)
            mask = np.zeros(photons.x.shape[0], dtype=bool)
            mask[self.dense_photon_ids] = True
            self.cherenkov_photons = photons.cut(mask)
        else:
            self.dense_photon_ids = None
            self.cherenkov_photons = None

    def __repr__(self):
        out = self.__class__.__name__
        out += "("
        out += "number " + str(self.number) + ", "
        out += "type '" + self.type
        out += "')"
        return out
