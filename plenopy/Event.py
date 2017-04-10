import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .PlenoscopeGeometry import PlenoscopeGeometry
from .RawLightFieldSensorResponse import RawLightFieldSensorResponse
from .LightField import LightField
from .HeaderRepresentation import str2float
from .HeaderRepresentation import read_float32_header
from .Image import Image
from .plot import Image as plt_Image
from . import Corsika
from . import SimulationTruth
from .plot import LightField as plt_LightFieldSequence


class Event(object):
    """
    number                      The event number in the run

    type                        A string as type indicator
                                "simulation" in case this event was simulated
                                "observation" in case this event was observed

    trigger_type                Indicates the trigger mechanism.

    light_field                 The light field recorded by the plenoscope.
                                Created using the raw light field sensor  
                                response and the lixel statistics appropriate 
                                for the given sensor plane 2 imaging system 
                                orientation and position of this event.

    simulation_truth_air_shower If type == "simulation"
                                Additional 'true' information known from the 
                                simulation itself 

    simulation_truth_detector   Optional, additional information about the 
                                electric pulse composition. Tells which air
                                shower photons contributed to which pulse and
                                if electric cross talk, after pulses or dark 
                                noise pulses are present

    raw_light_field_sensor_response     The raw light field sensor response
                                        of the plenoscope

    sensor_plane2imaging_system     The relative orientation and position of 
                                    the plenoscope's light field sensor with 
                                    respect to the plenoscope's imaging system 
                                    in the moment this event was recorded.
    """

    def __init__(self, path, light_field_geometry):
        self._path = os.path.abspath(path)
        self._read_event_header()

        self.raw_light_field_sensor_response = RawLightFieldSensorResponse(
            os.path.join(self._path, 'raw_light_field_sensor_response.phs'))

        self.light_field = LightField(
            self.raw_light_field_sensor_response,
            light_field_geometry)

        if self.type == 'SIMULATION':
            self._read_simulation_truth()

        self.number = int(os.path.basename(self._path))

    def _read_event_header(self):
        event_header_path = os.path.join(self._path, 'event_header.bin')
        self.header = read_float32_header(event_header_path)
        self.plenoscope_geometry = PlenoscopeGeometry(self.header)

        # TYPE
        assert self.header[  1-1] == str2float("PEVT")
        if self.header[  2-1] == 0.0:
            self.type = 'OBSERVATION'
        elif self.header[  2-1] == 1.0:
            self.type = 'SIMULATION'
        else:
            self.type = 'unknown: '+str(self.header[  2-1])

        # TRIGGER TYPE
        if self.header[  3-1] == 0.0:
            self.trigger_type = 'SELF_TRIGGER'
        elif self.header[  3-1] == 1.0:
            self.trigger_type = 'EXTERNAL_RANDOM_TRIGGER'
        elif self.header[  3-1] == 2.0:
            self.trigger_type = 'EXTERNAL_TRIGGER_BASED_ON_AIR_SHOWER_SIMULATION_TRUTH'
        else:
            self.trigger_type = 'unknown: '+str(self.header[  3-1])

    def _read_simulation_truth(self):
        sim_truth_path = os.path.join(self._path, 'simulation_truth')

        if self.trigger_type == 'EXTERNAL_TRIGGER_BASED_ON_AIR_SHOWER_SIMULATION_TRUTH':
            evth = Corsika.EventHeader(os.path.join(sim_truth_path, 'corsika_event_header.bin'))
            runh = Corsika.RunHeader(os.path.join(sim_truth_path, 'corsika_run_header.bin'))            
            simulation_truth_event = SimulationTruth.Event(evth=evth, runh=runh)

            try:
                simulation_truth_air_shower_photon_bunches = Corsika.PhotonBunches(
                    os.path.join(sim_truth_path, 'air_shower_photon_bunches.bin'))
            except(FileNotFoundError):
                simulation_truth_air_shower_photon_bunches = None         

            try:
                simulation_truth_detector = SimulationTruth.Detector(
                    self.light_field,
                    os.path.join(sim_truth_path, 'detector_pulse_origins.bin'))
            except(FileNotFoundError):
                simulation_truth_detector = None

            self.simulation_truth = SimulationTruth.SimulationTruth(
                event=simulation_truth_event,
                air_shower_photon_bunches=simulation_truth_air_shower_photon_bunches,
                detector=simulation_truth_detector)


    def __repr__(self):
        out = "Event("
        out += "number " + str(self.number) + ", "
        out += "type '" + self.type
        out += "')\n"
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


    def plot(self):
        """
        This will open a plot showing:

        1   Directional intensity distribution accross the field of view
            (the classical IACT image)

        2   Positional intensity distribution on the principal aperture plane

        3   The arrival time distribution of photo equivalents accross all 
            lixels 

        4   The photo equivalent distribution accross all lixels
        """
        fig, axs = plt.subplots(2, 2)
        plt.suptitle(self._plot_suptitle())

        pix_img_seq = self.light_field.pixel_sequence()
        t_m = time_slice_with_max_intensity(pix_img_seq)
        ts = np.max([t_m-1, 0])
        te = np.min([t_m+1, pix_img_seq.shape[0]-1])
        pixel_image = Image(
            pix_img_seq[ts:te].sum(axis=0),
            self.light_field.pixel_pos_cx,
            self.light_field.pixel_pos_cy)

        axs[0][0].set_title('directional image')
        plt_Image.add_pixel_image_to_ax(pixel_image, axs[0][0])

        pax_img_seq = self.light_field.paxel_sequence()
        t_m = time_slice_with_max_intensity(pax_img_seq)
        ts = np.max([t_m-1, 0])
        te = np.min([t_m+1, pax_img_seq.shape[0]-1])
        paxel_image = Image(
            pax_img_seq[ts:te].sum(axis=0),
            self.light_field.paxel_pos_x,
            self.light_field.paxel_pos_y)

        axs[0][1].set_title('principal aperture plane')
        plt_Image.add_paxel_image_to_ax(paxel_image, axs[0][1])

        plt_LightFieldSequence.add2ax_hist_arrival_time(self.light_field, axs[1][0])

        plt_LightFieldSequence.add2ax_hist_intensity(self.light_field, axs[1][1])
        #raw_intensity_patch = mpatches.Patch(color='blue', label='raw')
        #eff_corrected_intensity_patch = mpatches.Patch(
        #    color='green', label='efficiency corrected')
        #axs[1][1].legend(handles=[raw_intensity_patch,
        #                          eff_corrected_intensity_patch])
        plt.show()



def time_slice_with_max_intensity(sequence):
    max_along_slices = np.zeros(sequence.shape[0], dtype=np.uint16)
    for s, slic in enumerate(sequence):
        max_along_slices[s] = slic.max()
    return np.argmax(max_along_slices)