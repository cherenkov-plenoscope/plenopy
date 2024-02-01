import numpy as np
import glob
import os
from .light_field_geometry.LightFieldGeometry import LightFieldGeometry
from .event.Event import Event
from .tools.acp_format import all_folders_with_digit_names_in_path


class Run(object):
    """
    A run of Atmospheric Cherenkov Plenoscope (ACP) events.

    number_events           The number count of all events in this run.

    event_numbers           All event numbers found in this run
                            (in ascending order)

    light_field_geometry    The Plenoscope (light field) geometry during this
                            run.

    path                    The path of this run.
    """

    def __init__(self, path, light_field_geometry=None):
        """
        Parameters
        ----------
        path        The path to the directory representing the run.
        """
        self.path = os.path.abspath(path)
        if not os.path.isdir(self.path):
            raise NotADirectoryError(self.path)

        if light_field_geometry is None:
            self.light_field_geometry = LightFieldGeometry(
                os.path.join(self.path, "input", "plenoscope")
            )
        else:
            self.light_field_geometry = light_field_geometry
        self.event_numbers = self._event_numbers_in_run()
        self.number_events = self.event_numbers.shape[0]

    def _event_numbers_in_run(self):
        return all_folders_with_digit_names_in_path(self.path)

    def __getitem__(self, index):
        """
        Returns the index-th event of this run.

        Parameters
        ----------
        index       The index of the event to be returned. (starting at 0).
        """
        try:
            event_number = self.event_numbers[index]
        except IndexError:
            raise StopIteration
        event_path = os.path.join(self.path, str(event_number))
        return Event(event_path, self.light_field_geometry)

    def __len__(self):
        return self.number_events

    def __repr__(self):
        out = "Run("
        out += "'" + self.path + "', "
        out += str(self.number_events) + " events)"
        return out
