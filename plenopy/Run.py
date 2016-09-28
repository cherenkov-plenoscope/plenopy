import numpy as np
import glob
import os
from . import FileSystemFormat
from .LixelStatistics import LixelStatistics
from .Event import Event


class Run(object):
    """
    A run of plenoscope events.

    number_events   The number count of all events in this run

    event_numbers   [number_events]
                    All event numbers found in this run (in ascending order)
    """

    def __init__(self, path):
        """
        Parameters
        ----------
        path        The path to the directory representing the run.
        """
        self.path = os.path.abspath(path)
        if not os.path.isdir(self.path):
            raise NotADirectoryError(self.path)
        self.path_input = os.path.join(self.path, 'input')
        self.path_input_plenoscope = os.path.join(
            self.path_input, 'plenoscope')

        self.lixel_statistics = LixelStatistics(self.path_input_plenoscope)
        self.event_numbers = self._event_numbers_in_run()
        self.number_events = self.event_numbers.shape[0]

    def _event_numbers_in_run(self):
        return FileSystemFormat.all_folders_with_digit_names_in_path(self.path)

    def __getitem__(self, index):
        """
        Returns the index-th event of the run.

        Parameters
        ----------
        index       The index of the event to be returned. (starting at 0).      
        """

        try:
            event_number = self.event_numbers[index]
        except(IndexError):
            raise StopIteration
        event_path = os.path.join(self.path, str(event_number))
        return Event(event_path, self.lixel_statistics)

    def __repr__(self):
        out = 'Run('
        out += "path='" + self.path + "', "
        out += str(self.number_events) + ' events)\n'
        return out
