import numpy as np
import os
from .. import Corsika
from .Event import Event
from .. import FileSystemFormat

class Run(object):
    """
    Idealized Plenoscope Run

    number_events   The number count of all events in this run

    event_numbers   [number_events]
                    All event numbers found in this run (in ascending order)
    """

    def __init__(self, path):
        """
        Parameters
        ----------
        path        The path to the directory representing the corsika run.
                    The run must not be in EventIO format but rolled out into
                    a file system directory structure.
        """
        self.path = os.path.abspath(path)
        if not os.path.isdir(self.path):
            raise NotADirectoryError(self.path)

        self.event_numbers = self._event_numbers_in_run()
        self.number_events = self.event_numbers.shape[0]

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
        return Event(event_path)

    def _event_numbers_in_run(self):
        return FileSystemFormat.all_folders_with_digit_names_in_path(self.path)

    def __repr__(self):
        out = 'IdealizedPlenoscopeRun('
        out += "path='" + self.path + "', "
        out += str(self.number_events) + ' events)\n'
        return out
