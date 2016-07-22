#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, print_function, division
import numpy as np
import glob, os
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
        self.__path = os.path.abspath(path)
        if not os.path.isdir(self.__path):
            raise NotADirectoryError(self.__path)
        self.__path_input = os.path.join(self.__path, 'input')
        self.__path_input_plenoscope = os.path.join(self.__path_input, 'plenoscope')

        self.lixel_statistics = LixelStatistics(self.__path_input_plenoscope)
        self.event_numbers = self.__event_numbers_in_run()
        self.number_events = self.event_numbers.shape[0]

    def __event_numbers_in_run(self):
        files_in_run_folder = glob.glob(os.path.join(self.__path, '*'))
        events =[]
        for fi in files_in_run_folder:
            if os.path.isdir(fi) and os.path.basename(fi).isdigit():
                events.append(int(os.path.basename(fi)))
        events = np.array(events)
        events.sort()
        return events

    def event(self, i):
        """
        Returns the i-th event in the run.

        Parameters
        ----------
        i           The event_number of the event to be returned. 
                    This is the CORSIKA event number (starting at 1).
        """
        event_path = os.path.join(self.__path, str(i))
        return Event(event_path, self.lixel_statistics)        

    def __getitem__(self, index):
        """
        Returns an event of run.

        Parameters
        ----------
        i           The index of the event to be returned. (starting at 0).      
        """
        try:
            return self.event(index+1)
        except(FileNotFoundError):
            raise StopIteration

    def __str__(self):
        out = 'Run('
        out+="path='"+self.__path+"', "
        out+= str(self.number_events)+' events)\n'
        return out        

    def __repr__(self):
        return self.__str__()