#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, print_function, division
import numpy as np
import os
from .HeaderRepresentation import corsika_event_header_repr
from .HeaderRepresentation import corsika_run_header_repr


def CORSIKA_primary_ID2str(PRMPAR):
    """
    Return string   Convert the CORSIKA primary particle ID into a human 
                    readable string.

    Parameter
    ---------
    PRMPAR          The CORSIKA primary particle ID
    """
    PRMPAR = int(PRMPAR)
    if PRMPAR == 1:
        return 'gamma'
    elif PRMPAR == 2:
        return 'e^+'
    elif PRMPAR == 3:
        return 'e^-'
    elif PRMPAR == 5:
        return 'muon^+'
    elif PRMPAR == 6:
        return 'muon^-'
    elif PRMPAR == 7:
        return 'pion^0'
    elif PRMPAR == 8:
        return 'pion^+'
    elif PRMPAR == 9:
        return 'pion^-'
    elif PRMPAR == 14:
        return 'p'
    elif PRMPAR > 200:

        A = int(np.floor(PRMPAR / 100))
        Z = PRMPAR - A * 100
        if A == 4 and Z == 2:
            return 'He A4 Z2'
        else:
            return 'A' + str(round(A)) + ' Z' + str(round(Z))
    else:
        return str(PRMPAR)


class SimulationTruth(object):
    """
    Additional truth known from the simulation itself

    CORSIKA run header      [float 273] The raw run header

    CORSIKA event header    [float 273] The raw event header
    """

    def __init__(self, path):
        self.corsika_event_header = self.__read_273_float_header(
            os.path.join(path, 'corsika_event_header.bin')
        )

        self.corsika_run_header = self.__read_273_float_header(
            os.path.join(path, 'corsika_run_header.bin')
        )

    def __read_273_float_header(self, path):
        raw = np.fromfile(path, dtype=np.float32)
        return raw

    def __repr__(self):
        out = ''
        out += corsika_run_header_repr(self.corsika_run_header)
        out += '\n'
        out += corsika_event_header_repr(self.corsika_event_header)
        return out

    def short_event_info(self):
        """
        Return string       A short string to summarize the simulation truth.
                            Can be added in e.g. plot headings.
        """
        az = str(round(np.rad2deg(self.corsika_event_header[12 - 1]), 2))
        zd = str(round(np.rad2deg(self.corsika_event_header[11 - 1]), 2))
        core_y = str(round(0.01 * self.corsika_event_header[118], 2))
        core_x = str(round(0.01 * self.corsika_event_header[98], 2))
        E = str(round(self.corsika_event_header[4 - 1], 2))
        PRMPAR = self.corsika_event_header[3 - 1]
        run_id = str(int(self.corsika_run_header[2 - 1]))
        evt_id = str(int(self.corsika_event_header[2 - 1]))
        return str(
            "Run: " + run_id + ", Event: " + evt_id + ", " +
            CORSIKA_primary_ID2str(PRMPAR) + ', ' +
            "E: " + E + "GeV, \n" +
            "core pos: x=" + core_x + 'm, ' +
            "y=" + core_y + 'm, ' +
            "direction: Zd=" + zd + 'deg, ' +
            "Az=" + az + 'deg')
