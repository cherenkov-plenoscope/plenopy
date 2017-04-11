import numpy as np
import os
from .. import corsika

class Event(object):
    """
    Additional truth known from the event simulation

    CORSIKA run header      [float 273] run header

    CORSIKA event header    [float 273] event header
    """

    def __init__(self, evth, runh):
        self.corsika_event_header = evth
        self.corsika_run_header = runh

    def __repr__(self):
        out = ''
        out += corsika.run_header_repr(self.corsika_run_header.raw)
        out += '\n'
        out += corsika.event_header_repr(self.corsika_event_header.raw)
        return out

    def short_event_info(self):
        return corsika.short_event_info(
            self.corsika_run_header.raw,
            self.corsika_event_header.raw)