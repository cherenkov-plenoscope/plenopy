import numpy as np
from .primary_particle_id2str import primary_particle_id2str

def short_event_info(runh, evth):
    """
    Return string       A short string to summarize the simulation truth.
                        Can be added in e.g. plot headings.
    """
    az = str(round(np.rad2deg(evth[12 - 1]), 2))
    zd = str(round(np.rad2deg(evth[11 - 1]), 2))
    core_y = str(round(0.01 * evth[118], 2))
    core_x = str(round(0.01 * evth[98], 2))
    E = str(round(evth[4 - 1], 2))
    PRMPAR = evth[3 - 1]
    run_id = str(int(runh[2 - 1]))
    evt_id = str(int(evth[2 - 1]))
    return str(
        "Run: " + run_id + ", Event: " + evt_id + ", " +
        primary_particle_id2str(PRMPAR) + ', ' +
        "E: " + E + "GeV, \n" +
        "core pos: x=" + core_x + 'm, ' +
        "y=" + core_y + 'm, ' +
        "direction: Zd=" + zd + 'deg, ' +
        "Az=" + az + 'deg')