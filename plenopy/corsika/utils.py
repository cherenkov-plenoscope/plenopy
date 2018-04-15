import numpy as np
from ..tools import HeaderRepresentation as hr


def event_header_repr(h):
    """
    Returns a human readable string to represent the CORSIKA event header.

    Parameters
    ----------

    h           The raw CORSIKA event header float32 array.
    """
    hr.assert_shape_is_valid(h)
    hr.assert_marker_of_header_is(h, 'EVTH')

    out = 'CORSIKA event header\n'
    out += '  2 ' + 'event number ' + str(int(h[2 - 1])) + '\n'
    out += '  3 ' + 'particle id ' + str(int(h[3 - 1])) + '\n'
    out += '  4 ' + 'total energy ' + str(h[4 - 1]) + ' GeV\n'
    out += '  5 ' + 'starting altitude ' + str(h[5 - 1]) + ' g/cm^2\n'
    # out+= '  6 '+'number of first target if fixed'+str(h[6-1])+'\n'
    out += '  7 ' + \
        'z coordinate (height) of first interaction ' + str(h[7 - 1]) + ' cm\n'
    out += ' 11 ' + 'zenith angle Theta ' + \
        str(np.rad2deg(h[11 - 1])) + ' deg\n'
    out += ' 12 ' + 'azimuth angle Phi ' + \
        str(np.rad2deg(h[12 - 1])) + ' deg\n'
    out += ' 98 ' + 'number of uses ' + str(int(h[98 - 1])) + '\n'
    for i in range(int(h[98 - 1])):
        out += '    ' + 'reuse ' + str(i + 1) + ': core position x=' + str(
            h[98 - 1 + i + 1]) + ' cm, y=' + str(h[118 - 1 + i + 1]) + ' cm\n'
    return out


def run_header_repr(h):
    """
    Returns a human readable string to represent the CORSIKA hun header.

    Parameters
    ----------

    h           The raw CORSIKA run header float32 array.
    """
    hr.assert_shape_is_valid(h)
    hr.assert_marker_of_header_is(h, 'RUNH')

    out = 'CORSIKA run header\n'
    out += '  2 ' + 'run number ' + str(int(h[2 - 1])) + '\n'
    out += '  4 ' + 'date of begin run (yymmdd) ' + str(int(h[3 - 1])) + '\n'
    out += '  4 ' + 'version of program ' + str(h[4 - 1]) + '\n'
    out += ' 16 ' + 'slope of energy spectrum ' + str(h[16 - 1]) + '\n'
    out += ' 17 ' + 'lower limit of energy range ' + str(h[17 - 1]) + ' GeV\n'
    out += ' 18 ' + 'upper limit of energy range ' + str(h[18 - 1]) + ' GeV\n'
    out += '248 ' + 'XSCATT scatter range in x direction for Cherenkov ' + \
        str(h[248 - 1]) + ' cm\n'
    out += '249 ' + 'YSCATT scatter range in x direction for Cherenkov ' + \
        str(h[249 - 1]) + ' cm\n'
    return out


def prmpar_repr(PRMPAR):
    """
    Return string   Convert the CORSIKA primary particle ID (PRMPAR) to a human
                    readable string.

    Parameter
    ---------
    PRMPAR          The CORSIKA primary particle ID (PRMPAR)
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
        out = ''
        A = int(np.floor(PRMPAR / 100))
        Z = PRMPAR - A * 100
        if A == 4:
            out += 'He '
        elif A == 56:
            out += 'Fe '
        return out + 'A' + str(round(A)) + ' Z' + str(round(Z))
    else:
        return 'ID_'+str(PRMPAR)


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
        prmpar_repr(PRMPAR) + ', ' +
        "E: " + E + "GeV, \n" +
        "core pos: x=" + core_x + 'm, ' +
        "y=" + core_y + 'm, ' +
        "direction: Zd=" + zd + 'deg, ' +
        "Az=" + az + 'deg')
