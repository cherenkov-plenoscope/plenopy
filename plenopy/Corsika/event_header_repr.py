import numpy as np
from .. import HeaderRepresentation

def event_header_repr(h):
    HeaderRepresentation.assert_shape_is_valid(h)
    HeaderRepresentation.assert_marker_of_header_is(h, 'EVTH')

    out = 'CORSIKA event header\n'
    out += '  2 ' + 'event number ' + str(int(h[2 - 1])) + '\n'
    out += '  3 ' + 'particle id ' + str(int(h[3 - 1])) + '\n'
    out += '  4 ' + 'total energy ' + str(h[4 - 1]) + ' GeV\n'
    out += '  5 ' + 'starting altitude ' + str(h[5 - 1]) + ' g/cm^2\n'
    #out+= '  6 '+'number of first target if fixed'+str(h[6-1])+'\n'
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