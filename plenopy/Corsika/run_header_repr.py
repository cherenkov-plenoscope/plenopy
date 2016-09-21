import numpy as np
from .. import HeaderRepresentation

def run_header_repr(h):
    HeaderRepresentation.assert_shape_is_valid(h)
    HeaderRepresentation.assert_marker_of_header_is(h, 'RUNH')

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