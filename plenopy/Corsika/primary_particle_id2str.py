import numpy as np

def primary_particle_id2str(PRMPAR):
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
        return str(PRMPAR)