import numpy as np
import scipy.spatial.distance


def neighborhood(x, y, epsilon, itself=False):
    dists = scipy.spatial.distance_matrix(
        np.vstack([x, y]).T, np.vstack([x, y]).T
    )
    nn = dists <= epsilon
    if not itself:
        nn = nn * (dists != 0)
    return nn
