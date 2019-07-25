import numpy as np
from . import trigger


def estimate_nearest_neighbors(x, y, epsilon, itself=False):
    nn = trigger.neighborhood(x, y, epsilon, itself=itself)
    ns = []
    num_points = len(x)
    for i in range(num_points):
        ns.append(np.arange(num_points)[nn[i, :]])
    return ns


def islands_greater_equal_threshold(intensities, neighborhood, threshold):
    num_points = len(intensities)
    intensities = np.array(intensities)
    abv_thr = set(np.arange(num_points)[intensities >= threshold])
    to_do = abv_thr.copy()
    islands = []
    while len(to_do) > 0:
        i = next(iter(to_do))
        to_do.remove(i)
        island = [i]
        i_neighbors = set(neighborhood[i])
        candidates = to_do.intersection(i_neighbors)
        while len(candidates) > 0:
            c = next(iter(candidates))
            if c in abv_thr:
                island.append(c)
                candidates.remove(c)
                to_do.remove(c)
                c_neighbors = set(neighborhood[c])
                c_neighbors_to_do = to_do.intersection(c_neighbors)
                candidates = candidates.union(c_neighbors_to_do)
        islands.append(island)
    return islands
