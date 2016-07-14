#!/usr/bin/env python
from __future__ import absolute_import, print_function, division
import numpy as np

def circular_mask(xs, ys, x, y, r, smear=0.0):

    r_smear = r + smear

    if smear > 0.0:
        slope = -1.0/(r_smear - r)
        intercept = 1.0 - slope*r

    center = np.array([x,y])
    mask = np.zeros(shape=xs.shape)

    for pax in range(xs.shape[0]):
        pax_pos = np.array([xs[pax], ys[pax]])
        dist = np.linalg.norm(center - pax_pos)
        if dist <= r:
            mask[pax] = 1.0
        elif dist > r and dist < r_smear and smear > 0.0:
            mask[pax] = slope*dist + intercept

    return mask