#!/usr/bin/env python
from __future__ import absolute_import, print_function, division
import numpy as np


def circular(support_xs, support_ys, x, y, r, smear=0.0):
    """
    Return weighted mask

    Parameters
    ----------
    xs,ys   x and y positions of support points

    x,y     x and y position of circle

    r       radius of circle

    smear   [optional] If smear != 0.0, support points will
            be taken into account not abrupt but smootly according to 
            their distance to the circle's center

                /\ support point weight
                 |
            1.0 -|-------\ 
                 |       .\ 
                 |       . \ 
                 |       .  \ 
                 |       .   \ 
                 |       .    \ 
                 |       .     \ 
                 |       .      \ 
                 |       .       \ 
            0.0 -|-------|--------|--------------->
                         r       r+smear       distance of support point
                                               to center of circle
    """
    r_smear = r + smear

    if smear > 0.0:
        slope = -1.0 / (r_smear - r)
        intercept = 1.0 - slope * r

    center = np.array([x, y])
    mask = np.zeros(shape=support_xs.shape)

    for sup in range(support_xs.shape[0]):
        sup_pos = np.array([support_xs[sup], support_ys[sup]])
        dist = np.linalg.norm(center - sup_pos)
        if dist <= r:
            mask[sup] = 1.0
        elif dist > r and dist < r_smear and smear > 0.0:
            mask[sup] = slope * dist + intercept

    return mask
