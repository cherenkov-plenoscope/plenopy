#!/usr/bin/env python
from __future__ import absolute_import, print_function, division
import numpy as np
import matplotlib.pyplot as plt

def hist_intensity(ax, rpr):
    max_intensity = rpr.intensity.max()
    bins, bin_edges = np.histogram(rpr.intensity, bins=max_intensity)
    ax.set_yscale("log")
    ax.step(bin_edges[:-1], bins)
    ax.set_xlabel('p.e. in lixel #/1')
    ax.set_ylabel('number of lixels #/1')
