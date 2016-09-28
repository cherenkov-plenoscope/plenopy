import numpy as np
import matplotlib.pyplot as plt


def add2ax_hist_intensity(raw_response, ax):
    max_intensity = raw_response.intensity.max()
    bins, bin_edges = np.histogram(raw_response.intensity, bins=max_intensity)
    ax.set_yscale("log")
    ax.step(bin_edges[:-1], bins)
    ax.set_xlabel('p.e. in lixel #/1')
    ax.set_ylabel('number of lixels #/1')
