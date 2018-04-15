import numpy as np
import matplotlib.pyplot as plt


def add2ax_hist_arrival_time(
    sequence,
    time_slice_duration,
    ax,
    color='blue',
    start_time=0.0
):
    end_time = sequence.shape[0]*time_slice_duration
    arrival_times = np.linspace(start_time, end_time, sequence.shape[0])
    intensity_vs_time = np.sum(sequence, axis=1)
    ax.step(arrival_times, intensity_vs_time, color=color)
    ax.set_xlabel('arrival time t/s')
    ax.set_ylabel('p.e. #/1')


def add2ax_hist_intensity(sequence, ax, color='blue'):
    lixel_intensities = np.sum(sequence, axis=0)
    max_intensity = int(lixel_intensities.max())
    bins, bin_edges = np.histogram(lixel_intensities, bins=max_intensity)
    ax.set_yscale("log")
    ax.step(bin_edges[:-1], bins, color=color)
    ax.set_xlabel('p.e. in lixel #/1')
    ax.set_ylabel('number of lixels #/1')
