import numpy as np
import matplotlib.pyplot as plt


def add2ax_hist_arrival_time(light_field_sequence, ax, color='blue'):
    lfs = light_field_sequence
        
    start_time = 0.0
    end_time = lfs.number_time_slices*lfs.time_slice_duration
    arrival_times = np.linspace(start_time, end_time, lfs.number_time_slices)
    intensity_vs_time = np.sum(lfs.sequence, axis=1)
    ax.step(arrival_times, intensity_vs_time, color=color)
    ax.set_xlabel('arrival time t/s')
    ax.set_ylabel('p.e. #/1')


def add2ax_hist_intensity(light_field_sequence, ax, color='blue'):
    lfs = light_field_sequence    
    lixel_intensities = np.sum(lfs.sequence, axis=0)
    max_intensity = int(lixel_intensities.max())
    bins, bin_edges = np.histogram(lixel_intensities, bins=max_intensity)
    ax.set_yscale("log")
    ax.step(bin_edges[:-1], bins, color=color)
    ax.set_xlabel('p.e. in lixel #/1')
    ax.set_ylabel('number of lixels #/1')
