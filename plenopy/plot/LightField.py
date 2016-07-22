#!/usr/bin/env python
from __future__ import absolute_import, print_function, division
import numpy as np
import matplotlib.pyplot as plt


def add2ax_hist_arrival_time(light_field, ax, color='blue'):
    bins, bin_edges = np.histogram(
        light_field.arrival_time[light_field.valid_lixel],
        weights=light_field.intensity[light_field.valid_lixel],
        bins=int(np.ceil(0.5 * np.sqrt(light_field.valid_lixel.sum()))))
    ax.step(bin_edges[:-1], bins, color=color)
    ax.set_xlabel('arrival time t/s')
    ax.set_ylabel('p.e. #/1')


def add2ax_hist_intensity(light_field, ax, color='blue'):
    max_intensity = int(light_field.intensity.flatten().max())
    bins, bin_edges = np.histogram(
        light_field.intensity.flatten(), bins=max_intensity)
    ax.set_yscale("log")
    ax.step(bin_edges[:-1], bins, color=color)
    ax.set_xlabel('p.e. in lixel #/1')
    ax.set_ylabel('number of lixels #/1')
