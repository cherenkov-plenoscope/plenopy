#!/usr/bin/env python
from __future__ import absolute_import, print_function, division
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from PlotTools import add_pixel_image_to_ax
from PlotTools import add_paxel_image_to_ax


def save_sum_projections(event, path, threshold=0):
    # avoid side effects
    lf = deepcopy(event.light_field)
    lf.intensity[lf.intensity < threshold] = 0

    plt.rcParams.update({'font.size': 20})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, (ax_dir, ax_pap) = plt.subplots(1, 2, figsize=(16, 9))
    plt.suptitle(event.mc_truth.short_event_info())

    add_pixel_image_to_ax(lf.pixel_sum(), ax_dir)

    lf.intensity[:, lf.paxel_eff <= 0.4] = 0
    add_paxel_image_to_ax(lf.paxel_sum(), ax_pap)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(os.path.splitext(path)[0] + '.png', dpi=120)
    plt.close()
