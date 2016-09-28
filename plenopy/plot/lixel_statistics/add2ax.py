import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def symmetric_hist(vals, ax, nbins=None):
    if nbins == None:
        nbins = np.floor(np.sqrt(vals.shape[0]))
    bins, bin_esdges = np.histogram(vals, bins=nbins)
    bin_centers = 0.5 * (bin_esdges[1:] + bin_esdges[:-1])
    ax.set_xlim([1.025 * bin_esdges[0], 1.025 * bin_esdges[-1]])
    ax.step(bin_centers, bins)


def cx_mean_hist(lss, ax):
    cx_mean = lss.cx_mean[lss.efficiency > 0.0]
    cx_mean = np.rad2deg(cx_mean)
    symmetric_hist(cx_mean, ax)
    ax.set_xlabel('incoming direction mean c' + 'x' + '/deg')


def cy_mean_hist(lss, ax):
    cy_mean = lss.cy_mean[lss.efficiency > 0.0]
    cy_mean = np.rad2deg(cy_mean)
    symmetric_hist(cy_mean, ax)
    ax.set_xlabel('incoming direction mean c' + 'y' + '/deg')


def x_mean_hist(lss, ax):
    x_mean = lss.x_mean[lss.efficiency > 0.0]
    symmetric_hist(x_mean, ax)
    ax.set_xlabel('principal apertur mean ' + 'x' + '/m')


def y_mean_hist(lss, ax):
    y_mean = lss.y_mean[lss.efficiency > 0.0]
    symmetric_hist(y_mean, ax)
    ax.set_xlabel('principal apertur mean ' + 'y' + '/m')


def cx_std_hist(lss, ax):
    cx_std = lss.cx_std[lss.efficiency > 0.0]
    cx_std = cx_std[~np.isnan(cx_std)]
    cx_std = np.rad2deg(cx_std)
    symmetric_hist(cx_std, ax)
    ax.set_xlabel('incoming direction stddev c' + 'x' + '/deg')


def cy_std_hist(lss, ax):
    cy_std = lss.cy_std[lss.efficiency > 0.0]
    cy_std = cy_std[~np.isnan(cy_std)]
    cy_std = np.rad2deg(cy_std)
    symmetric_hist(cy_std, ax)
    ax.set_xlabel('incoming direction stddev c' + 'y' + '/deg')


def time_mean_hist(lss, ax):
    time_delay_mean = lss.time_delay_mean[lss.efficiency > 0.0]
    symmetric_hist(time_delay_mean, ax)
    ax.set_xlabel('relative arrival time mean t/s')


def time_std_hist(lss, ax):
    time_delay_std = lss.time_delay_std[lss.efficiency > 0.0]
    time_delay_std = time_delay_std[~np.isnan(time_delay_std)]
    symmetric_hist(time_delay_std, ax)
    ax.set_xlabel('relative arrival time stddev t/s')


def efficieny_hist(lss, ax):
    eff = lss.efficiency
    symmetric_hist(eff, ax)
    ax.set_xlabel('efficiency eff/1')
    ax.set_ylabel('number of lixel #/1')


def efficieny_relative_error_hist(lss, ax):
    rel_error = lss.efficiency_std[lss.efficiency >
                                   0.0] / lss.efficiency[lss.efficiency > 0.0]
    symmetric_hist(rel_error, ax)
    ax.set_xlabel('relative error efficiency')
    ax.set_ylabel('number of lixel #/1')


def hist_2d(x, y, ax, aspect='auto'):
    nbins_x = np.sqrt(x.shape[0])
    nbins_y = np.sqrt(y.shape[0])
    bins, xedges, yedges = np.histogram2d(x, y, bins=[nbins_x, nbins_y])
    im = ax.imshow(
        bins.T,
        interpolation='none',
        origin='low',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect=aspect
    )
    im.set_cmap('viridis')


def c_vs_c_std(lss, ax):
    c_mean = np.hypot(lss.cx_mean, lss.cy_mean)
    c_mean = np.rad2deg(c_mean)
    c_mean_valid = ~np.isnan(c_mean)

    c_std = np.hypot(lss.cx_std, lss.cy_std)
    c_std = np.rad2deg(c_std)
    c_std_valid = ~np.isnan(c_std)

    valid = c_mean_valid * c_std_valid
    hist_2d(c_mean[valid], c_std[valid], ax)
    ax.set_xlabel('c mean/deg')
    ax.set_ylabel('c std/deg')


def x_y_hist2d(lss, ax):
    valid_x = ~np.isnan(lss.x_mean)
    valid_y = ~np.isnan(lss.y_mean)
    valid = valid_x * valid_y
    x_pos = lss.x_mean[valid]
    y_pos = lss.y_mean[valid]

    hist_2d(x_pos, y_pos, ax, aspect='equal')
    ax.set_xlabel('principal aperture x/m')
    ax.set_ylabel('principal aperture y/m')


def cx_cy_hist2d(lss, ax):
    valid_cx = ~np.isnan(lss.cx_mean)
    valid_cy = ~np.isnan(lss.cy_mean)
    valid = valid_cx * valid_cy
    cx_pos = np.rad2deg(lss.cx_mean[valid])
    cy_pos = np.rad2deg(lss.cy_mean[valid])

    hist_2d(cx_pos, cy_pos, ax, aspect='equal')
    ax.set_xlabel('cx/deg')
    ax.set_ylabel('cy/deg')


def colored_lixels(lss, I, ax, cmap='viridis', vmin=None, vmax=None):
    I = I.flatten()
    valid = ~np.isnan(I)
    valid_I = I[valid]

    valid_polygons = []
    for i, poly in enumerate(lss.lixel_polygons):
        if valid[i]:
            valid_polygons.append(poly)

    coll = PolyCollection(
        valid_polygons,
        array=valid_I,
        cmap=cmap,
        edgecolors='none',
    )
    if vmin is None:
        vmin = valid_I.min()
    if vmax is None:
        vmax = valid_I.max()
    coll.set_clim([vmin, vmax])
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.set_aspect('equal')
    return coll  # to set colorbar
