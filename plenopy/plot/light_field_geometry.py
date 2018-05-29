import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from .FigureSize import FigureSize
import os


def save_all(light_field_geometry, out_dir, figure_size=None):
    """
    Parameters
    ----------
    light_field_geometry    The light_field_geometry of the Atmospheric
                            Cherenkov Plenoscope (ACP).

    out_dir                 The output directory to save the figures in.

    figure_size             The output figure size and resolution.
                            [optional]
    """
    plotter = PlotLightFieldGeometry(
        light_field_geometry=light_field_geometry,
        out_dir=out_dir,
        figure_size=figure_size)
    plotter.save()


def symmertic_hists(vals_array, ax, bin_edges=None, color_array=None):
    '''
    Adds histograms of multiple value arrays to the same plotting axes. If not
    provided, the bin edges are estimated automatically and applied to all
    histograms.

    vals_array          An array of arrays to be histogramed

    ax                  Tha axes to add the histograms to

    bin_edges           1D array of the bin edge positions

    color_array         1D array of strings for marker colors
    '''
    if bin_edges is None:
        number_entires = [vals.shape[0] for vals in vals_array]
        minima = [vals.min() for vals in vals_array]
        maxima = [vals.max() for vals in vals_array]
        number_bins = int(np.floor(np.sqrt(np.max(number_entires))))
        bin_edges = np.linspace(np.min(minima), np.max(maxima), number_bins)

    for i, vals in enumerate(vals_array):
        bins, bin_esdges = np.histogram(vals, bins=bin_edges)
        bin_centers = 0.5 * (bin_esdges[1:] + bin_esdges[:-1])
        if color_array is None:
            color = None
        else:
            color = color_array[i]
        ax.step(bin_centers, bins, color=color)


def symmetric_hist(vals, ax, nbins=None):
    if nbins is None:
        nbins = int(np.floor(np.sqrt(vals.shape[0])))
    bins, bin_esdges = np.histogram(vals, bins=nbins)
    bin_centers = 0.5 * (bin_esdges[1:] + bin_esdges[:-1])
    ax.set_xlim([1.025 * bin_esdges[0], 1.025 * bin_esdges[-1]])
    ax.step(bin_centers, bins)


def cx_mean_hist(lss, ax):
    cx_mean = lss.cx_mean[lss.efficiency > 0.0]
    cx_mean = np.rad2deg(cx_mean)
    symmetric_hist(cx_mean, ax)
    ax.set_xlabel('incident-direction mean c' + 'x' + '/deg')


def cy_mean_hist(lss, ax):
    cy_mean = lss.cy_mean[lss.efficiency > 0.0]
    cy_mean = np.rad2deg(cy_mean)
    symmetric_hist(cy_mean, ax)
    ax.set_xlabel('incident-direction mean c' + 'y' + '/deg')


def x_mean_hist(lss, ax):
    x_mean = lss.x_mean[lss.efficiency > 0.0]
    symmetric_hist(x_mean, ax)
    ax.set_xlabel('support-position mean ' + 'x' + '/m')


def y_mean_hist(lss, ax):
    y_mean = lss.y_mean[lss.efficiency > 0.0]
    symmetric_hist(y_mean, ax)
    ax.set_xlabel('support-position mean ' + 'y' + '/m')


def cx_std_hist(lss, ax):
    cx_std = lss.cx_std[lss.efficiency > 0.0]
    cx_std = cx_std[~np.isnan(cx_std)]
    cx_std = np.rad2deg(cx_std)
    symmetric_hist(cx_std, ax)
    ax.set_xlabel('incident-direction stddev c' + 'x' + '/deg')


def cy_std_hist(lss, ax):
    cy_std = lss.cy_std[lss.efficiency > 0.0]
    cy_std = cy_std[~np.isnan(cy_std)]
    cy_std = np.rad2deg(cy_std)
    symmetric_hist(cy_std, ax)
    ax.set_xlabel('incident-direction stddev c' + 'y' + '/deg')


def x_std_hist(lss, ax):
    x_std = lss.x_std[lss.efficiency > 0.0]
    symmetric_hist(x_std, ax)
    ax.set_xlabel('support-position stddev ' + 'x' + '/m')


def y_std_hist(lss, ax):
    y_std = lss.y_std[lss.efficiency > 0.0]
    symmetric_hist(y_std, ax)
    ax.set_xlabel('support-position stddev ' + 'y' + '/m')


def time_mean_hist(lss, ax):
    time_delay_mean = lss.time_delay_mean[lss.efficiency > 0.0]
    symmetric_hist(time_delay_mean, ax)
    ax.set_xlabel('relative-arrival-time-delay mean t/s')


def time_std_hist(lss, ax):
    time_delay_std = lss.time_delay_std[lss.efficiency > 0.0]
    time_delay_std = time_delay_std[~np.isnan(time_delay_std)]
    symmetric_hist(time_delay_std, ax)
    ax.set_xlabel('relative-arrival-time-delay stddev t/s')


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
    nbins_x = int(np.floor(np.sqrt(x.shape[0])))
    nbins_y = int(np.floor(np.sqrt(y.shape[0])))
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
    ax.set_xlabel('support-position x/m')
    ax.set_ylabel('support-position y/m')


def cx_cy_hist2d(lss, ax):
    valid_cx = ~np.isnan(lss.cx_mean)
    valid_cy = ~np.isnan(lss.cy_mean)
    valid = valid_cx * valid_cy
    cx_pos = np.rad2deg(lss.cx_mean[valid])
    cy_pos = np.rad2deg(lss.cy_mean[valid])

    hist_2d(cx_pos, cy_pos, ax, aspect='equal')
    ax.set_xlabel('cx/deg')
    ax.set_ylabel('cy/deg')


def colored_lixels(
    lss,
    I,
    ax,
    cmap='viridis',
    vmin=None,
    vmax=None,
    edgecolors='none'
):
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
        edgecolors=edgecolors,
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


class PlotLightFieldGeometry(object):
    def __init__(self, light_field_geometry, out_dir, figure_size=None):
        if figure_size is None:
            self.figure_size = FigureSize(
                relative_width=16,
                relative_hight=9,
                pixel_rows=2*1080,
                dpi=300)
        else:
            self.figure_size = figure_size

        self.lfg = light_field_geometry
        self.out_dir = out_dir

    def __style(self, ax):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()

    def fig_ax(self):
        fig = plt.figure(
            figsize=(self.figure_size.width, self.figure_size.hight))
        ax = fig.gca()
        self.__style(ax)
        return fig, ax

    def __save_fig(self, fig, filename):
        fig.savefig(
            os.path.join(self.out_dir, filename),
            bbox_inches='tight',
            dpi=self.figure_size.dpi)

    def save_cx_mean(self):
        fig, ax = self.fig_ax()
        cx_mean_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cx_mean.png')
        plt.close(fig)

    def save_cy_mean(self):
        fig, ax = self.fig_ax()
        cy_mean_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cy_mean.png')
        plt.close(fig)

    def save_x_mean(self):
        fig, ax = self.fig_ax()
        x_mean_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'x_mean.png')
        plt.close(fig)

    def save_x_stddev(self):
        fig, ax = self.fig_ax()
        x_std_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'x_stddev.png')
        plt.close(fig)

    def save_y_mean(self):
        fig, ax = self.fig_ax()
        y_mean_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'y_mean.png')
        plt.close(fig)

    def save_y_stddev(self):
        fig, ax = self.fig_ax()
        y_std_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'y_stddev.png')
        plt.close(fig)

    def save_cx_stddev(self):
        fig, ax = self.fig_ax()
        cx_std_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cx_stddev.png')
        plt.close(fig)

    def save_cy_stddev(self):
        fig, ax = self.fig_ax()
        cy_std_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cy_stddev.png')
        plt.close(fig)

    def save_time_mean(self):
        fig, ax = self.fig_ax()
        time_mean_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'time_delay_mean.png')
        plt.close(fig)

    def save_time_stddev(self):
        fig, ax = self.fig_ax()
        time_std_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'time_stddev.png')
        plt.close(fig)

    def save_efficiency(self):
        fig, ax = self.fig_ax()
        efficieny_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'efficiency.png')
        plt.close(fig)

    def save_efficiency_relative_error(self):
        fig, ax = self.fig_ax()
        efficieny_relative_error_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'efficiency_error.png')
        plt.close(fig)

    def save_c_mean_vs_c_std(self):
        fig, ax = self.fig_ax()
        c_vs_c_std(self.lfg, ax)
        self.__save_fig(fig, 'c_mean_vs_c_std.png')
        plt.close(fig)

    def save_x_y_hist2d(self):
        fig, ax = self.fig_ax()
        x_y_hist2d(self.lfg, ax)
        self.__save_fig(fig, 'x_y_mean_hist2d.png')
        plt.close(fig)

    def save_cx_cy_hist2d(self):
        fig, ax = self.fig_ax()
        cx_cy_hist2d(self.lfg, ax)
        self.__save_fig(fig, 'cx_cy_mean_hist2d.png')
        plt.close(fig)

    def save_sensor_plane_overview(self, I, name='unknown', unit='unknown'):
        fig, ax = self.fig_ax()
        coll = colored_lixels(self.lfg, I, ax)
        ax.set_ylabel('sensor plane y/m')
        ax.set_xlabel('sensor plane x/m')
        fig.colorbar(coll, label='support-position ' + name + '/' + unit)
        self.__save_fig(fig, 'overview_' + name + '.png')
        # zoom center
        outer_radius = 1.0 / np.sqrt(2.0) * np.hypot(
            self.lfg.lixel_positions_x.max(),
            self.lfg.lixel_positions_y.max()
        )
        zoom_radius = 1.0 / 10.0 * outer_radius
        ax.set_ylim([-zoom_radius, zoom_radius])
        ax.set_xlim([-zoom_radius, zoom_radius])
        self.__save_fig(fig, 'overview_' + name + '_zoom_center.png')
        # zoom pos x
        ax.set_ylim([-zoom_radius, zoom_radius])
        ax.set_xlim([
            0.95 * outer_radius - zoom_radius,
            0.95 * outer_radius + zoom_radius])
        self.__save_fig(fig, 'overview_' + name + '_zoom_pos_x.png')
        # zoom pos y
        ax.set_ylim([
            0.95 * outer_radius - zoom_radius,
            0.95 * outer_radius + zoom_radius])
        ax.set_xlim([-zoom_radius, zoom_radius])
        self.__save_fig(fig, 'overview_' + name + '_zoom_pos_y.png')
        plt.close(fig)

    def save(self):
        os.makedirs(self.out_dir, exist_ok=True)

        jobs = []
        jobs.append({'target': self.save_cx_mean, 'args': []})
        jobs.append({'target': self.save_cy_mean, 'args': []})

        jobs.append({'target': self.save_x_mean, 'args': []})
        jobs.append({'target': self.save_y_mean, 'args': []})

        jobs.append({'target': self.save_cx_stddev, 'args': []})
        jobs.append({'target': self.save_cy_stddev, 'args': []})

        jobs.append({'target': self.save_x_stddev, 'args': []})
        jobs.append({'target': self.save_y_stddev, 'args': []})

        jobs.append({'target': self.save_time_mean, 'args': []})
        jobs.append({'target': self.save_time_stddev, 'args': []})

        jobs.append({'target': self.save_efficiency, 'args': []})
        jobs.append(
            {'target': self.save_efficiency_relative_error, 'args': []})

        jobs.append({'target': self.save_c_mean_vs_c_std, 'args': []})
        jobs.append({'target': self.save_x_y_hist2d, 'args': []})
        jobs.append({'target': self.save_cx_cy_hist2d, 'args': []})

        jobs.append({
            'target': self.save_sensor_plane_overview,
            'args': [self.lfg.time_delay_mean, 'time_delay_to_pap', 's']})
        jobs.append({
            'target': self.save_sensor_plane_overview,
            'args': [
                self.lfg.time_delay_image_mean, 'time_delay_to_img', 's']})
        jobs.append({
            'target': self.save_sensor_plane_overview,
            'args': [self.lfg.efficiency, 'efficiency', '1']})
        jobs.append({
            'target': self.save_sensor_plane_overview,
            'args': [self.lfg.x_mean, 'x_mean', 'm']})
        jobs.append({
            'target': self.save_sensor_plane_overview,
            'args': [self.lfg.x_std, 'x_stddev', 'm']})
        jobs.append({
            'target': self.save_sensor_plane_overview,
            'args': [self.lfg.y_mean, 'y_mean', 'm']})
        jobs.append({
            'target': self.save_sensor_plane_overview,
            'args': [self.lfg.y_std, 'y_stddev', 'm']})

        jobs.append({
            'target': self.save_sensor_plane_overview,
            'args': [np.rad2deg(self.lfg.cx_mean), 'cx_mean', 'deg']})
        jobs.append({
            'target': self.save_sensor_plane_overview,
            'args': [np.rad2deg(self.lfg.cx_std), 'cx_stddev', 'deg']})
        jobs.append({
            'target': self.save_sensor_plane_overview,
            'args': [np.rad2deg(self.lfg.cy_mean), 'cy_mean', 'deg']})
        jobs.append({
            'target': self.save_sensor_plane_overview,
            'args': [np.rad2deg(self.lfg.cy_std), 'cy_stddev', 'deg']})

        for job in jobs:
            job['target'](*job['args'])
