import numpy as np
import matplotlib.pyplot as plt
import os
from . import plot as add2ax
from multiprocessing import Process
from ..tools import FigureSize

class PlotLightFieldGeometry(object):
    """
    Creates plots of the light_field_geometry and saves them to disk.
    """
    def __init__(self, light_field_geometry, out_dir, figure_size=None):
        """
        Parameters
        ----------
        light_field_geometry    The light_field_geometry of the Atmospheric
                                Cherenkov Plenoscope (ACP).

        out_dir                 The output directory to save the figures in.

        figure_size             The output figure size and resolution.
                                [optional]
        """
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
        add2ax.cx_mean_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cx_mean.png')
        plt.close(fig)

    def save_cy_mean(self):
        fig, ax = self.fig_ax()
        add2ax.cy_mean_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cy_mean.png')
        plt.close(fig)

    def save_x_mean(self):
        fig, ax = self.fig_ax()
        add2ax.x_mean_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'x_mean.png')
        plt.close(fig)

    def save_x_stddev(self):
        fig, ax = self.fig_ax()
        add2ax.x_std_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'x_stddev.png')
        plt.close(fig)

    def save_y_mean(self):
        fig, ax = self.fig_ax()
        add2ax.y_mean_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'y_mean.png')
        plt.close(fig)

    def save_y_stddev(self):
        fig, ax = self.fig_ax()
        add2ax.y_std_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'y_stddev.png')
        plt.close(fig)

    def save_cx_stddev(self):
        fig, ax = self.fig_ax()
        add2ax.cx_std_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cx_stddev.png')
        plt.close(fig)

    def save_cy_stddev(self):
        fig, ax = self.fig_ax()
        add2ax.cy_std_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cy_stddev.png')
        plt.close(fig)

    def save_time_mean(self):
        fig, ax = self.fig_ax()
        add2ax.time_mean_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'time_delay_mean.png')
        plt.close(fig)

    def save_time_stddev(self):
        fig, ax = self.fig_ax()
        add2ax.time_std_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'time_stddev.png')
        plt.close(fig)

    def save_efficiency(self):
        fig, ax = self.fig_ax()
        add2ax.efficieny_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'efficiency.png')
        plt.close(fig)

    def save_efficiency_relative_error(self):
        fig, ax = self.fig_ax()
        add2ax.efficieny_relative_error_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'efficiency_error.png')
        plt.close(fig)

    def save_c_mean_vs_c_std(self):
        fig, ax = self.fig_ax()
        add2ax.c_vs_c_std(self.lfg, ax)
        self.__save_fig(fig, 'c_mean_vs_c_std.png')
        plt.close(fig)

    def save_x_y_hist2d(self):
        fig, ax = self.fig_ax()
        add2ax.x_y_hist2d(self.lfg, ax)
        self.__save_fig(fig, 'x_y_mean_hist2d.png')
        plt.close(fig)

    def save_cx_cy_hist2d(self):
        fig, ax = self.fig_ax()
        add2ax.cx_cy_hist2d(self.lfg, ax)
        self.__save_fig(fig, 'cx_cy_mean_hist2d.png')
        plt.close(fig)

    def save_sensor_plane_overview(self, I, name='unknown', unit='unknown'):
        fig, ax = self.fig_ax()
        coll = add2ax.colored_lixels(self.lfg, I, ax)
        ax.set_ylabel('sensor plane y/m')
        ax.set_xlabel('sensor plane x/m')
        fig.colorbar(coll, label='principal aperture ' + name + '/' + unit)
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
        ax.set_xlim(
            [0.95 * outer_radius - zoom_radius, 0.95 * outer_radius + zoom_radius])
        self.__save_fig(fig, 'overview_' + name + '_zoom_pos_x.png')
        # zoom pos y
        ax.set_ylim(
            [0.95 * outer_radius - zoom_radius, 0.95 * outer_radius + zoom_radius])
        ax.set_xlim([-zoom_radius, zoom_radius])
        self.__save_fig(fig, 'overview_' + name + '_zoom_pos_y.png')
        plt.close(fig)

    def save(self):

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
        jobs.append({'target': self.save_efficiency_relative_error, 'args': []})

        jobs.append({'target': self.save_c_mean_vs_c_std, 'args': []})
        jobs.append({'target': self.save_x_y_hist2d, 'args': []})
        jobs.append({'target': self.save_cx_cy_hist2d, 'args': []})

        jobs.append({'target': self.save_sensor_plane_overview, 'args': [self.lfg.efficiency, 'efficiency', '1']})

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

        processes = []
        for job in jobs:
            processes.append(Process(target=job['target'], args=job['args']))
        for process in processes:
            process.start()
        for process in processes:
            process.join()
