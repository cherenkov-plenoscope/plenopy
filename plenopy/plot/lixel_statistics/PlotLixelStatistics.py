#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, print_function, division
import numpy as np
import matplotlib.pyplot as plt
import os
from . import add2ax


class PlotLixelStatistics(object):

    def __init__(self, lss, path):
        self.dpi = 300
        self.width = 2.0 * 1920.0 / self.dpi
        self.hight = 2.0 * 1080.0 / self.dpi
        self.lss = lss
        self.path = path

    def __style(self, ax):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()

    def fig_ax(self):
        fig = plt.figure(figsize=(self.width, self.hight))
        ax = fig.gca()
        self.__style(ax)
        return fig, ax

    def __save_fig(self, fig, filename):
        fig.savefig(
            os.path.join(self.path, filename),
            bbox_inches='tight',
            dpi=self.dpi)

    def save_cx_mean(self):
        fig, ax = self.fig_ax()
        add2ax.cx_mean_hist(self.lss, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cx_mean.png')
        plt.close(fig)

    def save_cy_mean(self):
        fig, ax = self.fig_ax()
        add2ax.cy_mean_hist(self.lss, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cy_mean.png')
        plt.close(fig)

    def save_x_mean(self):
        fig, ax = self.fig_ax()
        add2ax.x_mean_hist(self.lss, ax)
        ax.semilogy()
        self.__save_fig(fig, 'x_mean.png')
        plt.close(fig)

    def save_y_mean(self):
        fig, ax = self.fig_ax()
        add2ax.y_mean_hist(self.lss, ax)
        ax.semilogy()
        self.__save_fig(fig, 'y_mean.png')
        plt.close(fig)

    def save_cx_stddev(self):
        fig, ax = self.fig_ax()
        add2ax.cx_std_hist(self.lss, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cx_stddev.png')
        plt.close(fig)

    def save_cy_stddev(self):
        fig, ax = self.fig_ax()
        add2ax.cy_std_hist(self.lss, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cy_stddev.png')
        plt.close(fig)

    def save_time_mean(self):
        fig, ax = self.fig_ax()
        add2ax.time_mean_hist(self.lss, ax)
        ax.semilogy()
        self.__save_fig(fig, 'time_delay_mean.png')
        plt.close(fig)

    def save_time_stddev(self):
        fig, ax = self.fig_ax()
        add2ax.time_std_hist(self.lss, ax)
        ax.semilogy()
        self.__save_fig(fig, 'time_stddev.png')
        plt.close(fig)

    def save_efficiency(self):
        fig, ax = self.fig_ax()
        add2ax.efficieny_hist(self.lss, ax)
        ax.semilogy()
        self.__save_fig(fig, 'efficiency.png')
        plt.close(fig)

    def save_efficiency_relative_error(self):
        fig, ax = self.fig_ax()
        add2ax.efficieny_relative_error_hist(self.lss, ax)
        ax.semilogy()
        self.__save_fig(fig, 'efficiency_error.png')
        plt.close(fig)

    def save_c_mean_vs_c_std(self):
        fig, ax = self.fig_ax()
        add2ax.c_vs_c_std(self.lss, ax)
        self.__save_fig(fig, 'c_mean_vs_c_std.png')
        plt.close(fig)

    def save_x_y_hist2d(self):
        fig, ax = self.fig_ax()
        add2ax.x_y_hist2d(self.lss, ax)
        self.__save_fig(fig, 'x_y_mean_hist2d.png')
        plt.close(fig)

    def save_cx_cy_hist2d(self):
        fig, ax = self.fig_ax()
        add2ax.cx_cy_hist2d(self.lss, ax)
        self.__save_fig(fig, 'cx_cy_mean_hist2d.png')
        plt.close(fig)

    def save_sensor_plane_overview(self, I, name='unknown', unit='unknown'):
        fig, ax = self.fig_ax()
        coll = add2ax.colored_lixels(self.lss, I, ax)
        ax.set_ylabel('sensor plane y/m')
        ax.set_xlabel('sensor plane x/m')
        fig.colorbar(coll, label='principal aperture ' + name + '/' + unit)
        self.__save_fig(fig, 'overview_' + name + '.png')
        # zoom center
        outer_radius = 1.0 / np.sqrt(2.0) * np.hypot(
            self.lss.lixel_positions_x.max(),
            self.lss.lixel_positions_y.max()
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
        self.save_cx_mean()
        self.save_cy_mean()
        self.save_x_mean()
        self.save_y_mean()
        self.save_cx_stddev()
        self.save_cy_stddev()
        self.save_time_mean()
        self.save_time_stddev()
        self.save_efficiency()
        self.save_efficiency_relative_error()
        self.save_c_mean_vs_c_std()
        self.save_x_y_hist2d()
        self.save_cx_cy_hist2d()
        self.save_sensor_plane_overview(self.lss.efficiency, 'efficiency', '1')
        self.save_sensor_plane_overview(self.lss.x_mean, 'x_mean', 'm')
        self.save_sensor_plane_overview(self.lss.x_std, 'x_stddev', 'm')
        self.save_sensor_plane_overview(self.lss.y_mean, 'y_mean', 'm')
        self.save_sensor_plane_overview(self.lss.y_std, 'y_stddev', 'm')
        self.save_sensor_plane_overview(
            np.rad2deg(self.lss.cx_mean), 'cx_mean', 'deg')
        self.save_sensor_plane_overview(
            np.rad2deg(self.lss.cx_std), 'cx_stddev', 'deg')
        self.save_sensor_plane_overview(
            np.rad2deg(self.lss.cy_mean), 'cy_mean', 'deg')
        self.save_sensor_plane_overview(
            np.rad2deg(self.lss.cy_std), 'cy_stddev', 'deg')
