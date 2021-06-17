import numpy as np
import sebastians_matplotlib_addons as splt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import PolyCollection
import os


def save_all(light_field_geometry, out_dir, figure_style=splt.FIGURE_16_9):
    """
    Parameters
    ----------
    light_field_geometry    The light_field_geometry of the instrument.

    out_dir                 The output directory to save the figures in.

    figure_style            The output figure size and resolution.
    """
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}') # for \mathrm{}
    plt.rc('font', family='serif')

    plotter = PlotLightFieldGeometry(
        light_field_geometry=light_field_geometry,
        out_dir=out_dir,
        figure_style=figure_style)
    plotter.save()


def symmetric_hist(vals, ax, nbins=None):
    if nbins is None:
        nbins = int(np.floor(np.sqrt(vals.shape[0])))
    bins, bin_esdges = np.histogram(vals, bins=nbins)
    bin_centers = 0.5 * (bin_esdges[1:] + bin_esdges[:-1])
    ax.set_xlim([1.025 * bin_esdges[0], 1.025 * bin_esdges[-1]])
    ax.step(bin_centers, bins, color='k')
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)


def cx_mean_hist(lss, ax):
    cx_mean = lss.cx_mean[lss.efficiency > 0.0]
    cx_mean = np.rad2deg(cx_mean)
    symmetric_hist(cx_mean, ax)
    ax.set_xlabel(r'$\overline{c_x}$/$^\circ$')


def cy_mean_hist(lss, ax):
    cy_mean = lss.cy_mean[lss.efficiency > 0.0]
    cy_mean = np.rad2deg(cy_mean)
    symmetric_hist(cy_mean, ax)
    ax.set_xlabel(r'$\overline{c_y}$/$^\circ$')


def x_mean_hist(lss, ax):
    x_mean = lss.x_mean[lss.efficiency > 0.0]
    symmetric_hist(x_mean, ax)
    ax.set_xlabel(r'$\overline{x}$/m')


def y_mean_hist(lss, ax):
    y_mean = lss.y_mean[lss.efficiency > 0.0]
    symmetric_hist(y_mean, ax)
    ax.set_xlabel(r'$\overline{y}$/m')


def cx_std_hist(lss, ax):
    cx_std = lss.cx_std[lss.efficiency > 0.0]
    cx_std = cx_std[~np.isnan(cx_std)]
    cx_std = np.rad2deg(cx_std)
    symmetric_hist(cx_std, ax)
    ax.set_xlabel(r'$\sigma_{c_x}$/$^\circ$')


def cy_std_hist(lss, ax):
    cy_std = lss.cy_std[lss.efficiency > 0.0]
    cy_std = cy_std[~np.isnan(cy_std)]
    cy_std = np.rad2deg(cy_std)
    symmetric_hist(cy_std, ax)
    ax.set_xlabel(r'$\sigma_{c_y}$/$^\circ$')


def x_std_hist(lss, ax):
    x_std = lss.x_std[lss.efficiency > 0.0]
    symmetric_hist(x_std, ax)
    ax.set_xlabel(r'$\sigma_{x}$/m')


def y_std_hist(lss, ax):
    y_std = lss.y_std[lss.efficiency > 0.0]
    symmetric_hist(y_std, ax)
    ax.set_xlabel(r'$\sigma_{y}$/m')


def time_mean_hist(lss, ax):
    time_delay_mean = lss.time_delay_mean[lss.efficiency > 0.0]
    symmetric_hist(time_delay_mean, ax)
    ax.set_xlabel(r'$\overline{t}_\mathrm{pap}$/s')


def time_std_hist(lss, ax):
    time_delay_std = lss.time_delay_std[lss.efficiency > 0.0]
    time_delay_std = time_delay_std[~np.isnan(time_delay_std)]
    symmetric_hist(time_delay_std, ax)
    ax.set_xlabel(r'${\sigma_t}_\mathrm{pap}$/s')


def efficieny_hist(lss, ax):
    eff = lss.efficiency
    symmetric_hist(eff, ax)
    ax.set_xlabel(r'$\overline{\eta}$/1')


def efficieny_relative_error_hist(lss, ax):
    rel_error = lss.efficiency_std[lss.efficiency >
                                   0.0] / lss.efficiency[lss.efficiency > 0.0]
    symmetric_hist(rel_error, ax)
    ax.set_xlabel(r'$\sigma_\eta$/1')


def hist_2d(x, y, ax, aspect='auto', norm=None):
    nbins_x = int(np.floor(np.sqrt(x.shape[0])))
    nbins_y = int(np.floor(np.sqrt(y.shape[0])))
    bins, xedges, yedges = np.histogram2d(x, y, bins=[nbins_x, nbins_y])
    im = ax.imshow(
        bins.T,
        interpolation='none',
        origin='lower',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect=aspect,
        norm=norm
    )
    im.set_cmap('Greys')


def c_vs_c_std(lss, ax):
    c_mean = np.hypot(lss.cx_mean, lss.cy_mean)
    c_mean = np.rad2deg(c_mean)
    c_mean_valid = ~np.isnan(c_mean)

    c_std = np.hypot(lss.cx_std, lss.cy_std)
    c_std = np.rad2deg(c_std)
    c_std_valid = ~np.isnan(c_std)

    valid = c_mean_valid * c_std_valid
    hist_2d(
        c_mean[valid],
        c_std[valid],
        ax,
        norm=colors.PowerNorm(gamma=1/2))
    ax.set_xlabel(r'$\sqrt{\overline{c_x}^2 + \overline{c_y}^2}$/$^\circ$')
    ax.set_ylabel(r'$\sqrt{\sigma_{c_x}^2 + \sigma_{c_y}^2}$/$^\circ$')


def x_y_hist2d(lss, ax):
    valid_x = ~np.isnan(lss.x_mean)
    valid_y = ~np.isnan(lss.y_mean)
    valid = valid_x * valid_y
    x_pos = lss.x_mean[valid]
    y_pos = lss.y_mean[valid]

    hist_2d(
        x_pos,
        y_pos,
        ax,
        aspect='equal',
        norm=colors.PowerNorm(gamma=1/2))
    ax.set_xlabel(r'$\overline{x}$/m')
    ax.set_ylabel(r'$\overline{y}$/m')


def cx_cy_hist2d(lss, ax):
    valid_cx = ~np.isnan(lss.cx_mean)
    valid_cy = ~np.isnan(lss.cy_mean)
    valid = valid_cx * valid_cy
    cx_pos = np.rad2deg(lss.cx_mean[valid])
    cy_pos = np.rad2deg(lss.cy_mean[valid])

    hist_2d(
        cx_pos,
        cy_pos,
        ax,
        aspect='equal',
        norm=colors.PowerNorm(gamma=1/2))
    ax.set_xlabel(r'$\overline{c_x}$/$^\circ$')
    ax.set_ylabel(r'$\overline{c_y}$/$^\circ$')


def colored_lixels(
    lss,
    I,
    ax,
    cmap='Greys',
    vmin=None,
    vmax=None,
    edgecolors='none',
    linewidths=None
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
        linewidths=linewidths,
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
    def __init__(self, light_field_geometry, out_dir, figure_style):
        self.figure_style = figure_style
        self.lfg = light_field_geometry
        self.out_dir = out_dir

    def fig_ax(self):
        fig = splt.figure(self.figure_style)
        ax = splt.add_axes(fig=fig, span=[0.15, 0.15, 0.8, 0.8])
        return fig, ax

    def __save_fig(self, fig, filename):
        fig.savefig(os.path.join(self.out_dir, filename))

    def save_cx_mean(self):
        fig, ax = self.fig_ax()
        cx_mean_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cx_mean.jpg')
        plt.close(fig)

    def save_cy_mean(self):
        fig, ax = self.fig_ax()
        cy_mean_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cy_mean.jpg')
        plt.close(fig)

    def save_x_mean(self):
        fig, ax = self.fig_ax()
        x_mean_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'x_mean.jpg')
        plt.close(fig)

    def save_x_stddev(self):
        fig, ax = self.fig_ax()
        x_std_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'x_std.jpg')
        plt.close(fig)

    def save_y_mean(self):
        fig, ax = self.fig_ax()
        y_mean_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'y_mean.jpg')
        plt.close(fig)

    def save_y_stddev(self):
        fig, ax = self.fig_ax()
        y_std_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'y_std.jpg')
        plt.close(fig)

    def save_cx_stddev(self):
        fig, ax = self.fig_ax()
        cx_std_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cx_std.jpg')
        plt.close(fig)

    def save_cy_stddev(self):
        fig, ax = self.fig_ax()
        cy_std_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'cy_std.jpg')
        plt.close(fig)

    def save_time_mean(self):
        fig, ax = self.fig_ax()
        time_mean_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 't_mean.jpg')
        plt.close(fig)

    def save_time_stddev(self):
        fig, ax = self.fig_ax()
        time_std_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 't_std.jpg')
        plt.close(fig)

    def save_efficiency(self):
        fig, ax = self.fig_ax()
        efficieny_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'eta_mean.jpg')
        plt.close(fig)

    def save_efficiency_relative_error(self):
        fig, ax = self.fig_ax()
        efficieny_relative_error_hist(self.lfg, ax)
        ax.semilogy()
        self.__save_fig(fig, 'eta_std.jpg')
        plt.close(fig)

    def save_c_mean_vs_c_std(self):
        fig, ax = self.fig_ax()
        c_vs_c_std(self.lfg, ax)
        self.__save_fig(fig, 'c_mean_vs_c_std.jpg')
        plt.close(fig)

    def save_x_y_hist2d(self):
        fig, ax = self.fig_ax()
        x_y_hist2d(self.lfg, ax)
        self.__save_fig(fig, 'x_y_mean_hist2d.jpg')
        plt.close(fig)

    def save_cx_cy_hist2d(self):
        fig, ax = self.fig_ax()
        cx_cy_hist2d(self.lfg, ax)
        self.__save_fig(fig, 'cx_cy_mean_hist2d.jpg')
        plt.close(fig)


    def save_sensor_plane_overview(
        self,
        I,
        name,
        unit,
        simple_name,
        indicate_zoom_region_center=True,
        indicate_zoom_region_pos_x=True,
        indicate_zoom_region_pos_y=True,
    ):
        fig, ax = self.fig_ax()
        coll = colored_lixels(self.lfg, I, ax)
        ax.set_ylabel(r'sensor-plane $y$/m')
        ax.set_xlabel(r'sensor-plane $x$/m')
        fig.colorbar(coll, label=name + '/' + unit)

        outer_radius = 1.0 / np.sqrt(2.0) * np.hypot(
            self.lfg.lixel_positions_x.max(),
            self.lfg.lixel_positions_y.max())
        zoom_radius = 1.0 / 10.0 * outer_radius

        # zoom center
        zoom_center_x = [-zoom_radius, zoom_radius]
        zoom_center_y = [-zoom_radius, zoom_radius]
        ax.set_ylim(zoom_center_y)
        ax.set_xlim(zoom_center_x)
        self.__save_fig(fig, 'overview_' + simple_name + '_zoom_center.jpg')

        # zoom pos x
        zoom_posx_x = [
            0.95 * outer_radius - zoom_radius,
            0.95 * outer_radius + zoom_radius]
        zoom_posx_y = [-zoom_radius, zoom_radius]
        ax.set_xlim(zoom_posx_x)
        ax.set_ylim(zoom_posx_y)
        self.__save_fig(fig, 'overview_' + simple_name + '_zoom_pos_x.jpg')

        # zoom pos y
        zoom_posy_x = [-zoom_radius, zoom_radius]
        zoom_posy_y = [
            0.95 * outer_radius - zoom_radius,
            0.95 * outer_radius + zoom_radius]
        ax.set_xlim(zoom_posy_x)
        ax.set_ylim(zoom_posy_y)
        self.__save_fig(fig, 'overview_' + simple_name + '_zoom_pos_y.jpg')

        # full
        outer_fov_r = 0.95 * outer_radius + zoom_radius
        if indicate_zoom_region_center:
            splt.ax_add_box(
                ax,
                xlim=zoom_center_x,
                ylim=zoom_center_y,
                linewidth=1)
        if indicate_zoom_region_pos_x:
            splt.ax_add_box(
                ax,
                xlim=zoom_posx_x,
                ylim=zoom_posx_y,
                linewidth=1)
        if indicate_zoom_region_pos_y:
            splt.ax_add_box(
                ax,
                xlim=zoom_posy_x,
                ylim=zoom_posy_y,
                linewidth=1)
        ax.set_ylim(1.01*np.array([-outer_fov_r, outer_fov_r]))
        ax.set_xlim(1.01*np.array([-outer_fov_r, outer_fov_r]))
        self.__save_fig(fig, 'overview_' + simple_name + '.jpg')
        plt.close(fig)


    def save(self):
        os.makedirs(self.out_dir, exist_ok=True)

        self.save_cx_mean()
        self.save_cy_mean()
        self.save_cx_stddev()
        self.save_cy_stddev()

        self.save_x_mean()
        self.save_y_mean()
        self.save_x_stddev()
        self.save_y_stddev()

        self.save_time_mean()
        self.save_time_stddev()

        self.save_efficiency()
        self.save_efficiency_relative_error()

        self.save_c_mean_vs_c_std()
        self.save_x_y_hist2d()
        self.save_cx_cy_hist2d()

        self.save_sensor_plane_overview(
            I=self.lfg.time_delay_mean,
            name=r'$\overline{t}_\mathrm{pap}$',
            unit='s',
            simple_name='t_mean_aperture',
            indicate_zoom_region_center=True,
            indicate_zoom_region_pos_x=True,
            indicate_zoom_region_pos_y=True)

        self.save_sensor_plane_overview(
            I=self.lfg.time_delay_image_mean,
            name=r'$\overline{t}_\mathrm{img}$',
            unit='s',
            simple_name='t_mean_image',
            indicate_zoom_region_center=True,
            indicate_zoom_region_pos_x=True,
            indicate_zoom_region_pos_y=True)

        self.save_sensor_plane_overview(
            I=self.lfg.efficiency,
            name=r'$\overline{\eta}$',
            unit='1',
            simple_name='eta_mean',
            indicate_zoom_region_center=True,
            indicate_zoom_region_pos_x=True,
            indicate_zoom_region_pos_y=True)

        self.save_sensor_plane_overview(
            I=self.lfg.x_mean,
            name=r'$\overline{x}$',
            unit='m',
            simple_name='x_mean',
            indicate_zoom_region_center=False,
            indicate_zoom_region_pos_x=True,
            indicate_zoom_region_pos_y=False)

        self.save_sensor_plane_overview(
            I=self.lfg.x_std,
            name=r'$\sigma_x$',
            unit='m',
            simple_name='x_std',
            indicate_zoom_region_center=False,
            indicate_zoom_region_pos_x=False,
            indicate_zoom_region_pos_y=False)

        self.save_sensor_plane_overview(
            I=self.lfg.y_mean,
            name=r'$\overline{y}$',
            unit='m',
            simple_name='y_mean',
            indicate_zoom_region_center=False,
            indicate_zoom_region_pos_x=True,
            indicate_zoom_region_pos_y=False)

        self.save_sensor_plane_overview(
            I=self.lfg.y_std,
            name=r'$\sigma_y$',
            unit='m',
            simple_name='y_std',
            indicate_zoom_region_center=False,
            indicate_zoom_region_pos_x=False,
            indicate_zoom_region_pos_y=False)

        self.save_sensor_plane_overview(
            I=np.rad2deg(self.lfg.cx_mean),
            name=r'$\overline{c_x}$',
            unit=r'$^\circ$',
            simple_name='cx_mean',
            indicate_zoom_region_center=False,
            indicate_zoom_region_pos_x=False,
            indicate_zoom_region_pos_y=False)

        self.save_sensor_plane_overview(
            I=np.rad2deg(self.lfg.cx_std),
            name=r'$\sigma_{c_x}$',
            unit=r'$^\circ$',
            simple_name='cx_std',
            indicate_zoom_region_center=False,
            indicate_zoom_region_pos_x=False,
            indicate_zoom_region_pos_y=False)

        self.save_sensor_plane_overview(
            I=np.rad2deg(self.lfg.cy_mean),
            name=r'$\overline{c_y}$',
            unit=r'$^\circ$',
            simple_name='cy_mean',
            indicate_zoom_region_center=False,
            indicate_zoom_region_pos_x=False,
            indicate_zoom_region_pos_y=False)

        self.save_sensor_plane_overview(
            I=np.rad2deg(self.lfg.cy_std),
            name=r'$\sigma_{c_y}$',
            unit=r'$^\circ$',
            simple_name='cy_std',
            indicate_zoom_region_center=False,
            indicate_zoom_region_pos_x=False,
            indicate_zoom_region_pos_y=False)
