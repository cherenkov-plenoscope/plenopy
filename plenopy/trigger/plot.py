import numpy as np
import sebastians_matplotlib_addons as splt
import os
from .. import plot


def _make_fig_ax(figsize):
    fig = plt.figure(
        figsize=(figsize.width, figsize.hight),
        dpi=figsize.dpi
    )
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig, ax


def add2ax_histogram(ax, counts, bin_edges, style='k-'):
    number_bins = len(counts)
    assert len(bin_edges) == number_bins + 1
    for i in range(number_bins):
        x_start = bin_edges[i]
        x_end = bin_edges[i+1]
        ax.plot([x_start, x_end], [counts[i], counts[i]], style)


def write_figures_to_directory(
    trigger_geometry,
    trigger_summation_statistics,
    out_dir,
    figure_style=splt.FIGURE_16_9,
):
    os.makedirs(out_dir, exist_ok=True)
    stats = trigger_summation_statistics

    for focus in range(trigger_geometry['number_foci']):

        fig = splt.figure(style=figure_style)
        ax = splt.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
        plot.image.add2ax(
            ax=ax,
            I=np.array(stats['foci'][focus]['number_lixel_in_pixel']),
            px=np.rad2deg(trigger_geometry['image']['pixel_cx_rad']),
            py=np.rad2deg(trigger_geometry['image']['pixel_cy_rad']),
            colormap='viridis',
            hexrotation=30,
            vmin=None,
            vmax=None,
            colorbar=True
        )
        fig.suptitle('number lixel in pixel')
        ax.set_xlabel('cx/deg')
        ax.set_ylabel('cy/deg')
        fig.savefig(
            os.path.join(
                out_dir,
                'focus_{:06d}_lixel_in_pixel_overview.jpg'.format(focus)
            )
        )
        splt.close_figure(fig)

        fig = splt.figure(style=figure_style)
        ax = splt.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
        num_lip = stats['foci'][focus]['number_lixel_in_pixel']
        bin_edges = np.arange(np.min(num_lip), np.max(num_lip)+1)
        splt.ax_add_histogram(
            ax=ax,
            bincounts=np.histogram(num_lip, bins=bin_edges)[0],
            bin_edges=bin_edges,
            linestyle='-',
            linecolor="k",
        )
        ax.set_xlabel('number lixel in pixel')
        ax.set_ylabel('number pixel')
        ax.semilogy()
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        fig.savefig(
            os.path.join(
                out_dir,
                'focus_{:06d}_lixel_in_pixel_histogram.jpg'.format(focus)
            )
        )
        splt.close_figure(fig)

        fig = splt.figure(style=figure_style)
        ax = splt.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
        num_pil = stats['foci'][focus]['number_pixel_in_lixel']
        bin_edges = np.arange(np.min(num_pil), np.max(num_pil)+1)
        splt.ax_add_histogram(
            ax=ax,
            bincounts=np.histogram(num_pil, bins=bin_edges)[0],
            bin_edges=bin_edges,
            linestyle='-',
            linecolor="k",
        )
        ax.set_xlabel('number pixel in lixel')
        ax.set_ylabel('number lixel')
        ax.semilogy()
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        fig.savefig(
            os.path.join(
                out_dir,
                'focus_{:06d}_pixel_in_lixel_histogram.jpg'.format(focus)
            )
        )
        splt.close_figure(fig)
