#!/usr/bin/python
import argparse
import numpy as np
import sebastians_matplotlib_addons as splt
import os
import plenopy

parser = argparse.ArgumentParser(
    prog="plenopy/trigger/scripts/plot.py",
    description=("Plot the trigger's geometry of the plenoscope"),
)

parser.add_argument(
    "trigger_geometry_path",
    metavar="PATH",
    type=str,
    help="directory containing the trigger's geometry.",
)

args = parser.parse_args()

trg_geom_dir = os.path.abspath(args.trigger_geometry_path)
plot_dir = os.path.join(trg_geom_dir, "plot")

os.makedirs(plot_dir, exist_ok=True)

MATPLOTLIB_RCPARAMS = {
    "mathtext.fontset": "cm",
    "font.family": "STIXGeneral",
}
FIGURE_STYLE = {"rows": 720, "cols": 1280, "fontsize": 1.0}
AX_SPAN = [0.2, 0.2, 0.75, 0.75]
splt.matplotlib.rcParams.update(MATPLOTLIB_RCPARAMS)


trigger_geometry = plenopy.trigger.geometry.read(trg_geom_dir)
trigger_summation_statistics = plenopy.trigger.geometry.init_summation_statistics(
    trigger_geometry=trigger_geometry
)

stats = trigger_summation_statistics

cxy_lim = [-3.5, 3.5]

for focus in range(trigger_geometry["number_foci"]):

    fig = splt.figure(FIGURE_STYLE)
    ax = splt.add_axes(fig=fig, span=AX_SPAN)
    plenopy.plot.image.add2ax(
        ax=ax,
        I=np.array(stats["foci"][focus]["number_lixel_in_pixel"]),
        px=np.rad2deg(trigger_geometry["image"]["pixel_cx_rad"]),
        py=np.rad2deg(trigger_geometry["image"]["pixel_cy_rad"]),
        colormap="viridis",
        hexrotation=30,
        vmin=None,
        vmax=None,
        colorbar=True,
    )
    ax.set_xlim(cxy_lim)
    ax.set_ylim(cxy_lim)
    ax.set_aspect("equal")
    fig.suptitle("number lixel in pixel")
    ax.set_xlabel(r"$c_x\,/\,1^{\circ}$")
    ax.set_ylabel(r"$c_y\,/\,1^{\circ}$")
    fig.savefig(
        os.path.join(
            plot_dir, "focus_{:06d}_lixel_in_pixel_overview.jpg".format(focus),
        )
    )
    splt.close(fig)

    fig = splt.figure(FIGURE_STYLE)
    ax = splt.add_axes(fig=fig, span=AX_SPAN)
    num_lip = stats["foci"][focus]["number_lixel_in_pixel"]
    bin_edges = np.arange(np.min(num_lip), np.max(num_lip) + 1)
    splt.ax_add_histogram(
        ax=ax,
        bincounts=np.histogram(num_lip, bins=bin_edges)[0],
        bin_edges=bin_edges,
        linestyle="-",
        linecolor="k",
    )
    ax.set_xlabel("number lixel in pixel")
    ax.set_ylabel("number pixel")
    ax.semilogy()
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    fig.savefig(
        os.path.join(
            plot_dir,
            "focus_{:06d}_lixel_in_pixel_histogram.jpg".format(focus),
        )
    )
    splt.close(fig)

    fig = splt.figure(FIGURE_STYLE)
    ax = splt.add_axes(fig=fig, span=AX_SPAN)
    num_pil = stats["foci"][focus]["number_pixel_in_lixel"]
    bin_edges = np.arange(np.min(num_pil), np.max(num_pil) + 1)
    splt.ax_add_histogram(
        ax=ax,
        bincounts=np.histogram(num_pil, bins=bin_edges)[0],
        bin_edges=bin_edges,
        linestyle="-",
        linecolor="k",
    )
    ax.set_xlabel("number pixel in lixel")
    ax.set_ylabel("number lixel")
    ax.semilogy()
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    fig.savefig(
        os.path.join(
            plot_dir,
            "focus_{:06d}_pixel_in_lixel_histogram.jpg".format(focus),
        )
    )
    splt.close(fig)
