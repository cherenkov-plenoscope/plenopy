import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable


def show(image):
    fig, ax = plt.subplots()
    add_pixel_image_to_ax(image, ax)
    plt.show()


def add2ax(
    ax,
    I,
    px,
    py,
    colormap="viridis",
    hexrotation=30,
    vmin=None,
    vmax=None,
    colorbar=True,
    norm=None,
):
    if vmin is None:
        vmin = I.min()
    if vmax is None:
        vmax = I.max()
    fov = np.abs(px).max() * 1.01
    Area = fov * fov
    bin_radius = 1.15 * np.sqrt((Area / I.shape[0]))

    nfov = fov + bin_radius

    orientation = np.deg2rad(hexrotation)

    patches = []
    for d in range(I.shape[0]):
        patches.append(
            RegularPolygon(
                (px[d], py[d]),
                numVertices=6,
                radius=bin_radius,
                orientation=orientation,
            )
        )

    p = PatchCollection(
        patches, cmap=colormap, alpha=1, edgecolor="none", norm=norm
    )
    p.set_array(I)
    p.set_clim([vmin, vmax])

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(p, cax=cax)

    ax.add_collection(p)
    return p


def add_pixel_image_to_ax(
    img, ax, colormap="viridis", vmin=None, vmax=None, colorbar=True
):
    ax.set_xlabel("cx/deg")
    ax.set_ylabel("cy/deg")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return add2ax(
        ax=ax,
        I=img.intensity,
        px=np.rad2deg(img.pixel_pos_x),
        py=np.rad2deg(img.pixel_pos_y),
        colormap=colormap,
        hexrotation=30,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
    )


def add_paxel_image_to_ax(
    img, ax, colormap="viridis", vmin=None, vmax=None, colorbar=True
):
    ax.set_xlabel("x/m")
    ax.set_ylabel("y/m")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return add2ax(
        ax=ax,
        I=img.intensity,
        px=img.pixel_pos_x,
        py=img.pixel_pos_y,
        colormap=colormap,
        hexrotation=0,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
    )
