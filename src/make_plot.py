import numpy as np
import matplotlib.pyplot as plt
import matplotlib
def plot_with_slider(x, y, grids, xlim=None, ylim=None, fix_clim=False, walls=None):
    import numpy as np
    from matplotlib.widgets import Slider

    # optional mask
    if walls is not None:
        grids = [np.ma.array(grid, mask=walls) for grid in grids]

    pore_length = len(grids)
    n_vals = range(pore_length)

    # axis limits
    if xlim is None:
        xlim = (x.min(), x.max())
    if ylim is None:
        ylim = (y.min(), y.max())

    # global clim if requested
    if fix_clim:
        vmin = min(Z.min() for Z in grids)
        vmax = max(Z.max() for Z in grids)
    else:
        vmin = vmax = None

    # --- Plot ---
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    cmap_ = matplotlib.colormaps["CMRmap_r"]
    cmap_.set_bad(color="green")

    # start from middle slice (logical 0)
    idx0 = pore_length // 2
    im = ax.imshow(
        grids[idx0],
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        aspect="equal",
        cmap=cmap_,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(fr"Polymer chains grafted at slice $z \in [{0, 1}]$")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\phi$")

    # --- Slider setup (centered) ---
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
    slider = Slider(
        ax_slider,
        "slice",
        valmin=-idx0,
        valmax=pore_length - idx0 - 1,
        valinit=0,
        valstep=1,
    )

    def update(val):
        logical_idx = int(slider.val)           # centered index
        actual_idx = logical_idx + idx0         # convert to array index
        im.set_data(grids[actual_idx])
        ax.set_title(fr"Polymer chains grafted at slice $z \in [{slider.val, slider.val+1}]$")
        if not fix_clim:
            im.set_clim(grids[actual_idx].min(), grids[actual_idx].max())
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()  # opens in external GUI window
    ax.set_xlabel("$z$")
    ax.set_ylabel("$r$")
    return fig, ax



def plot_grid(x, y, grid, xlim=None, ylim=None, walls=None):
    # apply mask if walls given
    if walls is not None:
        grid = np.ma.array(grid, mask=walls)

    # handle axis limits
    if xlim is None:
        xlim = (x.min(), x.max())
    if ylim is None:
        ylim = (y.min(), y.max())

    # --- Plot ---
    fig, ax = plt.subplots()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    cmap_ = matplotlib.colormaps["CMRmap_r"]
    cmap_.set_bad(color="green")

    im = ax.imshow(
        grid,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        aspect="equal",
        cmap=cmap_,
    )

    ax.set_title(r"Polymer volume concentration")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\phi$")

    ax.set_xlabel("$z$")
    ax.set_ylabel("$r$")

    return fig, ax