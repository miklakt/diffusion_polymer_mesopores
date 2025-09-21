import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle

def mirror_extend(arr, axis=0):
    """
    Mirror (reflect) a numpy array along one axis and keep original,
    so the domain doubles along that axis.

    Parameters
    ----------
    arr : np.ndarray
        Input array (any number of dimensions).
    axis : int
        Axis along which to mirror.

    Returns
    -------
    np.ndarray
        Extended array with shape doubled along the given axis.
    """
    mirrored = np.flip(arr, axis=axis)
    extended = np.concatenate([mirrored, arr], axis=axis)
    return extended

def plot_with_slider(z, r, grids, 
                     zlim=None, rlim=None, 
                     fix_clim=True, 
                     walls=None, 
                     mirror=True, hatch_width=9.0
                     ):
    """
    Interactive heatmap plot with slider over slices.

    Parameters
    ----------
    z, r : array-like
        Coordinates along pore axis (z) and radial axis (r).
    grids : list of 2D arrays
        Each grid has shape (len(r), len(z)).
    zlim, rlim : tuple, optional
        Axis limits.
    fix_clim : bool, optional
        If True, color limits are fixed globally across all frames.
    walls : 2D array, optional
        Mask array, True for excluded cells (e.g. walls).
    mirror : bool, default True
        If True, mirror along z-axis while preserving orientation.
    hatch_width : float, default 9.0
        Line width of the hatched background stripes.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from matplotlib.patches import Rectangle
    import matplotlib as mpl

    # mirror if requested
    if mirror:
        r = np.concatenate((-r[::-1], r))
        grids = [mirror_extend(grid.T, axis=1).T for grid in grids]
        if walls is not None:
            walls = mirror_extend(walls.T, axis=1).T

    # apply mask (do this AFTER mirroring)
    if walls is not None:
        grids = [np.ma.array(grid, mask=walls) for grid in grids]
    else:
        grids = [np.ma.masked_invalid(grid) for grid in grids]

    pore_length = len(grids)

    # axis limits
    if zlim is None:
        zlim = (z.min(), z.max())
    if rlim is None:
        rlim = (r.min(), r.max())

    # global clim if requested
    if fix_clim:
        vmin = min(grid.min() for grid in grids)
        vmax = max(grid.max() for grid in grids)
    else:
        vmin = vmax = None

    # --- Plot ---
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax.set_xlim(*zlim)
    ax.set_ylim(*rlim)

    cmap_ = plt.colormaps["CMRmap_r"].copy()
    cmap_.set_bad(color=(0, 0, 0, 0))  # transparent for masked values

    # start from middle slice (logical 0)
    idx0 = pore_length // 2
    im = ax.imshow(
        grids[idx0],
        extent=[z.min(), z.max(), r.min(), r.max()],
        origin="lower",
        aspect="equal",
        cmap=cmap_,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title("Contribution to polymer volume concentration for\n"+ fr"chains grafted at slice $z \in [{0, 1}]$")
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
        logical_idx = int(slider.val)
        actual_idx = logical_idx + idx0
        im.set_data(grids[actual_idx])
        ax.set_title(fr"Polymer chains grafted at slice $z \in [{slider.val, slider.val+1}]$")
        if not fix_clim:
            im.set_clim(grids[actual_idx].min(), grids[actual_idx].max())
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # hatch background: green with stripes
    with mpl.rc_context({'hatch.linewidth': hatch_width}):
        bg = Rectangle(
            (zlim[0], rlim[0]),
            zlim[1] - zlim[0],
            rlim[1] - rlim[0],
            facecolor="green",
            edgecolor="darkgreen",
            hatch="/",   # 45 degrees
            linewidth=0.0,
            zorder=-1
        )
        ax.add_patch(bg)

    ax.set_xlabel("$z$")
    ax.set_ylabel("$r$")

    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda val, _: f"{abs(val):.0f}"))

    plt.show()

    return fig, ax




def plot_grid(z, r, grid, 
              zlim=None, rlim=None, 
              walls=None, mirror=True, 
              hatch_width=9.0
              ):
    """
    Plot a single phi(r,z) heatmap with optional mirrored axis and hatched background.

    Parameters
    ----------
    z, r : array-like
        Coordinates along pore axis (z) and radial axis (r).
    grid : 2D array
        Array of shape (len(r), len(z)) with phi values.
    zlim, rlim : tuple, optional
        Axis limits.
    walls : 2D array, optional
        Mask array, True for excluded cells (e.g. walls).
    mirror : bool, default False
        If True, mirror along z-axis while keeping orientation.
    hatch_width : float, default 9.0
        Line width of the hatched background stripes.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib as mpl

    # mirror if requested
    if mirror:
        r = np.concatenate((-r[::-1], r))
        grid = mirror_extend(grid.T, axis=1).T
        if walls is not None:
            walls = mirror_extend(walls.T, axis=1).T

    # apply mask if walls given
    if walls is not None:
        grid = np.ma.array(grid, mask=walls)
    else:
        grid = np.ma.masked_invalid(grid)

    # handle axis limits
    if zlim is None:
        zlim = (z.min(), z.max())
    if rlim is None:
        rlim = (r.min(), r.max())

    # --- Plot ---
    fig, ax = plt.subplots()
    ax.set_xlim(*zlim)
    ax.set_ylim(*rlim)

    cmap_ = plt.colormaps["CMRmap_r"]
    cmap_.set_bad(color=(0, 0, 0, 0))  # transparent for masked values

    im = ax.imshow(
        grid,
        extent=[z.min(), z.max(), r.min(), r.max()],
        origin="lower",
        aspect="equal",
        cmap=cmap_,
    )

    ax.set_title(r"Polymer volume concentration")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\phi$")

    ax.set_xlabel("$z$")
    ax.set_ylabel("$r$")

    # --- Hatched background ---
    with mpl.rc_context({'hatch.linewidth': hatch_width}):
        bg = Rectangle(
            (zlim[0], rlim[0]),
            zlim[1] - zlim[0],
            rlim[1] - rlim[0],
            facecolor="green",
            edgecolor="darkgreen",
            hatch="/",   # 45 degrees
            linewidth=0.0,
            zorder=-1
        )
        ax.add_patch(bg)

    # Force r-axis ticks positive
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda val, _: f"{abs(val):.0f}"))

    return fig, ax


def plot_phi_with_colloid_slider(
    SCF_results,
    zlim=None,
    rlim=None,
    fix_clim=True,
    walls=None,
    mirror=True,
    hatch_width=5.0,
    **kwargs
):
    """
    Plot phi(r,z) heatmaps with a slider over colloid positions.
    Annotates free energy value in lower-right corner.
    If SCF_results contains 'colloid_mask', it is merged with walls.

    Extra keyword arguments (**kwargs) are used to filter SCF_results.
    After filtering, only 'colloid_position' is allowed to vary.
    If ambiguity remains (more than one value in a scalar column),
    a ValueError is raised.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.widgets import Slider
    from matplotlib.patches import Rectangle
    import matplotlib as mpl

    if zlim is not None:
        df = SCF_results.loc[(SCF_results.colloid_position>zlim[0])&(SCF_results.colloid_position<zlim[1])].copy()
    else:
        df = SCF_results.copy()

    # --- Apply kwargs filters ---
    for key, value in kwargs.items():
        if key not in df.columns:
            raise KeyError(f"Column {key!r} not in SCF_results")
        df = df[df[key] == value]

    if df.empty:
        raise ValueError("No matching rows after applying filters")

    # --- Check uniqueness of fixed parameters ---
    ignore_cols = {"colloid_position", "phi", "colloid_mask", "free_energy"}
    scalar_cols = [
        c for c in df.columns
        if c not in ignore_cols
        and not df[c].apply(lambda x: isinstance(x, (list, np.ndarray))).any()
    ]

    for c in scalar_cols:
        if df[c].nunique() > 1:
            raise ValueError(
                f"Ambiguous dataset: column {c!r} has multiple values, "
                f"please fix with a keyword argument."
            )

    # --- Prepare data ---
    df_sorted = df.sort_values("colloid_position").reset_index(drop=True)
    positions = df_sorted["colloid_position"].to_numpy()
    free_energies = df_sorted["free_energy"].to_numpy()
    Rs = df_sorted["R"].iloc[0]
    Zs = df_sorted["Z"].iloc[0]
    grids = list(df_sorted["phi"])

    # Check if colloid_mask column exists
    if "colloid_mask" in df_sorted.columns:
        colloid_masks = list(df_sorted["colloid_mask"])
        if walls is not None:
            walls = [np.logical_or(walls, cm) for cm in colloid_masks]
        else:
            walls = colloid_masks

    # Mirror if requested
    if mirror:
        Rs = np.concatenate((-Rs[::-1], Rs))
        grids = [mirror_extend(grid.T, axis=1).T for grid in grids]
        if walls is not None:
            walls = [mirror_extend(w.T, axis=1).T for w in walls]

    # Apply mask (AFTER mirroring)
    if walls is not None:
        grids = [np.ma.array(g, mask=w) for g, w in zip(grids, walls)]
    else:
        grids = [np.ma.masked_invalid(g) for g in grids]

    extent = [Zs.min(), Zs.max(), Rs.min(), Rs.max()]

    # Axis limits
    if zlim is None:
        zlim = (Zs.min(), Zs.max())
    if rlim is None:
        rlim = (Rs.min(), Rs.max())

    # Global clim if requested
    if fix_clim:
        vmin = min(g.min() for g in grids)
        vmax = max(g.max() for g in grids)
    else:
        vmin = vmax = None

    # --- Plot ---
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax.set_xlim(*zlim)
    ax.set_ylim(*rlim)

    cmap_ = plt.colormaps["CMRmap_r"].copy()
    cmap_.set_bad(color=(0, 0, 0, 0))

    idx0 = 0
    im = ax.imshow(
        grids[idx0],
        extent=extent,
        origin="lower",
        aspect="equal",
        cmap=cmap_,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(fr"Colloid at position $z = {positions[idx0]}$")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\phi$")

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda val, _: f"{abs(val):.0f}")
    )

    # Free energy annotation
    text_template = "Free energy: {:+.3f}"
    free_text = ax.text(
        0.98, 0.02,
        text_template.format(free_energies[idx0]),
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )

    # --- Slider setup ---
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
    slider = Slider(
        ax_slider,
        "colloid z",
        valmin=0,
        valmax=len(positions) - 1,
        valinit=idx0,
        valstep=1,
    )
    slider.valtext.set_text(str(positions[idx0]))

    def update(val):
        idx = int(slider.val)
        im_arr = grids[idx]
        im.set_data(im_arr)
        ax.set_title(fr"Colloid at position $z = {positions[idx]}$")
        slider.valtext.set_text(str(positions[idx]))
        free_text.set_text(text_template.format(free_energies[idx]))
        if not fix_clim:
            im.set_clim(im_arr.min(), im_arr.max())
        fig.canvas.draw_idle()

    slider.on_changed(update)

    ax.set_xlabel("$z$")
    ax.set_ylabel("$r$")

    # --- Hatched background ---
    with mpl.rc_context({'hatch.linewidth': hatch_width}):
        bg = Rectangle(
            (extent[0], extent[2]),
            extent[1] - extent[0],
            extent[3] - extent[2],
            facecolor="green",
            edgecolor="darkgreen",
            hatch="//",
            linewidth=0.0,
            zorder=-1
        )
        ax.add_patch(bg)

    # plt.show()
    return fig





def interactive_plot_phi(SCF_results, **plot_kwargs):
    """
    Interactive widget with dropdowns for scalar columns.
    User selects filters and presses Update to draw a new figure.
    Old figures are closed to avoid accumulation.
    """
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig = plt.Figure()
    ignore_cols = {"colloid_position", "phi", "colloid_mask", "free_energy"}
    scalar_cols = [
        c for c in SCF_results.columns
        if c not in ignore_cols
        and not SCF_results[c].apply(lambda x: isinstance(x, (list, np.ndarray))).any()
    ]
    
    # Create dropdowns
    dropdowns = {}
    for col in scalar_cols:
        options = [None] + sorted(SCF_results[col].unique().tolist())
        dropdowns[col] = widgets.Dropdown(
            options=options,
            description=col,
            value=None,
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px')
        )
    
    # Buttons
    update_btn = widgets.Button(description="Update", button_style="success")
    reset_btn = widgets.Button(description="Reset", button_style="warning")
    
    # out = widgets.Output()
    
    def get_filters():
        return {col: dd.value for col, dd in dropdowns.items() if dd.value is not None}
    
    def update_plot(_=None):
        nonlocal fig
        filters = get_filters()
        plt.close(fig)
        clear_output()
        display(controls)
        fig = plot_phi_with_colloid_slider(SCF_results, **plot_kwargs, **filters)
        # fig.show()
        # out.clear_output()
    
    def reset_filters(_=None):
        nonlocal fig
        for dd in dropdowns.values():
            dd.value = None
        plt.close(fig)
        # with out:
        #     clear_output()
        #     plt.close(fig)
    
    update_btn.on_click(update_plot)
    reset_btn.on_click(reset_filters)
    
    controls = widgets.VBox(list(dropdowns.values()) + [widgets.HBox([update_btn, reset_btn])])
    print("TO PLOT SELECT PARAMETERS:")
    display(controls)
    # fig
    # clear_output(wait=True)


def plot_volume_and_surface_matrices_cylinder(volume, surface):

    colloid_diameter = np.shape(volume)[1]
    fig, ax = plt.subplots(nrows = 2, sharex = True)
    ax[0].pcolormesh(volume, edgecolor = "black", linewidth = 0.7)
    ax[1].pcolormesh(surface, edgecolor = "black", linewidth = 0.7)
    ax[1].set_xticks(np.arange(0,colloid_diameter+1, 1))
    ax[0].set_yticks(np.arange(0,colloid_diameter//2+1, 1))
    ax[1].set_yticks(np.arange(0,colloid_diameter//2+1, 1))
    ax[1].set_xlim(0,colloid_diameter)
    ax[0].set_ylim(0,colloid_diameter//2)
    ax[1].set_ylim(0,colloid_diameter//2)
    ax[0].set_yticks(np.arange(0,colloid_diameter//2+1, 1))
    ax[1].set_yticks(np.arange(0,colloid_diameter//2+1, 1))
    ax[1].set_xlabel("$i$")
    ax[0].set_ylabel("$k$")
    ax[1].set_ylabel("$k$")
    ax[0].set_aspect("equal", adjustable="box") 
    ax[1].set_aspect("equal", adjustable="box") 
    fig.suptitle("Cylindrical particle volume (top) and surface (bottom) projection matrices")
    print("Cylindrical particle volume projection matrix:")
    print(volume)
    print()
    print("Cylindrical particle surface projection matrix:")
    print(surface)

def plot_insertion_free_energy(SCF_results, pore_length, show_ground_state=False, xlim = None, **kwargs):
    """
    Plot insertion free energy profile ΔF_SF-SCF as a function of colloid position.

    Parameters
    ----------
    SCF_results : pandas.DataFrame
        Must contain columns ["colloid_position", "free_energy"] and metadata columns.
    pore_length : int
        Length of the pore (in lattice units).
    ground_state_pos : int, optional
        Colloid position chosen as ground state (default: None).
    **kwargs :
        Additional filters to subset SCF_results, e.g. chi_PS=0.6, chi_PC=0.0, colloid_diameter=6.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    # --- Apply kwargs filters ---
    df = SCF_results.copy()
    for key, value in kwargs.items():
        if key not in df.columns:
            raise KeyError(f"Column {key!r} not in SCF_results")
        df = df[df[key] == value]

    if df.empty:
        raise ValueError("No matching rows after applying filters")

    # --- Check uniqueness of fixed parameters ---
    ignore_cols = {"colloid_position", "phi", "colloid_mask", "free_energy"}
    scalar_cols = [
        c for c in df.columns
        if c not in ignore_cols
        and not df[c].apply(lambda x: isinstance(x, (list, np.ndarray))).any()
    ]
    for c in scalar_cols:
        if df[c].nunique() > 1:
            raise ValueError(
                f"Ambiguous dataset: column {c!r} has multiple values, "
                f"please fix with a keyword argument."
            )

    # --- Main plot ---
    fig, ax = plt.subplots()

    ax.plot(df["colloid_position"], df["free_energy"], marker="s")
    ax.set_ylabel(r"$\Delta F_{\text{SF-SCF}}$")
    ax.set_xlabel(r"$z_c$")

    if xlim is not None:
        ax.set_xlim(xlim)

    # Ground-state reference point
    if show_ground_state:
        ref_idx = df["colloid_position"].abs().idxmax()
        ground_state = df.loc[ref_idx]
        ax.annotate(
            "    Ground state",
            xy=(ground_state["colloid_position"], ground_state["free_energy"]),
            xycoords="data",
            textcoords="offset points",
            xytext=(10, 40),  # offset in points
            arrowprops=dict(arrowstyle="->"),
        )
        ax.plot(
            ground_state["colloid_position"],
            ground_state["free_energy"],
            marker="s",
            color="red",
            zorder=5,
        )
        if xlim is None:
            ax.set_xlim((ground_state["colloid_position"]-5, 0))

    # Vertical line at pore entrance (half pore length)
    entrance_z = -pore_length // 2
    ax.axvline(entrance_z, color="k", linestyle="-")

    # Labels for exterior and interior
    ylim = ax.get_ylim()
    ax.text(entrance_z - 5, ylim[1] * 0.95, "exterior", ha="right", va="top")
    ax.text(entrance_z + 5, ylim[1] * 0.05, "interior", ha="left", va="bottom")

    # Shaded interior region
    ax.axvspan(entrance_z, 0, facecolor="green", alpha=0.1, zorder=0)

    return fig, ax

def interactive_sphere_kernels(colloid_diameter: int, r_shift_max: int = 60):
    """
    Build an interactive figure showing volume and surface kernels
    for a sphere of given diameter, with adjustable radial shift.

    Parameters
    ----------
    colloid_diameter : int
        Diameter of the colloid (lattice units).
    r_shift_max : int, optional
        Maximum radial shift allowed by the slider (default: 60).
    """
    from src.sphere import generate_sphere_volume_surface_matrices
    from matplotlib.widgets import Slider
    from matplotlib.ticker import MaxNLocator
    # Initial state
    initial_r_shift = 0
    volume, surface, extent = generate_sphere_volume_surface_matrices(
        colloid_diameter // 2, (0, initial_r_shift)
    )

    # ------------------------------------------------------------------
    # Create figure with two subplots (volume and surface)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Volume kernel plot
    im1 = ax1.imshow(volume, extent=extent, origin="lower", aspect="equal")
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.6)
    cbar1.set_label("Volume")
    ax1.set_xlabel("$z$")
    ax1.set_ylabel("$r$")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Surface kernel plot
    im2 = ax2.imshow(surface, extent=extent, origin="lower", aspect="equal")
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.6)
    cbar2.set_label("Surface")
    ax2.set_xlabel("$z$")
    ax2.set_ylabel("$r$")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.subplots_adjust(bottom=0.25)  # leave space for slider

    # ------------------------------------------------------------------
    # Slider for r_shift
    # ------------------------------------------------------------------
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, "r_shift", 0, r_shift_max, valinit=initial_r_shift, valstep=1)

    def update(val):
        r_shift = slider.val
        volume, surface, extent = generate_sphere_volume_surface_matrices(
            colloid_diameter // 2, (0, r_shift)
        )

        # Update images and extents
        im1.set_data(volume)
        im1.set_extent(extent)

        im2.set_data(surface)
        im2.set_extent(extent)

        im1.set_clim(0, np.max(volume))
        im2.set_clim(0, np.max(surface))

        fig.canvas.draw_idle()

    slider.on_changed(update)

    # plt.show()
    return fig, (ax1, ax2), slider