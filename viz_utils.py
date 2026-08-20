import colorsys
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import mne
from mne_connectivity.viz import plot_connectivity_circle
import numpy as np


def desaturate(color, factor=0.5):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s *= factor
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return r, g, b


def visualize_connectivity_circular(
    con_fname, con_fig_fname, subjects_dir, subject, parcellation='aparc',
    vmin=None, vmax=None, colormap='viridis', interp_colormap=None, title=None,
    n_lines=200, names=None, padding=7.5, fontsize_colorbar=12,
    fontsize_title=20, fontsize_names=10, node_width=3, node_linewidth=1,
    linewidth=2, colorbar_size=0.5, colorbar_label=None, fig=None,
    figsize=(8, 8), ax=None, colorbar=True, legend=False, desaturate_factor=None):
    """Visualize connectivity as a circular graph and save the figure.

    This is a convenience wrapper around :func:`mne_connectivity.viz
    .plot_connectivity_circle` that loads a connectivity matrix from disk
    and performs some annotation handling specific to the project's
    parcellations.

    Parameters
    ----------
    con_fname : str
        Path to a NumPy file containing the connectivity matrix.
    con_fig_fname : str
        Output filename for the generated figure.
    subjects_dir : str
        FreeSurfer subjects directory.
    subject : str
        Subject identifier (without the 'sub-' prefix).
    parcellation : str, optional
        Parcellation name used to read labels (default: 'aparc').
    vmin, vmax : float | None, optional
        Colorbar bounds. If None, sensible defaults are selected from the
        connectivity values.
    colormap : str | Colormap, optional
        Colormap to use for the plot (default: 'viridis').
    interp_colormap : dict | None, optional
        If provided, a dict with keys ``'order'`` and ``'diverging'`` is
        used to interpolate the colormap.
    title : str | None, optional
        Figure title.
    n_lines : int, optional
        Number of strongest links to display as lines.
    names : {None, False, True, 'grouped', 'grouped_combined', 'abbrev'}
        How label names are rendered. See the earlier implementation for
        behaviour when using ``'aparc_nimeg'``.
    padding : float, optional
        Padding for the figure layout.
    fontsize_colorbar, fontsize_title, fontsize_names : float, optional
        Font sizes for the colorbar, title and label names.
    node_width, node_linewidth, linewidth : float, optional
        Styling parameters for the nodes and lines.
    colorbar_size : float | None, optional
        Relative colorbar size (shrink factor).
    colorbar_label : str | None, optional
        Label for the colorbar.
    fig : matplotlib.figure.Figure | None, optional
        Figure to draw into. If None, a new figure is created.
    figsize : tuple, optional
        Size of the figure (in inches) used when a new figure is created.
    ax : matplotlib.axes.Axes | None, optional
        Axis to draw into. If None, one will be created.
    colorbar : bool, optional
        Whether to append a colorbar to the figure (default True).
    legend : bool, optional
        Whether to draw the legend for abbreviations (only for
        ``'aparc_nimeg'``).
    desaturate_factor : float | None, optional
        If not None, factor by which to desaturate label colors (between 0 and 1).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure containing the circular connectivity plot.
    """
    labels = mne.read_labels_from_annot(subject,
                                        parc=parcellation,
                                        subjects_dir=subjects_dir,
                                        sort=False)
    labels = [label for label in labels if 'unknown' not in label.name]
    label_colors = [label.color for label in labels]
    if desaturate_factor is not None:
        label_colors = [desaturate(color, desaturate_factor) for color in label_colors]

    label_names = [label.name for label in labels]
    lh_labels = [label_name for label_name in label_names if label_name.endswith('lh')]
    rh_labels = [label_name for label_name in label_names if label_name.endswith('rh')]

    name_to_abbrev = dict(
        superiorfrontal="SFG",
        frontalpole="FP",
        rostralmiddlefrontal="MFGr",
        parsopercularis="IFGo",
        parsorbitalistriangularis="IFG",
        medialorbitofrontal="MOFC",
        lateralorbitofrontal="LOFC",
        caudalmiddlefrontal="CMFG",
        insula="INS",
        precentral="PRG",
        postcentral="POG",
        paracentral="PAL",
        rostralanteriorcingulate="RAC",
        caudalanteriorcingulate="CAG",
        posteriorcingulate="POC",
        isthmuscingulate="IMC",
        superiortemporal="STG",
        middletemporal="MTG",
        inferiortemporal="ITG",
        fusiform="FUS",
        parahippocampal="PAH",
        superiorparietal="SPC",
        inferiorparietal="IPC",
        supramarginal="SMG",
        precuneus="PC",
        cuneuspericalcarine="CP",
        lateraloccipital="LOC",
        lingual="LG"
    )

    name_to_printname = dict(
        superiorfrontal="Superior frontal gyrus",
        frontalpole="Frontal pole",
        rostralmiddlefrontal="Rostral middle frontal gyrus",
        parsopercularis="Pars opercularis",
        parsorbitalistriangularis="Pars triangularis and orbitalis",
        medialorbitofrontal="Medial orbitofrontal cortex",
        lateralorbitofrontal="Lateral orbitofrontal cortex",
        caudalmiddlefrontal="Caudal middle frontal gyrus",
        insula="Insular cortex",
        precentral="Precentral gyrus",
        postcentral="Postcentral gyrus",
        paracentral="Paracentral lobule",
        rostralanteriorcingulate="Rostral anterior cingulate cortex",
        caudalanteriorcingulate="Caudal anterior cingulate cortex",
        posteriorcingulate="Posterior cingulate cortex",
        isthmuscingulate="Isthmus–cingulate cortex",
        superiortemporal="Superior temporal gyrus",
        middletemporal="Middle temporal gyrus",
        inferiortemporal="Inferior temporal gyrus",
        fusiform="Fusiform gyrus",
        parahippocampal="Parahippocampal gyrus",
        superiorparietal="Superior parietal cortex",
        inferiorparietal="Inferior parietal cortex",
        supramarginal="Supramarginal gyrus",
        precuneus="Precuneus cortex",
        cuneuspericalcarine="Cuneus and pericalcarine cortex",
        lateraloccipital="Lateral occipital cortex",
        lingual="Lingual gyrus"
    )

    if parcellation == 'aparc_nimeg':
        hemis = [label_name.split('-')[-1] for label_name in label_names]
        anat_names = [label_name.split('_')[0].split('-')[0] for label_name in label_names]
        label_anat_regions = ['-'.join([hemi, anat_name]) for hemi, anat_name in zip(hemis, anat_names)]
        label_unique_indices = np.asarray(sorted(np.unique(label_anat_regions, return_index=True)[1]))
        label_counts = np.append(label_unique_indices[1:], 90) - label_unique_indices
        roi_mid_label_idx = np.floor(label_unique_indices + label_counts / 2).astype(int)
        group_boundaries = [0, 11, 12, 19, 23, 32, 40, 45, 50, 58, 67, 71, 78, 79]

        node_order = lh_labels + rh_labels[::-1]
        # ensure group_boundaries are ints to satisfy type checks
        gb_int = tuple(int(x) for x in group_boundaries)
        node_angles = mne.viz.circular_layout(
            label_names,
            node_order,
            start_pos=90,
            group_boundaries=gb_int,
            group_sep=5,
        )
        label_names = [anat_names[label_idx] if label_idx in roi_mid_label_idx else '' for label_idx in range(90)]

    else:
        labels_sorted = sorted(labels, key=lambda lab: np.min(lab.pos[:, 1]), reverse=True)
        label_names_sorted = [label.name for label in labels_sorted]
        lh_labels = [label_name for label_name in label_names_sorted if label_name.endswith('lh')]
        rh_labels = [label_name for label_name in label_names_sorted if label_name.endswith('rh')]
        node_order = lh_labels + rh_labels[::-1]
        # use integer division for group_boundaries to satisfy type checks
        node_angles = mne.viz.circular_layout(
            label_names,
            node_order,
            start_pos=90,
            group_boundaries=[0, len(label_names) // 2]
        )
    con = np.load(con_fname)
    con = con.squeeze(-1) if len(con.shape) == 3 and con.shape[-1] == 1 else con

    if names == 'grouped' and parcellation == 'aparc_nimeg':
        label_names = ['' for _ in range(len(label_names))]
        lh_indices = np.array([5, 11, 15, 21, 27, 35, 42])
        rh_indices = lh_indices + 45
        for idx, label_name in enumerate(
                ['Frontal', 'Insula', 'Central', 'Cingulate', 'Temporal', 'Parietal', 'Occipital']):
            label_names[lh_indices[idx]] = label_name
            label_names[rh_indices[idx]] = label_name
    elif names == 'grouped_combined' and parcellation == 'aparc_nimeg':
        label_names = [name_to_abbrev[name] if name != '' else '' for name in label_names]
        lh_indices = np.array([5, 11, 15, 21, 27, 35, 42])
        label_names = [name if idx >= 45 else '' for idx, name in enumerate(label_names)]
        for idx, name in zip(lh_indices,
                             ['Frontal', 'Insula', 'Central', 'Cingulate', 'Temporal', 'Parietal', 'Occipital']):
            label_names[idx] = name
    elif names == 'abbrev':
        label_names = [name_to_abbrev[name] if name != '' else '' for name in label_names]
    elif not names and names is not None:
        label_names = label_names if names else ['' for i in range(len(label_names))]

    if interp_colormap is not None:
        colormap = _interp_colormap(colormap, interp_colormap['order'], diverging=interp_colormap['diverging'])


    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
        else:
            ax = fig.add_axes((0, 0, 1, 1), projection='polar')

    fig, ax = plot_connectivity_circle(
        con,
        label_names,
        n_lines=n_lines,
        vmin=vmin,
        vmax=vmax,
        node_angles=node_angles,
        node_colors=label_colors,
        title=title,
        fontsize_title=fontsize_title,
        fontsize_names=fontsize_names,
        fontsize_colorbar=fontsize_colorbar,
        textcolor='black',
        facecolor='white',
        node_width=node_width,
        node_linewidth=node_linewidth,
        linewidth=linewidth,
        node_edgecolor='black',
        colormap=colormap,
        colorbar_size=colorbar_size,
        colorbar=False,
        colorbar_pos=(0, 0.2),
        padding=padding,
        ax=ax)

    if colorbar:
        if vmin is None:
            vmin = np.sort(con.ravel())[-n_lines]
        if vmax is None:
            vmax = np.max(con)
        norm = plt.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array(np.linspace(vmin, vmax))
        colorbar_kwargs = dict(
            anchor=(0, 0.19)
        )
        if colorbar_size is not None:
            colorbar_kwargs.update(shrink=colorbar_size)
        ticks = [vmin, (vmax + vmin) / 2, vmax]
        cb = fig.colorbar(sm, ax=ax, ticks=ticks, aspect=14, **colorbar_kwargs)
        cb_yticks = plt.getp(cb.ax.axes, "yticklabels")
        cb.ax.tick_params(labelsize=fontsize_colorbar, direction='out', length=1.25, width=0.25)
        for spine in cb.ax.spines.values():
            spine.set_linewidth(0.25)
        cb.set_label(colorbar_label, size=fontsize_colorbar)
        plt.setp(cb_yticks, color='k')

    if legend and parcellation == 'aparc_nimeg':
        ax = fig.add_axes((0.65, 0.2, 0.35, 0.6))
        indices = [45 + i for i, name in enumerate(label_names[45:]) if name != '']
        n_labels = len(indices)
        n_col = n_labels // 2
        for num, idx in enumerate(indices):
            color = labels[idx].color
            x = (1, 1) if num < n_col else (20, 1)
            y = (n_col - 3 * num, 1) if num < n_col else (n_col - 3 * (num - n_col), 1)
            ax.broken_barh([x], y, color=color, edgecolor='k', linewidth=node_linewidth)
            label_name = anat_names[idx]
            ax.text(x[0] + 2 * x[1], y[0], s=f"{name_to_abbrev[label_name]:4s}", fontsize=fontsize_names)
            ax.text(x[0] + 2 * x[1] + 3.3, y[0], s=f"{name_to_printname[label_name]}", fontsize=fontsize_names)
        ax.set_xlim(0, 40)
        ax.axis('off')

    fig.savefig(con_fig_fname, facecolor='white', transparent=True, dpi=600)

    return fig


def numpy_to_stc(data, parc='aparc_sub_nimeg', subjects_dir=None):
    """Convert an array of region values to a SourceEstimate.

    Parameters
    ----------
    data : array-like
        Array of values with length equal to the number of labels in the
        chosen parcellation (after removing 'unknown' and 'Background').
    parc : str, optional
        Parcellation name used to read labels (default 'aparc_sub_nimeg').
    subjects_dir : str | None, optional
        FreeSurfer SUBJECTS_DIR where the ``fsaverage`` labels are found. If
        None the environment variable ``SUBJECTS_DIR`` is used.

    Returns
    -------
    stc : mne.SourceEstimate
        A SourceEstimate object produced from the provided region values.
    """
    if subjects_dir is None:
        subjects_dir = os.environ['SUBJECTS_DIR']
    labels = mne.read_labels_from_annot('fsaverage',
                                        parc=parc,
                                        subjects_dir=subjects_dir,
                                        sort=False)
    labels = list(filter(lambda label: 'unknown' not in label.name
                         and 'Background' not in label.name, labels))
    stc = mne.labels_to_stc(labels, data)
    return stc


def set_colormap_alpha(cmap, alpha):
    if isinstance(cmap, str):
        cmap = plt.colormaps[cmap]
    if not isinstance(cmap, mcolors.ListedColormap):
        cmap_colors = cmap(np.linspace(0, 1, 256))
        cmap = mcolors.ListedColormap(cmap_colors, name=cmap.name + '_alpha')
    else:
        cmap_colors = np.zeros((cmap.N, 4))
        cmap_colors[:, :3] = cmap.colors
    cmap_colors[:, 3] = alpha
    cmap.colors = cmap_colors
    return cmap


def visualize_source_estimate(stc, fig_fname, subjects_dir, subject='fsaverage',
                              colorbar=True, colorbar_label='Activation',
                              backend='matplotlib',
                              colormap='gnuplot', clim='auto',
                              transparent_overlay=True, transparent_background=False,
                              title=None, title_fontsize=30,
                              colorbar_fontsize=16, figsize=(8, 5),
                              highlight_labels=None, alpha=1.0):
    """Create a figure with medial and lateral views of a SourceEstimate.

    Parameters
    ----------
    stc : mne.SourceEstimate | str
        Either a SourceEstimate instance or a path that can be read by
        ``mne.read_source_estimate``.
    fig_fname : str
        Output filename for the assembled figure (PNG, etc.).
    subjects_dir : str
        FreeSurfer subjects directory for ``mne`` visualization functions.
    subject : str, optional
        Subject identifier (default: 'fsaverage').
    colorbar : bool, optional
        Whether to include a colorbar (default True).
    colorbar_label : str, optional
        Label for the colorbar (default 'Activation').
    backend : {'matplotlib', 'pyvistaqt'}, optional
        Visualization backend to use (default 'matplotlib').
    colormap : str | Colormap, optional
        Colormap to use for rendering the brain.
    clim : {'auto', 'symmetric', 'full-range'} | dict, optional
        Color limits specification; if a dict is provided it is passed to
        ``mne.SourceEstimate.plot`` as the ``clim`` argument.
    transparent_overlay : bool, optional
        If True, the generated brain images will have transparent
        color overlays when supported by the backend.
    transparent_background : bool, optional
        If True, the generated brain images will have transparent
        backgrounds when supported by the backend.
    title : str | None, optional
        Figure title.
    title_fontsize : int, optional
        Title font size.
    colorbar_fontsize : int, optional
        Font size for colorbar tick labels.
    figsize : tuple, optional
        Figure size in inches.
    highlight_labels : list | None, optional
        List of label names to highlight on the brain.
    alpha : float, optional
        Transparency for the brain rendering.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The assembled figure with medial and lateral views.
    """
    from mne import SourceEstimate
    from matplotlib.colorbar import make_axes
    if not isinstance(stc, SourceEstimate):
        stc = mne.read_source_estimate(stc)

    fig, axs = plt.subplots(2,
                            2,
                            gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
                            figsize=figsize,
                            subplot_kw={'frame_on': False, 'fc': 'white'})
    images = []
    if clim == 'symmetric':
        max_value = np.max(np.abs(stc.data))

        clim = {
            'kind': 'value',
            'lims': [-np.round(max_value, 2), 0,
                     np.round(max_value, 2)]
        }
        print("Clims set to:", clim)
    if clim == 'full-range':
        clim = {
            'kind': 'value',
            'lims': [np.round(np.min(stc.data), 2),
                     np.round(np.mean([np.min(stc.data), np.max(stc.data)]), 2),
                     np.round(np.max(stc.data), 2)]
        }
    for view in ['lat', 'med']:
        for hemi in ['lh', 'rh']:
            try:
                brain = stc.plot(
                    subject=subject,
                    colormap=colormap,
                    subjects_dir=subjects_dir,
                    views=view,
                    cortex=[(0.9, 0.9, 0.9), (0.8, 0.8, 0.8)],
                    hemi=hemi,
                    smoothing_steps=0,
                    backend=backend,
                    clim=clim,
                    time_label=None,
                    background='k' if transparent_background else 'w',
                    colorbar=False,
                    transparent=transparent_overlay,
                    alpha=alpha,
                    surface='inflated'
                )
                if highlight_labels is not None:
                    labels = mne.read_labels_from_annot(
                        subject,
                        parc='aparc_nimeg',
                        subjects_dir=subjects_dir,
                        hemi=hemi
                    )
                    labels_to_highlight = [label for label in labels
                                           if label.name in highlight_labels]
                    for label in labels_to_highlight:
                        brain.add_label(
                            label,
                            color='k',
                            alpha=0.8,
                            borders=3,
                            hemi=hemi
                        )
            except OverflowError:
                continue

            if backend == 'matplotlib':
                canvas = FigureCanvas(brain)
                canvas.draw()
                width, height = brain.get_size_inches() * brain.get_dpi()
                # use frombuffer for bytes -> array conversion
                img = np.frombuffer(canvas.tostring_rgb(),
                                    dtype='uint8').reshape((int(height),
                                                            int(width),
                                                            3))
                images.append(img)
            else:
                brain.toggle_interface(False)
                images.append(brain.screenshot())
                brain.close()
    for ax, img in zip(axs.flat, images):
        if transparent_background:
            nonwhite_pix = (img != 0).any(-1)
        else:
            nonwhite_pix = (img != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        img = img[nonwhite_row][:, nonwhite_col]
        rgba = 255 * np.ones((img.shape[0], img.shape[1], 4), dtype=int)
        rgba[..., :3] = img
        black_mask = np.all(img == 0, axis=-1)
        rgba[black_mask, 3] = 0 if transparent_background else 1
        img = rgba
        ax.tick_params(bottom=False,
                       left=False,
                       labelbottom=False,
                       labelleft=False)
        ax.imshow(img)

    if colorbar:
        cax, kw = make_axes(axs,
                            location='bottom',
                            orientation='horizontal',
                            shrink=0.35,
                            fraction=0.1,
                            pad=0.08,
                            aspect=15)
        cbar = mne.viz.plot_brain_colorbar(cax,
                                           clim=clim,
                                           colormap=colormap,
                                           orientation='horizontal',
                                           bgcolor='white',
                                           transparent=transparent_overlay)
        cbar.set_label(colorbar_label, fontsize=colorbar_fontsize)
        cbar.ax.tick_params(labelsize=colorbar_fontsize,
                            direction='out',
                            length=1.25,
                            width=0.25)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(0.25)
        cbar.outline.set_visible(True)
        if 'pos_lims' in clim:
            cbar.ax.set_xticks(cbar.ax.get_xticks()[[0, 2, 4]])
            cbar.ax.set_xticklabels(
                ['{:.2f}'.format(tick) for tick in cbar.ax.get_xticks()])

    if title is not None:
        fig.suptitle(title, x=0.5, fontsize=title_fontsize)

    fig.savefig(fig_fname, dpi=600, transparent=transparent_background)
    return fig


def add_sig(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c='k')
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom')
    return ax
