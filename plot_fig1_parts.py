"""Plot the graphical elements for Figure 1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib import colors as mcolors

from config import analysis_root
from filenames import FileNames
from viz_utils import (visualize_connectivity_circular, desaturate,
                       numpy_to_stc, visualize_source_estimate)
from utils import min_max_scale

fn = FileNames(analysis_root)
bold = font_manager.FontProperties(family='Arial', weight='bold')

sns.set_theme(
    style='white', context='paper',
    palette=sns.color_palette('tab10')
)
plt.style.use('./style.mplstyle')
cm = 1 / 2.54

#%% Plot age histogram
included_subjects = pd.read_csv(
    "/m/nbe/scratch/restmeg/data/camcan-bids/processed/aggregated/"
    "mean_flip_cwt_morlet_5_aparc_nimeg/included_subjects.csv", index_col=0)
participants = pd.read_csv("/m/nbe/scratch/restmeg/data/"
                           "camcan-bids/rawdata/participants.tsv", sep='\t')
participants.rename(columns={'participant_id': 'subject'}, inplace=True)
participants = participants.merge(included_subjects, on='subject', how='inner',
                                  suffixes=(None, "_included"))
bins = np.arange(18, 98, 10)
males = participants[participants['sex'] == 'MALE']['age']
females = participants[participants['sex'] == 'FEMALE']['age']
print(f"Number of males: {len(males)}")
print(f"Number of subjects: {len(participants)}")

fig, ax = plt.subplots(figsize=(4.5*cm, 4*cm))
red = desaturate('tab:red', factor=0.5)
blue = desaturate('tab:blue', factor=0.5)
ax.hist([males, females],
        label=['Men', 'Women'],
        stacked=True,
        bins=bins,
        color=[blue, red])
ax.set_xticks(bins)
ax.set_xlabel('Age (y.)', fontsize=7)
ax.set_ylabel('Number of participants', fontsize=7)
ax.set_title(f'Participants (n = {len(participants)})', fontsize=8, pad=6)
ax.set_ylim(0, 128)
ax.set_xlim(17, 89)
lines, labels = ax.get_legend_handles_labels()
lgd = fig.legend(lines, labels, bbox_to_anchor=(0.585, 0.88), loc='upper left', alignment='left',
                 framealpha=0, fontsize=7)
fig.tight_layout()
fig.savefig(f'{fn.figure_dir}/fig1_camcan_participants.svg', bbox_extra_artists=(lgd,),
            bbox_inches='tight', transparent=True)
fig.savefig(f"{fn.figure_dir}/fig1_camcan_participants.png", bbox_extra_artists=(lgd,),
            bbox_inches='tight', dpi=600)

#%% Plot MEG helmet
import mne
bids_dir = Path("/m/nbe/scratch/restmeg/data/camcan-bids")
subject = 'sub-CC120208'
raw = mne.io.read_raw_fif(bids_dir / f"derivatives/maxfilter/{subject}/meg/{subject}_task-rest_proc-tsss_meg.fif")
subjects_dir = Path("/m/nbe/scratch/restmeg/data/camcan-bids/derivatives/recon")
trans_fname = (bids_dir / f"processed/analysis/{subject}/trans/{subject}-trans.fif")
bem = mne.read_bem_surfaces(subjects_dir / f"{subject}/bem/{subject}-head.fif")

fig = mne.viz.plot_alignment(info=raw.info, trans=trans_fname, subject=subject, subjects_dir=subjects_dir,
                             surfaces=dict(head=0.5, pial=0.8), meg=dict(helmet=0.1, sensors=0.4), dig=False,
                             bem=bem, sensor_colors=desaturate('tab:gray', 0.5))
fig.plotter.set_background('white')
mne.viz.set_3d_view(fig,
                    azimuth=50,
                    elevation=70,
                    roll=-110,
                    distance=0.6,
                    focalpoint=(0, 0, 0))
fig.plotter.screenshot(fn.figure_dir / 'fig1_sensor_alignment.png', transparent_background=True)
fig.plotter.close()

#%% Plot parcellation
fig, axs = plt.subplots(2, 2, gridspec_kw={'wspace': 0, 'hspace': 0.15},
                        figsize=(5.5*cm, 4*cm),
                        subplot_kw={'frame_on': False, 'fc': 'white'})
images = []
for view in ['lat', 'med']:
    for hemi in ['lh', 'rh']:
        brain = mne.viz.Brain('fsaverage',
                              subjects_dir=subjects_dir,
                              cortex=[(0.98, 0.98, 0.98), (0.85, 0.85, 0.85)],
                              surf='inflated',
                              background='black',
                              views=view, hemi=hemi)
        n_vert = brain.geo[hemi].nn.shape[0]
        verts = np.full(n_vert, -1)
        labels = mne.read_labels_from_annot('fsaverage',
                                            'aparc_nimeg',
                                            subjects_dir=subjects_dir,
                                            sort=False)
        labels = [label for label in labels if label.hemi == hemi]
        ctab = np.zeros((len(labels), 5))
        for i, label in enumerate(labels):
            verts[label.vertices] = i + 1
            ctab[i, :3] = np.round(
                np.array(desaturate(label.color[:3], 0.3)) * 255,
                0).astype(int)
            if np.all(ctab[i, :3] == 255):
                ctab[i, :3] = 250
            ctab[i, 3] = 150
            ctab[i, 4] = i + 1
        unknown_idx = [i for i, label in enumerate(labels) if 'unknown' in label.name][0]
        verts[verts == -1] = unknown_idx + 1
        brain.add_annotation((verts, ctab), borders=False)
        brain.add_annotation("aparc_sub_nimeg", borders=True, color='k',
                             alpha=0.2)
        images.append(brain.screenshot())
        brain.close()
for ax, img in zip(axs.flat, images):
    nonwhite_pix = (img != 0).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    img = img[nonwhite_row][:, nonwhite_col]
    # The following removes the white background by setting the alpha channel to 0 for white pixels
    rgba = 255 * np.ones((img.shape[0], img.shape[1], 4), dtype=int)
    rgba[..., :3] = img
    white_mask = np.all(img == 0, axis=-1)
    rgba[white_mask, 3] = 0
    img = rgba
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.imshow(img)
fig.savefig(fn.figure_dir / f'fig1_parcellation.svg', dpi=600, transparent=True)
fig.savefig(fn.figure_dir / f'fig1_parcellation.png', dpi=600)

#%% Plot some connectivity matrices
freqs = np.geomspace(1, 100, 32)
freq_indices = np.array([8, 10, 13, 16])
for freq_idx in freq_indices:
    method_cons = []
    for method in ['wpli', 'aec']:
        example_con_sub = np.load(
            bids_dir / f"processed/analysis/{subject}/connectivity/{subject}_task-rest_proc-tsss_meg_mean_flip_connectivity-"
                       f"matrix_{method}_cwt_morlet_n-cycles-5_broad_1_000-100_000Hz_aparc_nimeg_averaged.npy")
        example_con_sub = min_max_scale(example_con_sub)
        example_con_sub = example_con_sub[..., freq_idx]
        fig, ax = plt.subplots(figsize=(2.3*cm, 2.3*cm))
        ax.imshow(example_con_sub if method == 'aec' else example_con_sub.T,
                  cmap='Blues',
                  vmin=0,
                  vmax=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(True)
        fig.savefig(fn.figure_dir / f'fig1_{subject}_{freqs[freq_idx]:.2f}Hz_{method}_matrix.svg',
                    dpi=600, transparent=True)
        method_cons.append(example_con_sub)
    shared_con = method_cons[0].T + method_cons[1]
    fig, ax = plt.subplots(figsize=(2.3*cm, 2.3*cm))
    ax.imshow(shared_con, cmap='Blues', vmin=0, vmax=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
    fig.savefig(fn.figure_dir / f'fig1_{subject}_{freqs[freq_idx]:.2f}Hz_shared_con_matrix.svg', dpi=600, transparent=True)

#%% Plot some "fake" ablated connectivity
freq_idx = 16
example_con_sub = np.load(
            bids_dir / f"processed/analysis/{subject}/connectivity/{subject}_task-rest_proc-tsss_meg_mean_flip_connectivity-"
                       f"matrix_{method}_cwt_morlet_n-cycles-5_broad_1_000-100_000Hz_aparc_nimeg_averaged.npy")
example_con_sub = min_max_scale(example_con_sub)
example_con_sub = example_con_sub[..., freq_idx]
example_con_sub[0:20, ...] = 0
example_con_sub[50:60, ...] = 0
example_con_sub[85:90, ...] = 0
example_con_sub[..., 0:20] = 0
example_con_sub[..., 50:60] = 0
example_con_sub[..., 85:90] = 0
fig, ax = plt.subplots(figsize=(2.3*cm, 2.3*cm))
ax.imshow(example_con_sub, cmap='Blues', vmin=0, vmax=0.8)
ax.set_xticks([])
ax.set_yticks([])
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(True)
fig.savefig(fn.figure_dir / f'fig1_{subject}_{freqs[freq_idx]:.2f}Hz_ablated_con_matrix.svg', dpi=600, transparent=True)

#%% Plot connectivity circle
freq_idx = 16
example_con_sub = np.load(bids_dir / f"processed/analysis/{subject}/connectivity/{subject}_task-rest_proc-tsss_meg_"
                         f"mean_flip_connectivity-matrix_wpli_cwt_morlet_n-cycles-5_broad_1_000-100_000Hz_"
                         f"aparc_nimeg_averaged.npy")
example_con_sub = min_max_scale(example_con_sub)
example_con_sub = example_con_sub[..., freq_idx]
con_fname = "example_con.npy"
np.save(fn.figure_dir / con_fname, example_con_sub)

fig = plt.figure(figsize=(6*cm, 5*cm))
fig = visualize_connectivity_circular(
    fn.figure_dir / con_fname,
    fn.figure_dir / f'fig1_{subject}_11Hz_con.svg',
    subjects_dir,
    'fsaverage',
    parcellation='aparc_nimeg',
    title=f'',
    colormap='Blues',
    n_lines=None,
    names='grouped',
    padding=12,
    fontsize_title=7,
    fontsize_colorbar=6,
    colorbar_size=0.3,
    node_width=3,
    node_linewidth=0.2,
    linewidth=0.5,
    fontsize_names=6,
    fig=fig,
    vmin=0.,
    vmax=0.8,
    colorbar=False,
    desaturate_factor=0.3,
)
fig.savefig(fn.figure_dir / f'fig1_{subject}_11Hz_con.png', dpi=600, transparent=True)


fig = plt.figure(figsize=(1.3*cm, 2*cm))

# Create a dedicated axis for the colorbar
cax = fig.add_axes([0.3, 0.05, 0.15, 0.9])  # [left, bottom, width, height]
norm = mcolors.Normalize(vmin=0, vmax=0.8)
cmap = plt.cm.Blues
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
# Label and ticks
cbar.set_label('WPLI (scaled)', fontsize=7, labelpad=3)
cbar.set_ticks([0.0, 0.4, 0.8])
# Style ticks
cbar.ax.tick_params(labelsize=6, length=2, width=0.5, direction='out')

# Optional: thin outline
cbar.outline.set_linewidth(0.5)

fig.savefig(fn.figure_dir / 'fig1_con_matrix_colorbar.svg',
            transparent=True, dpi=600)


#%% Plot node strength
ns = np.mean(example_con_sub + example_con_sub.T, axis=0)
stc = numpy_to_stc(ns, 'aparc_nimeg', subjects_dir=subjects_dir)
# cmap = set_colormap_alpha(plt.get_cmap("Blues"), alpha=0.5)
clim = dict(kind='value', lims=[0.2, 0.3, 0.4])

fig = visualize_source_estimate(
    stc,
    fn.figure_dir / f'fig1_{subject}_11Hz_node_strength.svg',
    subjects_dir,
    colorbar=False,
    colorbar_label='Node strength',
    colormap='Blues',
    backend='pyvistaqt',
    clim=clim,
    title='',
    transparent_overlay=True,
    transparent_background=True,
    colorbar_fontsize=7,
    figsize=(4*cm, 4*cm),
    alpha=1
)

fig = plt.figure(figsize=(1.3*cm, 2*cm))
cax = fig.add_axes([0.3, 0.05, 0.15, 0.9])  # [left, bottom, width, height]
norm = mcolors.Normalize(vmin=0.2, vmax=0.4)
cmap = plt.cm.Blues
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
# Label and ticks
cbar.set_label('Node strength', fontsize=7, labelpad=3)
cbar.set_ticks([0.2, 0.3, 0.4])
# Style ticks
cbar.ax.tick_params(labelsize=6, length=2, width=0.5, direction='out')

# Optional: thin outline
cbar.outline.set_linewidth(0.5)

fig.savefig(fn.figure_dir / 'fig1_node_strength_colorbar.svg',
            transparent=True, dpi=600)


#%% Placeholder informative patterns
ns = np.mean(example_con_sub + example_con_sub.T, axis=0)
ns[0:20] = 0
ns[50:60] = 0
ns[85:90] = 0
stc = numpy_to_stc(ns, 'aparc_nimeg', subjects_dir=subjects_dir)
# cmap = set_colormap_alpha(plt.get_cmap("Blues"), alpha=0.5)
clim = dict(kind='value', lims=[0.2, 0.3, 0.4])

fig = visualize_source_estimate(
    stc,
    fn.figure_dir / f'fig1_{subject}_11Hz_patterns.svg',
    subjects_dir,
    colorbar=False,
    colorbar_label='Node strength',
    colormap='Blues',
    backend='pyvistaqt',
    clim=clim,
    title='',
    transparent_overlay=True,
    transparent_background=True,
    colorbar_fontsize=7,
    figsize=(4*cm, 4*cm),
    alpha=1
)

