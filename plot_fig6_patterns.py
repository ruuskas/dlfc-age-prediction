"""Plot the age extracted patterns."""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import seaborn as sns
from matplotlib import font_manager
from viz_utils import visualize_source_estimate, set_colormap_alpha

from config import subjects_dir, analysis_root

bold = font_manager.FontProperties(family='Arial', weight='bold')

sns.set_theme(
    style='white', context='paper',
    palette=sns.color_palette('tab10')
)
plt.style.use('./style.mplstyle')
cm = 1 / 2.54

data_ids = ["aec_aec", "wpli_wpli", "aec_wpli"]
data_id_to_title = {
    "aec_aec": "AEC",
    "wpli_wpli": "wPLI",
    "aec_wpli": "AEC + wPLI"
}

for data_id in data_ids:
    stc = mne.read_source_estimate(data_dir / f"eero/{data_id}_row-col_80_run_2_stc_activation")
    fig_fname = analysis_root / "figures/fig5_{data_id}_patterns.svg"
    visualize_source_estimate(
        stc,
        fig_fname,
        subjects_dir,
        subject='fsaverage',
        colorbar=False,
        colormap=set_colormap_alpha("RdBu_r", 0.8),
        backend='pyvistaqt',
        clim=dict(kind='value', pos_lims=[0.0, 0.5, 1.0]),
        transparent_overlay=True,
        title=data_id_to_title[data_id],
        title_fontsize=10,
        figsize=(6*cm, 4.5*cm)
    )

