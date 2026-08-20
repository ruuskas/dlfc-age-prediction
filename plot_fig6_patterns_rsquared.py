"""Plot the R-squared of the ablated model."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

from config import analysis_root

bold = font_manager.FontProperties(family='Arial', weight='bold')
sns.set_theme(
    style='white', context='paper',
    palette=sns.color_palette('tab10')
)
plt.style.use('./style.mplstyle')
cm = 1 / 2.54

data = pd.read_excel(analysis_root / "ablation_model_results.ods",
                     skiprows=21)
n = 3
fig, axes = plt.subplots(
    1, n,
    figsize=(38*cm, 0.3 * n),  # wide and relatively narrow
    squeeze=False
)
colors = ["#64748B", "#2A9D8F"][::-1]

for i, (_, row) in enumerate(data.iterrows()):
    ax = axes[0, i]

    values = [row["R2 ablation"], row["R2 orig"]]
    labels = ["R2 ablation", "R2 orig"]

    ax.barh(labels, values, height=0.7, color=colors)
    ax.set_xlim(0.3, 1)  # remove/change if R² isn't in [0, 1]

    # Add values at the end of each bar
    for j, value in enumerate(values):
        ax.text(value + 0.02, j, f"{value:.2f}", va="center", fontsize=22)

    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_xticks([])  # remove x-ticks
    ax.set_yticks([])  # remove y-ticks

plt.tight_layout()
plt.show()
plt.savefig(analysis_root / "figures/fig5_ablation_r2.svg", dpi=300,
            bbox_inches='tight')
