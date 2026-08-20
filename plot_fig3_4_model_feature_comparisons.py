"""Plot the aggregated age prediction errors for the different model families and FC
datasets."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from matplotlib import font_manager
from statannotations.Annotator import Annotator

from config import analysis_root
from filenames import FileNames

bold = font_manager.FontProperties(family='Arial', weight='bold')
sns.set_theme(
    style='white', context='paper',
    palette=sns.color_palette('tab10')
)
plt.style.use('./style.mplstyle')
cm = 1 / 2.54

fn = FileNames(analysis_root)
results = pd.read_csv(fn.results_folds)

# https://coolors.co/a49e8d-504136-689689-b2e6d4-83e8ba
palette = {
    'Sensor': '#4A6C6F',
    'Source': '#AF5D63',
}

fig, ax = plt.subplots(ncols=1, figsize=(8 * cm, 6 * cm))
# The following shows all DL models across folds
# The barplot shows the mean across folds
# The whiskers show the standard deviation across folds and runs
results_subset = results.copy()
results_subset = results_subset[(results_subset['model'] == 'DL')]
sns.barplot(
    data=results_subset,
    y='mae_folds',
    x='fc',
    palette=palette,
    hue='level',
    gap=0.1,
    estimator='mean',
    capsize=0.15,
    width=0.8,
    linewidth=0,
    edgecolor='white',
    saturation=0.6,
    errorbar=('sd', 1),
    err_kws={'color': (0.1, 0.1, 0.1), 'linewidth': 1.5},
    ax=ax,
)

ax.grid(axis='y', linestyle='-', alpha=0.5, linewidth=0.5)
ax.set_ylabel('Mean Absolute Error (years)')
ax.tick_params(axis='x', labelsize=8, pad=4)
ax.set_xlabel('')
ax.set_ylim(0, 12)

# Label each bar with its numerical value
for container in ax.containers:
    ax.bar_label(
        container, fmt='%.1f', label_type='center',
        padding=20, fontsize=8, color='white',
    )

sns.despine(ax=ax)

annotator = Annotator(ax, [(("wPLI", "Source"), ("wPLI", "Sensor")),
                           (("AEC", "Source"), ("AEC", "Sensor")),
                           (("AEC + wPLI", "Source"), ("AEC + wPLI", "Sensor")),
                           ],
                      data=results_subset, x='fc', y='mae_folds', hue='level')
# annotator = Annotator(ax, [("wPLI", "AEC"), ("wPLI", "AEC + wPLI"),
#                            ("AEC", "AEC + wPLI")],
#                       data=results_subset, x='fc', y='mae_folds')
annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                    verbose=2,
                    text_offset=1, fontsize=8,
                    comparisons_correction='Bonferroni',
                    correction_format='replace',
                    hide_non_significant=True,
                    line_height=0.02,
                    line_offset=0,
                    line_width=1.5,)
annotator.apply_and_annotate()


comparisons = [
    [("wPLI", "Source"), ("wPLI", "Sensor")],
    [("AEC", "Source"), ("AEC", "Sensor")],
    [("AEC + wPLI", "Source"), ("AEC + wPLI", "Sensor")],
    [("wPLI", "AEC")],
    [("wPLI", "AEC + wPLI")],
    [("AEC", "AEC + wPLI")]
]
for comparison in comparisons:
    if len(comparison) == 2:
        group1 = results_subset[
            (results_subset['fc'] == comparison[0][0]) &
            (results_subset['level'] == comparison[0][1])
        ]['mae_folds']
        group2 = results_subset[
            (results_subset['fc'] == comparison[1][0]) &
            (results_subset['level'] == comparison[1][1])
        ]['mae_folds']
    else:
        group1 = results_subset[
            (results_subset['fc'] == comparison[0][0])
        ]['mae_folds']
        group2 = results_subset[
            (results_subset['fc'] == comparison[0][1])
        ]['mae_folds']
    res = mannwhitneyu(group1, group2, alternative='two-sided')
    print(f"Mann-Whitney U test result for {comparison}: U={res.statistic}, "
          f"p-value={res.pvalue * len(comparisons):.4g} (Bonferroni corrected)")

# Legend for model type
plt.legend(
    title=None, loc='upper right', frameon=False,)
plt.tight_layout()
plt.savefig(fn.figure_dir / 'fig4_aec_wpli_level.png', dpi=600)
plt.savefig(fn.figure_dir / 'fig4_aec_wpli_level.svg', transparent=True)
plt.show()

# Print the relevant numbers
print("Mean and standard deviation of MAE for each FC:")
print(results_subset.groupby("fc")["mae_folds"].agg(["mean", "std", "count"])
      .reset_index().to_string(index=False))
print("\nMean and standard deviation of MAE for each level:")
print(results_subset.groupby(["level", "fc"])["mae_folds"].agg(["mean", "std", "count"])
      .reset_index().to_string(index=False))

#%%
fig, ax = plt.subplots(figsize=(7 * cm, 6 * cm))
palette = {
    'ML': '#E18335',
    'DL': '#1E91D6',
}

results_subset = results.copy()
results_subset.loc[results_subset['model'].isin(['Ridge', 'SVR']), "model"] = 'ML'
results_subset = results_subset[results_subset["level"] == "Source"]
res = mannwhitneyu(
    results_subset[results_subset["model"] == "ML"]["mae_folds"],
    results_subset[results_subset["model"] == "DL"]["mae_folds"],
    alternative='two-sided'
)
print(f"Mann-Whitney U test result: U={res.statistic}, p-value={res.pvalue}")
# The following shows the mean across folds, runs, model types and features
sns.violinplot(
    data=results_subset,
    y='mae_folds',
    x='model',
    palette=palette,
    hue='model',
    legend=False,
    gap=0,
    inner='quart',
    width=0.6,
    linewidth=1,
    fill=True,
    saturation=0.6,
    ax=ax,
)

# Mean points
means = results_subset.groupby("model")["mae_folds"].mean()

plt.scatter(
    range(len(means)),
    means,
    color='w',
    marker='d',
    s=10,
    zorder=5,
    label='Mean'
)

ax.grid(axis='y', linestyle='-', alpha=0.5, linewidth=0.5)
ax.set_ylabel('Mean Absolute Error (years)')
ax.tick_params(axis='x', labelsize=8, pad=4)
ax.set_xlabel('')
ax.set_ylim(5, 15)
ax.set_xticklabels(["FC-CNN", "SVR and RR"], rotation=0, ha='center')

# Label each bar with its numerical value
for container in ax.containers:
    ax.bar_label(
        container, fmt='%.1f', label_type='center',
        padding=20, fontsize=8, color='white',
    )

sns.despine(ax=ax)
annotator = Annotator(ax, [("DL", "ML")], data=results_subset, x='model',
                      y='mae_folds')
annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',
                    verbose=2,
                    text_offset=1, fontsize=8)
annotator.apply_and_annotate()

plt.tight_layout()
plt.savefig(
    fn.figure_dir / 'fig3_ml_dl.png', dpi=600)
plt.savefig(fn.figure_dir / 'fig3_ml_dl.svg', transparent=True)
plt.savefig(fn.figure_dir / 'fig3_ml_dl.pdf', transparent=False)
plt.show()

print("\nMean and standard deviation of MAE for model types:")
print(results_subset.groupby(["model"])["mae_folds"].agg(["mean", "std", "count"])
      .reset_index().to_string(index=False))
