"""Plot the age prediction errors for the different models."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib import font_manager

from config import analysis_root
from filenames import FileNames

# palette https://coolors.co/2b193d-2c365e-484d6d-4b8f8c-c5979d
bold = font_manager.FontProperties(family='Arial', weight='bold')

sns.set_theme(
    style='white', context='paper',
    palette=sns.color_palette('tab10')
)
plt.style.use('./style.mplstyle')
cm = 1 / 2.54

fn = FileNames(analysis_root)
results = pd.read_csv(fn.results_folds)

dummy_results = pd.read_csv(fn.dummy_model_results)
dummy_results['model'] = 'Dummy'
dummy_results['features'] = 'Mean Age'
dummy_results['level'] = ''
dummy_results['fc'] = ''
results = pd.concat([results, dummy_results], ignore_index=True)

results = results.replace({
    'features': {
        "Asymmetric": "Depthwise"
    }
})

results['label'] = (
    results['model'] + ' – '
    + results['features'] + ' – '
    + results['level'] + ' '
    + results['fc']
)

# Sort labels by mean MAE for ordering
label_order = (
    results.groupby('label')['mae_folds']
    .mean()
    .sort_values()
    .index
    .tolist()
)

# Palette by model type
palette = {
    'Ridge': '#E18335',
    'SVR': '#E18335',
    'DL': '#1E91D6',
    'Dummy': '#A0A0A0',
    "FC-CNN": '#1E91D6',
    "ML models": '#E18335',
}

fig, ax = plt.subplots(figsize=(12 * cm, 16 * cm))
fig.subplots_adjust(left=0.42, right=0.9, bottom=0.06, top=0.95)

meanprops = dict(
    marker='D',
    markerfacecolor='w',
    markeredgecolor='#333333',
    markersize=2.5,
    markeredgewidth=0.6,
)

sns.boxplot(
    data=results,
    y='label',
    x='mae_folds',
    order=label_order,
    palette=palette,
    hue="model",
    linewidth=1,
    saturation=0.6,
    width=0.5,
    fliersize=1,
    whis=1.5,  # 1.5 times the IQR
    ax=ax,
    showmeans=True,
    meanline=False,
    meanprops=meanprops,
)

ax.legend([], [], frameon=False)
ax.set_ylabel('')
ax.set_xlabel('Mean Absolute Error (years)')
ax.tick_params(axis='y')
ax.set_yticklabels([])

for i in range(len(label_order)):
    row = results[results['label'] == label_order[i]]
    mean = row['mae_folds'].mean()
    ax.text(1.05, i, f"{mean:.2f}",
            transform=ax.get_yaxis_transform(),
            horizontalalignment='left',
            fontsize=6, fontproperties=bold)
    kwargs = dict(
        transform=ax.get_yaxis_transform(),
        fontsize=7,
        horizontalalignment='left',
    )
    y = i + 0.22
    features = f"{row['level'].iloc[0]} {row['fc'].iloc[0]}"
    if (model := row['model'].iloc[0]) == "DL":
        model = row['features'].iloc[0]
    elif "Graph" in row["features"].iloc[0]:
        features = f"{features} + Graph"
    ax.text(-0.8, y, model, **kwargs)
    ax.text(-0.45, y, features, **kwargs)
ax.text(1.05, -1, "Mean", horizontalalignment='left',
        fontsize=8, transform=ax.get_yaxis_transform(), fontproperties=bold)
ax.text(-0.8, -1, "Model", horizontalalignment='left',
        fontsize=8, transform=ax.get_yaxis_transform(), fontproperties=bold)
ax.text(-0.45, -1, "Features", horizontalalignment='left',
        fontsize=8, transform=ax.get_yaxis_transform(), fontproperties=bold)
sns.despine(ax=ax)
fig.legend(
    handles=[Patch(facecolor=sns.desaturate(palette[m], 0.6), label=m)
             for m in ["FC-CNN", "ML models"]],
    loc='upper right', bbox_to_anchor=(0.9, 0.96), fontsize=8)

plt.savefig(
    fn.figure_dir / 'fig4_age_prediction_errors.png', dpi=600)
plt.savefig(fn.figure_dir / 'fig4_age_prediction_errors.svg',
            transparent=True)
plt.savefig(fn.figure_dir / 'fig4_age_prediction_errors.pdf')
plt.show()
