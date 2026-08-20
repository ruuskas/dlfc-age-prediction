"""Collect functional connectivity matrices for all subjects.

1. Load the individual AEC and WPLI lower triangle matrices for all subjects.
2. Load the individual PSD arrays for all subjects.
3. Create combined matrices with AEC, WPLI, and AEC in the upper triangle and 
WPLI in the lower triangle.
3. Min-max normalize each metric independently.
4. Save the combined matrices and labels (ages) for all subjects.
"""

import numpy as np
import pandas as pd

from config import (source_data_dir, analysis_root, freq_mask)
from filenames import SubFileNames, FileNames
from utils import min_max_scale

# Init filenames
fn = FileNames(analysis_root)

# Load the list of included subjects
participants = pd.read_csv(fn.included_subjects, header=0, index_col=0)
subjects = participants["subject"].to_list()

# Save labels (participant ages)
labels = participants["age"].to_numpy()
n_subjects = len(labels)
np.save(fn.ages_all_subjects, labels)

# Load the AEC and WPLI data for all subjects
# Loop to reduce memory load
for sensor_space in [True, False]:
    for scale in [True, False]:
        aec_all = []
        wpli_all = []
        for subject in subjects:
            fn_sub = SubFileNames(source_data_dir, analysis_root, subject)
            if sensor_space:
                aec = np.load(fn_sub.sensor_aec)
                wpli = np.load(fn_sub.sensor_wpli)
            else:
                aec = np.load(fn_sub.source_aec)
                wpli = np.load(fn_sub.source_wpli)
            if scale:
                # Min-max scaling
                aec = min_max_scale(aec)
                wpli = min_max_scale(wpli)
            aec_all.append(aec)
            wpli_all.append(wpli)
        aec_all = np.array(aec_all)[..., freq_mask]
        wpli_all = np.array(wpli_all)[..., freq_mask]

        # Create and save combined matrices
        wpli_wpli = wpli_all + wpli_all.transpose(0, 2, 1, 3)
        np.save(fn.fc_data("wpli", "wpli",
                           minmax=scale, sensor_space=sensor_space),
                wpli_wpli)
        del wpli_wpli

        aec_wpli = aec_all + wpli_all.transpose(0, 2, 1, 3)
        np.save(fn.fc_data("aec", "wpli",
                           minmax=scale, sensor_space=sensor_space),
                aec_wpli)
        del aec_wpli

        aec_aec = aec_all + aec_all.transpose(0, 2, 1, 3)
        np.save(fn.fc_data("aec", "aec",
                           minmax=scale, sensor_space=sensor_space),
                aec_aec)
        del aec_aec
