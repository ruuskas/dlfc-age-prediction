"""Average functional connectivity in canonical frequency bands."""

import numpy as np
import pandas as pd

from config import analysis_root, freqs, fbands, source_data_dir, freq_mask
from filenames import FileNames, SubFileNames


# Init filenames
fn = FileNames(analysis_root)

# Load the list of included subjects
participants = pd.read_csv(fn.included_subjects, header=0, index_col=0)
subjects = participants["subject"].to_list()

# Load the AEC and WPLI data for all subjects
# Loop to reduce memory load
for sensor_space in [True, False]:
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
        aec_all.append(aec)
        wpli_all.append(wpli)
    aec_all = np.array(aec_all)[..., freq_mask]
    wpli_all = np.array(wpli_all)[..., freq_mask]
    for fband, (fmin, fmax) in fbands.items():
        freq_mask_fband = (freqs >= fmin) & (freqs < fmax)
        wpli_fband = wpli_all[..., freq_mask_fband].mean(axis=-1)
        np.save(fn.fc_data("wpli", "tril",
                           minmax=False, sensor_space=sensor_space,
                           fband=fband), wpli_fband)
        del wpli_fband

        aec_fband = aec_all[..., freq_mask_fband].mean(axis=-1)
        np.save(fn.fc_data("aec", "tril",
                           minmax=False, sensor_space=sensor_space,
                           fband=fband), aec_fband)
        del aec_fband
