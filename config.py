"""Set global configuration variables."""

from pathlib import Path

import numpy as np

# Directory paths
prefix = Path("/m/nbe/scratch/restmeg")
source_data_dir = prefix / "data/camcan-bids/processed/analysis/"
subjects_dir = prefix / "data/camcan-bids/derivatives/recon"
analysis_root = prefix / "dlfc_age"
working_directory = prefix / "dlfc_age" / "dlfc_age_pipeline"

freqs = np.geomspace(1, 100, 32)
# The following line is needed for the sparse freqs analysis only
freqs = np.array([f for f in freqs if f in np.geomspace(1, 100, 32)[1::2]])
freq_mask = np.array([1 if f in freqs else 0
                      for f in np.geomspace(1, 100, 32)]).astype(bool)

fbands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 100)
}

data_ids = [
    "source_wpli_wpli_minmax_scaled_12folds",
    "source_aec_wpli_minmax_scaled_12folds",
    "source_aec_aec_minmax_scaled_12folds",
    "sensor_wpli_wpli_minmax_scaled_12folds",
    "sensor_aec_wpli_minmax_scaled_12folds",
    "sensor_aec_aec_minmax_scaled_12folds"
]

models = [
    "m1_depthwise",
    "m2_sym3d",
    "m3_conv3d"
]
