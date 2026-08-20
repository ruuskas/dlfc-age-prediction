import pickle

import numpy as np
import pandas as pd

from filenames import FileNames
import config


fn = FileNames(config.analysis_root)
ages = np.load(fn.ages_all_subjects)

# Get the CV fold indices from the DL model training
fname = ("/m/nbe/scratch/restmeg/dlfc_age/dl_models/tfr/"
         "sensor_aec_aec_minmax_scaled_12folds_meta.pkl")
with open(fname, "rb") as f:
    dl_meta = pickle.load(f)
test_folds = dl_meta.data["folds"][0]

train_folds = [np.setdiff1d(np.arange(len(ages)), test_fold) for test_fold in
               test_folds]
split = zip(train_folds, test_folds)
print("Using the same train/test split as the DL model training.")
maes = []
for fold_ix, (train_ix, test_ix) in enumerate(split):
    print(f"Fold {fold_ix + 1}/12")
    y_train, y_test = ages[train_ix], ages[test_ix]
    mean_age = np.mean(y_train)
    mae = np.mean(np.abs(y_test - mean_age))
    maes.append(mae)

print("Dummy model MAE:", np.mean(maes), "±", np.std(maes))
df = pd.DataFrame({
    "mae_folds": maes})
df.to_csv(fn.dummy_model_results, index=False)
