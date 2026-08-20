"""Create hyperparameter JSON files for different models.

This script generates hyperparameter combinations for different models and saves
them to JSON files."""

import json
import os

from sklearn.model_selection import ParameterGrid

from config import working_directory, analysis_root
from filenames import FileNames

fn = FileNames(analysis_root)
os.chdir(working_directory)

#%% symmetric model
param_grid = {
    'model_id': ["m2_sym3d"],
    'n_latent_dense': [32],
    'n_latent_cross': [1],
    'n_latent': [8],
    'spatial_dropout': [0.1],
    'dropout': [0.4],
    'attention_ratio': [4],  # set 0, if no attention
    'attention2': [False],
    'loss': ['MAE'],
    'learn_rate': [1e-3],
    'batch_size': [50],
    'iterations': [100],
    'early_stop': [10],
    'stddev': [0.05],
    'depthwise': [True],
    'l1_lambda': [0.0],
    'run_id': ["run1", "run2", "run3"],
 }
param_combinations = list(ParameterGrid(param_grid))
print(f"Saving {len(param_combinations)} hyperparameter combinations for "
      f"the symmetric model.")

# Save to a JSON file
with open(fn.hyperparam_file("m1_depthwise"), 'w') as json_file:
    json.dump(param_combinations, json_file, indent=4)

#%% depthwise model
param_grid = {
    'model_id': ["m1_depthwise"],
    'n_latent_dense': [32],
    'n_latent_cross': [1],
    'n_latent1': [16],
    'n_latent2': [0],
    'spatial_dropout': [0.1],
    'dropout': [0.2],
    'attention_rate': [4],  # set 0, if no attention
    'attention2': [False],
    'loss': ['MAE'],
    'learn_rate': [1e-3],
    'batch_size': [50],
    'iterations': [100],
    'early_stop': [10],
    'stddev': [0.1],
    'depthwise': [True],
    'l1_lambda': [0.0],
    'run_id': ["run1", "run2", "run3"],
 }
param_combinations = list(ParameterGrid(param_grid))
print(f"Saving {len(param_combinations)} hyperparameter combinations for "
      f"the weighted sum model.")

# Save to a JSON file
with open(fn.hyperparam_file('m2_sym3d'), 'w') as json_file:
    json.dump(param_combinations, json_file, indent=4)

#%% 3d convolutional model
param_grid = {
    'model_id': ["m3_conv3d"],
    'n_latent_dense': [32],
    'n_latent': [32],
    'spatial_dropout': [0.1],
    'dropout': [0.4],
    'attention_ratio': [0],  # no attention
    'loss': ['MAE'],
    'learn_rate': [1e-3],
    'batch_size': [50],
    'iterations': [100],
    'early_stop': [10],
    'stddev': [0.1],
    'l1_lambda': [0.0],
    'run_id': ["run1", "run2", "run3"],
 }
param_combinations = list(ParameterGrid(param_grid))
print(f"Saving {len(param_combinations)} hyperparameter combinations for "
      f"the 3D convolutional model.")

# Save to a JSON file
with open(fn.hyperparam_file('m3_conv3d'), 'w') as json_file:
    json.dump(param_combinations, json_file, indent=4)
