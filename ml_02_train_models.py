"""Run ridge regression and SVR models."""

import argparse
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from config import analysis_root
from filenames import FileNames

parser = argparse.ArgumentParser(description="Train ridge regression and SVR models.")
parser.add_argument('model_name', type=str,
                    help="Name of the model to train (e.g., 'ridge', 'svr').")
parser.add_argument("metric", type=str,
                    help="Functional connectivity metric to use (e.g., 'wpli', 'aec').")
parser.add_argument("--use-graph-features", action="store_true",
                    help="Whether to use graph features in addition to flat FC features.",
                    default=False)
parser.add_argument("--sensor-space", action="store_true",
                    help="Whether to use sensor space features instead of source space.",
                    default=False)
parser.add_argument("--use-dl-split", action="store_true",
                    help="Whether to use the same train/test split as the DL model training.",
                    default=False)
args = parser.parse_args()
model_name = args.model_name
if model_name not in ["ridge", "svr"]:
    raise ValueError(f"Invalid model name: {model_name}. Must be 'ridge' or 'svr'.")
metric = args.metric
if metric not in ["wpli", "aec"]:
    raise ValueError(f"Invalid metric: {metric}. Must be 'wpli' or 'aec'.")
use_graph_features = args.use_graph_features
sensor_space = args.sensor_space
use_dl_split = args.use_dl_split

# Load features and labels
fn = FileNames(analysis_root)
y = pd.read_csv(fn.included_subjects, header=0, index_col=0)["age"].values
X_fc = pd.read_csv(fn.con_features(metric, sensor_space=sensor_space),
                   index_col=0)
X_fc = X_fc.values
if use_graph_features:
    X_graph = pd.read_csv(fn.graph_features(metric, sensor_space=sensor_space),
                          index_col=0)
    X_graph = X_graph.values

scaler = StandardScaler()
# Outer cross validation loop
if use_dl_split:
    # Get the train/test split from the DL model training
    fname = ("/m/nbe/scratch/restmeg/dlfc_age/dl_models/tfr/"
             "sensor_aec_aec_minmax_scaled_12folds_meta.pkl")
    with open(fname, "rb") as f:
        dl_meta = pickle.load(f)
    test_folds = dl_meta.data["folds"][0]
    n_fc = X_fc.shape[0]
    train_folds = [np.setdiff1d(np.arange(n_fc), test_fold) for test_fold in test_folds]
    split = zip(train_folds, test_folds)
    print("Using the same train/test split as the DL model training.")
else:
    # Make random split
    cv_outer = KFold(n_splits=12, shuffle=True, random_state=42)
    split = cv_outer.split(X_fc)

results = pd.DataFrame(columns=["best_params", "mse_train",
                                "r2_train", "mae_train", "mse_test",
                                "r2_test", "mae_test"])
print(f"Training {model_name} model with {metric} features "
      f"{'including graph features ' if use_graph_features else ''}"
       f"in {'sensor space' if sensor_space else 'source space'}...")
start_time = time.time()
for fold_ix, (train_ix, test_ix) in enumerate(split):
    print(f"    Fold {fold_ix + 1}/10")
    X_train_fc, X_test_fc = X_fc[train_ix].copy(), X_fc[test_ix].copy()
    y_train, y_test = y[train_ix], y[test_ix]

    X_train_scaled = scaler.fit_transform(X_train_fc)
    X_test_scaled = scaler.transform(X_test_fc)
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    if use_graph_features:
        X_train_graph, X_test_graph = (X_graph[train_ix].copy(),
                                       X_graph[test_ix].copy())
        X_train = np.hstack((X_train_pca, X_train_graph))
        X_test = np.hstack((X_test_pca, X_test_graph))
    else:
        X_train, X_test = X_train_pca, X_test_pca

    # Scale features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model_name == "ridge":
        model = Ridge()
        param_grid = {
            'alpha': np.logspace(1, 4, 50)
        }
    # execute search
    elif model_name == "svr":
        model = SVR()
        param_grid = {
            'C': np.logspace(-1, 3, 10),
            'kernel': ['rbf'],
            'gamma': [1e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2]
        }
    else:
        raise ValueError(f"Invalid model name: {model_name}. "
                         f"Must be 'ridge' or 'svr'.")

    search = GridSearchCV(
        model,
        param_grid,
        scoring='neg_mean_absolute_error',
        cv=5,
        n_jobs=-1,
        refit=True)

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    results.loc[fold_ix] = {
        "best_params": search.best_params_,
        "mse_train": mean_squared_error(y_train, y_train_pred),
        "r2_train": r2_score(y_train, y_train_pred),
        "mae_train": mean_absolute_error(y_train, y_train_pred),
        "mse_test": mean_squared_error(y_test, y_test_pred),
        "r2_test": r2_score(y_test, y_test_pred),
        "mae_test": mean_absolute_error(y_test, y_test_pred)
    }
print("Training completed in "f"{(time.time() - start_time) / 60:.2f} minutes.")
results.loc["mean"] = results.mean(numeric_only=True)
results.loc["std"] = results.std(numeric_only=True)
results_path = fn.ml_results(model_name, metric, sensor_space=sensor_space,
                             use_graph_features=use_graph_features)
results.to_csv(results_path)
print("Results saved to CSV at: ", results_path)
print("Done!")
