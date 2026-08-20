"""Find all the results from the experiments and collect into a single file."""

import numpy as np
import pandas as pd

from config import analysis_root
from filenames import FileNames

fn = FileNames(analysis_root)
results = pd.DataFrame(columns=["model", "fc", "features", "level", "run",
                                "mae", "rmse", "r2", "mae_folds", "rmse_folds"])
counter = 0
# DL results
# The model names here should correspond to the model scope names in mneflow.
for dl_model in ["depthwise_12folds", "symmetric_12folds", "conv3d_12folds"]:
    dl_results = pd.read_csv(fn.tfr_dir / "models" / f"{dl_model}_log.csv")
    for metric in ["wPLI", "AEC", "AEC + wPLI"]:
        for level in ["Sensor", "Source"]:
            level_str = f"{level.lower()}"
            if metric in ["wPLI", "AEC"]:
                metric_str = f"{metric.lower()}_{metric.lower()}"
            elif metric == "AEC + wPLI":
                metric_str = f"aec_wpli"
            else:
                raise ValueError(f"Invalid metric: {metric}")
            data_id = f"{level_str}_{metric_str}_minmax_scaled_12folds"
            rows = dl_results[dl_results["data_id"] == data_id]
            if dl_model == "depthwise_12folds":
                features = "Depthwise"
            elif dl_model == "symmetric_12folds":
                features = "Symmetric"
            elif dl_model == "conv3d_12folds":
                features = "3D Convolution"
            else:
                raise ValueError(f"Invalid model: {dl_model}")
            if rows.empty:
                print(f"Warning: No results found for {dl_model}, "
                      f"{metric}, {level}")
                continue
            elif len(rows) != 3:
                print(f"Warning: Incorrect number of results found for {dl_model}, "
                      f"{metric}, {level}. Expected 3, got {len(rows)}. Using the first row.")
                row = rows.iloc[0:1]
            for i in range(len(rows)):
                row = rows.iloc[i]
                if isinstance(row["cv_losses"], str):
                    cv_losses = row["cv_losses"]
                    cv_losses = [np.array([float(x) for x in
                                           cv_losses.strip("[]").split(",")])]
                    cv_metrics = row["cv_metrics"]
                    cv_metrics = [np.array([float(x) for x in
                                            cv_metrics.strip("[]").split(",")])]
                else:
                    cv_losses = row["cv_losses"]
                    cv_metrics = row["cv_metrics"]
                row_results = pd.DataFrame({
                    "model": "DL",
                    "fc": metric,
                    "features": features,
                    "level": level,
                    "run": i+1,
                    "mae": row["validation loss"],
                    "rmse": row["validation metric"],
                    "r2": row["r2"],
                    "fold": [np.arange(len(cv_losses[0]))],
                    "mae_folds": cv_losses,
                    "rmse_folds": cv_metrics
                })
                results = pd.concat([results, row_results],
                                    ignore_index=True)
                if row_results.empty:
                    print(f"Warning: No results found for {dl_model}, "
                          f"{metric}, {level}")
                counter += 1
print("number of DL runs checked:", counter)

# ML results
for ml_model in ["Ridge", "SVR"]:
    for metric in ["wPLI", "AEC"]:
        for features in ["FC", "FC + Graph"]:
            for level in ["Sensor", "Source"]:
                sensor_space = True if level == "Sensor" else False
                use_graph_features = True if features == "FC + Graph" else False
                ml_results = pd.read_csv(fn.ml_results(
                    ml_model.lower(), metric.lower(), sensor_space=sensor_space,
                    use_graph_features=use_graph_features
                ), index_col=0)
                folds = ml_results[~ml_results.index.isin(["mean", "std"])]
                folds.index = folds.index.astype(int)
                folds.sort_index(inplace=True)
                row_results = pd.DataFrame({
                    "model": ml_model,
                    "fc": metric,
                    "features": features,
                    "level": level,
                    "run": "N/A",
                    "mae": ml_results.loc["mean", "mae_test"],
                    "rmse": np.sqrt(ml_results.loc["mean", "mse_test"]),
                    "r2": ml_results.loc["mean", "r2_test"],
                    "fold": [folds.index.values],
                    "mae_folds": [folds["mae_test"].values],
                    "rmse_folds": [np.sqrt(folds["mse_test"].values)]})
                results = pd.concat([results, row_results],
                                    ignore_index=True)
                if row_results.empty:
                    print(f"Warning: No results found for {ml_model}, "
                          f"{metric}, {features}, {level}")
results.to_csv(fn.results_runs, index=False)
results_folds = results.explode(["fold", "mae_folds", "rmse_folds"], ignore_index=True)
results_folds.drop(columns=["mae", "rmse"], inplace=True)
results_folds.to_csv(fn.results_folds, index=False)
results_folds.groupby(["model", "fc", "features", "level"]).agg({
    "mae_folds": ["mean", "std", "median", "min", "max"],
    "rmse_folds": ["mean", "std", "median", "min", "max"],
}).sort_values(by=("mae_folds", "mean")).to_csv(fn.results_summary)
