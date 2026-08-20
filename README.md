# Age prediction from functional connectivity using deep learning

This repository contains the code for the paper "Interpretable Decoding of Frequency-Resolved
Functional Connectivity" (Saarro, Ruuskanen, Caivano, Parkkonen, 
Zubarev, in prep.).

## Prerequisites
Install the required Python packages.
- ```conda env create -f env/dlfc_env.yml```

To obtain the latest `mneflow` version, navigate to
https://github.com/zubara/mneflow.git, clone the repository, and install from
the `dev` branch.

## Dataset
This study uses the Cam-CAN dataset (Taylor et al., NeuroImage 2017), which is 
publicly available upon request from the Cambridge Centre for Ageing and 
Neuroscience (Cam-CAN) website: https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/.

## Configuration
Before running any of the code, specify the dataset path in `config.py`.

## Source-space functional connectivity estimation
The source space functional connectivity estimation and preprocessing pipeline 
is implemented separately (Ruuskanen et al., Human Brain Mapping 2026). The code 
for this part is available upon request from the authors.

## Sensor-level functional connectivity estimation and post-processing
Assuming preprocessed data in appropriate format and location,
- ```python fc_01_compute_sensor_space_connectivity.py <subject_id>```.

To run on all subjects in parallel, use the batch script (modify for your cluster and data paths)
- ```sbatch sh/compute_sensor_space_fc.sh```.

Aggregate all connectivity files across subjects and average in frequency bands for the ML models.
- ```python fc_02_aggregate_connectivity_data.py```
- ```python fc_03_average_connectivity_fbands.py```.

## Age prediction using deep learning
- ```python dl_01_create_hyperparameter_json.py```
- ```python dl_02_create_tfrecord_file.py```
- ```python dl_03_train_dl_models.py <model_name> <dataset_idx> <run_idx>```

The relevant file paths are defined in `filenames.py`. The deep learning models
are defined in `mneflow`. Look up `sh/train_dl_models.sh` for example usage. 
Use `--help` for more information.

## Age prediction using machine learning
- ```python ml_01_make_features.py```
- ```python ml_02_train_models.py model_name metric```

Use `python ml_02_train_models.py --help` for information on the arguments.

## Run the dummy model
```python dummy_model.py```

## Evaluation of the results
- ```python collect_results.py```

Plot each figure by running the individual scripts.
