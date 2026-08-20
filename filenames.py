"""Manage subject-specific filepaths for MEG connectivity analysis."""

from pathlib import Path


class FileNames(object):
    """Convenience class for managing file paths"""
    def __init__(self, analysis_root):
        """Initialize the FileNames object with the analysis root directory.

        Parameters
        ----------
        analysis_root : Path
            The root directory for the analysis outputs.
        """
        self.analysis_root = analysis_root

    @property
    def data_dir(self):
        data_dir = self.analysis_root / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @property
    def dl_dir(self):
        dl_dir = self.analysis_root / 'dl_models'
        dl_dir.mkdir(parents=True, exist_ok=True)
        return dl_dir

    @property
    def ml_dir(self):
        ml_dir = self.analysis_root / 'ml_models'
        ml_dir.mkdir(parents=True, exist_ok=True)
        return ml_dir

    @property
    def tfr_dir(self):
        tfr_dir = self.dl_dir / 'tfr'
        tfr_dir.mkdir(parents=True, exist_ok=True)
        return tfr_dir

    @property
    def included_subjects(self):
        included_subjects = self.data_dir / "included_subjects.csv"
        return included_subjects

    @property
    def ages_all_subjects(self):
        ages_file = self.data_dir / "ages_all_subjects.npy"
        return ages_file

    def fc_data(self, lower_triangle, upper_triangle,
                minmax=True, sensor_space=False, fband=None):
        scale_str = "_min_max_scaled" if minmax else ""
        if fband is not None:
            scale_str += f"_fband_{fband}"
        space_str = "sensor_space_" if sensor_space else "source_space_"
        fname = self.data_dir / (f"{space_str}fc_data_all_subjects_"
                                 f"{lower_triangle}"
                                 f"_{upper_triangle}{scale_str}.npy")
        return fname

    @property
    def hyperparam_dir(self):
        hyperparam_dir = self.dl_dir / 'hyperparameters'
        hyperparam_dir.mkdir(parents=True, exist_ok=True)
        return hyperparam_dir

    def hyperparam_file(self, model_name):
        hyperparam_file = self.hyperparam_dir / (f"{model_name}_"
                                                 f"hyperparameters.json")
        return hyperparam_file

    def fc_flat(self, metric, fband, sensor_space=False):
        space_str = "sensor_space_" if sensor_space else "source_space_"
        fname = self.data_dir / (f"{space_str}fc_data_all_subjects_"
                                 f"{metric}_fband_{fband}_flat.npy")
        return fname

    def graph_features(self, metric, sensor_space=False):
        space_str = "sensor_space_" if sensor_space else "source_space_"
        fname = self.ml_dir / (f"{space_str}graph_features_all_subjects_"
                               f"{metric}.csv")
        return fname

    def con_features(self, metric, sensor_space=False):
        space_str = "sensor_space_" if sensor_space else "source_space_"
        fname = self.ml_dir / (f"{space_str}con_features_all_subjects_"
                               f"{metric}.csv")
        return fname

    def ml_results(self, model_name, metric, sensor_space=False,
                   use_graph_features=False):
        space_str = "sensor_space_" if sensor_space else "source_space_"
        graph_str = "graph_" if use_graph_features else ""
        fname = self.ml_dir / (f"{space_str}{metric}_{graph_str}{model_name}_"
                               f"results.csv")
        return fname

    @property
    def results_summary(self):
        summary_file = self.analysis_root / "results_summary.csv"
        return summary_file

    @property
    def results_runs(self):
        runs_file = self.analysis_root / "results_runs.csv"
        return runs_file

    @property
    def results_folds(self):
        folds_file = self.analysis_root / "results_folds.csv"
        return folds_file

    @property
    def figure_dir(self):
        figure_dir = self.analysis_root / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        return figure_dir

    @property
    def dummy_model_results(self):
        dummy_results_file = self.analysis_root / "dummy_model_results.csv"
        return dummy_results_file


class SubFileNames(object):
    """Convenience class to manage subject-specific file paths for MEG
    connectivity analysis."""
    def __init__(self, source_data_dir, analysis_root, subject):
        """Initialize the SubFileNames object with directory paths and
        subject ID.

        Parameters
        ----------
        source_data_dir : Path
            The source directory where the MEG connectivity data is stored.
        analysis_root : Path
            The root directory for the analysis outputs.
        subject : str
            The subject ID for which to manage file paths.
        """
        self.source_data_dir = source_data_dir
        self.analysis_root = analysis_root
        self.subject = subject

    @property
    def connectivity_dir(self):
        con_dir = self.source_data_dir / self.subject / "connectivity"
        return con_dir

    @property
    def parcel_timeseries(self):
        ts_file = (self.connectivity_dir /
                   f"{self.subject}_task-rest_proc-tsss_meg_mean_flip_aparc_sub"
                   f"_nimeg_label_ts.npy")
        if ts_file.exists():
            return ts_file
        else:
            raise FileNotFoundError(f"Parcel timeseries file not found for "
                                    f"subject {self.subject} at {ts_file}")

    @property
    def fc_data_dir(self):
        fc_data_dir = self.analysis_root / "fc_data" / self.subject
        fc_data_dir.mkdir(parents=True, exist_ok=True)
        return fc_data_dir

    @property
    def aec_dir(self):
        aec_dir = self.fc_data_dir / "aec"
        aec_dir.mkdir(parents=False, exist_ok=True)
        return aec_dir
    
    @property
    def wpli_dir(self):
        wpli_dir = self.fc_data_dir / "wpli"
        wpli_dir.mkdir(parents=False, exist_ok=True)
        return wpli_dir

    @property
    def source_aec(self):
        aec_file_path = (
            self.connectivity_dir /
            f'{self.subject}_task-rest_proc-tsss_meg_mean_flip_connectivity-'
            f'matrix_aec_cwt_morlet_n-cycles-5_broad_1_000'
            f'-100_000Hz_aparc_nimeg_averaged.npy')
        if aec_file_path.exists():
            return aec_file_path
        else:
            raise FileNotFoundError(f"AEC file not found for subject "
                                    f"{self.subject} at "f"{aec_file_path}")

    @property
    def source_wpli(self):
        wpli_file_path = (
            self.connectivity_dir /
            f'{self.subject}_task-rest_proc-tsss_meg_mean_flip_connectivity-'
            f'matrix_wpli_cwt_morlet_n-cycles-5_broad_1_000'
            f'-100_000Hz_aparc_nimeg_averaged.npy')
        if wpli_file_path.exists():
            return wpli_file_path
        else:
            raise FileNotFoundError(f"WPLI file not found for subject "
                                    f"{self.subject} at "f"{wpli_file_path}")

    @property
    def psd_file(self):
        psd_file_path = (self.source_data_dir / self.subject / "psd" /
                         f'{self.subject}_task-rest_proc-tsss_meg-psd.npy')
        if psd_file_path.exists():
            return psd_file_path
        else:
            return None
    
    @property
    def epochs(self):
        epoch_file = (self.source_data_dir / self.subject / "epochs" /
                      f'{self.subject}_task-rest_proc-tsss_meg-epo.fif')
        if epoch_file.exists():
            return epoch_file
        else:
            return None

    @property
    def sensor_wpli_dir(self):
        wpli_dir = self.fc_data_dir / "sensor_wpli"
        wpli_dir.mkdir(parents=False, exist_ok=True)
        return wpli_dir
    
    @property
    def sensor_wpli(self):
        sensor_wpli_path = (self.sensor_wpli_dir /
                            (f'{self.subject}_task-rest_connectivity-matrix_'
                             f'wpli_morlet_n-cycles-5_broad_1_000-100_000Hz_'
                             f'sensor_space_204_gradiometers.npy'))
        return sensor_wpli_path

    @property
    def sensor_aec_dir(self):
        aec_dir = self.fc_data_dir / "sensor_aec"
        aec_dir.mkdir(parents=False, exist_ok=True)
        return aec_dir
        
    @property
    def sensor_aec(self):
        sensor_aec_path = (self.sensor_aec_dir /
                           f'{self.subject}_task-rest_connectivity-matrix_aec_'
                           f'morlet_n-cycles-5_broad_1_000-100_000Hz_sensor_'
                           f'space_204_gradiometers.npy')
        return sensor_aec_path

    @property
    def sensor_psd_dir(self):
        sensor_psd = (self.analysis_root / "fc_data" / self.subject /
                      "sensor_psd")
        sensor_psd.mkdir(parents=False, exist_ok=True)
        return sensor_psd
        
    @property
    def sensor_psd(self):
        sensor_psd = self.sensor_psd_dir / (f'{self.subject}_task-rest_psd_'
                                            f'morlet_n-cycles-5_1-100Hz_sensor'
                                            f'_space_204_gradiometers.npy')
        return sensor_psd
        
    def frequency_banded_aec(self, band):
        aec_fband_path = self.aec_dir / (f'{self.subject}_aec_connectivity_'
                                         f'{band}_Hz_90_90_resolution.npy')
        return aec_fband_path
        
    def frequency_banded_wpli(self, band):
        wpli_fband_path = self.wpli_dir / (f'{self.subject}_wpli_connectivity_'
                                           f'{band}_Hz_90_90_resolution.npy')
        return wpli_fband_path
