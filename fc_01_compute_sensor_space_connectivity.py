"""Compute sensor space connectivity for one subject."""

import argparse

import mne
from mne.time_frequency import tfr_array_morlet
import mne_connectivity
import numpy as np

from config import analysis_root, source_data_dir, freqs
from filenames import SubFileNames

parser = argparse.ArgumentParser(
    description="Compute sensor space connectivity for one subject.")
parser.add_argument("subject", type=str,
                    help="Subject ID (e.g. sub-CC110033)")
args = parser.parse_args()
subject = args.subject

fn_sub = SubFileNames(source_data_dir, analysis_root, subject)
epochs = mne.read_epochs(fn_sub.epochs)
epochs = epochs.pick("grad")

# Compute WPLI
print("Computing WPLI...")
wpli = mne_connectivity.spectral_connectivity_time(
    epochs,
    method='wpli',
    average=True,
    mode='cwt_morlet',
    sfreq=epochs.info["sfreq"],
    decim=4,
    sm_times=0,
    padding=0.,
    faverage=False,
    n_jobs=4,
    freqs=freqs,
    n_cycles=5
)
wpli = wpli.get_data(output='dense')
np.save(fn_sub.sensor_wpli, wpli)

# Compute AEC
print("Computing AEC...")
sfreq = epochs.info["sfreq"]
epochs = epochs.get_data()
for i, freq in enumerate(freqs):
    print("analysing frequency: ", freq)
    tfr = tfr_array_morlet(
        epochs,
        sfreq,
        [freq],
        decim=4,
        n_jobs=4,
        output='complex',
        n_cycles=5,
        use_fft=True
    )
    tfr = np.squeeze(tfr)
    aec_f = mne_connectivity.envelope_correlation(
        tfr,
        orthogonalize="pairwise")
    aec_f = aec_f.combine()
    aec_f = aec_f.get_data(output="dense").squeeze()
    aec_f = np.tril(aec_f, k=-1)
    if i == 0:
        aec = aec_f[..., np.newaxis]
    else:
        aec = np.dstack([aec, aec_f])
np.save(fn_sub.sensor_aec, aec)
