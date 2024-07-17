# Execution code for TRF analysis
# uses the methods developed in Reichenbach lab
# To use this code you need to install spyeeg: https://github.com/phg17/sPyEEG

import numpy as np
from os.path import join
import mne
from scipy.stats import zscore
import scipy.io as sio
import spyeeg

# Parameters for data loading and saving
data_dir = '...'  # path to MEG data
meg_file = 'meg_filename.fif'  # preprocessed MEG filename (.fif file)
audio_dir = '...'  # path to audios

sr = 1000  # sampling rate
alpha = [1e-5, 1e-3, 1e-1, 0, 1, 10, 100, 10000000]  # regularization parameters

save_dir = '...'  # path where results should be saved

# Load MEG data and topology information
meg_fname = join(data_dir, meg_file)
meg = mne.io.read_raw_fif(meg_fname, preload=True, verbose=False)
info = meg.info

# Get MEG data and apply filtering
meg_data = meg.get_data().T
meg_data = zscore(meg_data)

# Load audio data
fs_audio, audio = sio.wavfile.read(audio_dir)

# Estimate fundamental wavefrom from audio
fw = spyeeg.feat.signal_f0wav(audio, srate=fs_audio, resample=sr)

# Create features matrix
xtrf = np.hstack([fw])
ytrf = meg_data[:]

# Fit TRF
trf = spyeeg.models.TRF.TRFEstimator(tmin=-0.025, tmax=0.145, srate=sr, alpha=alpha)
trf.fit_from_cov(xtrf, ytrf, part_length=60, clear_after=False)

# Uncomment the following lines for cross-validation
# scores = trf.xval_eval(xtrf, ytrf, n_splits=5, lagged=False, drop=True, train_full=True, scoring="corr", segment_length=None, fit_mode='from_cov', verbose=True)

# Extract coefficients
coef_fw = trf.get_coef()[:, 0, :, :].T

# Save TRF coefficients
np.save(join(save_dir, 'coef_fw.npy'), coef_fw)

# Plot TRFs using MNE
ev1 = mne.EvokedArray(coef_fw, info, tmin=trf.tmin)
ev1.plot_joint()

