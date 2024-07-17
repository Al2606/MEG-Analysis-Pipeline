# Democode for Source_Recon

from src import SourceRecon

# subject specifications
subjects = ['P10'] # List of subjects as strings
subjects_dir = '.../subjects/' # Path to subject folders
ROI = 'cortex'
meg_data_part = 'meg_audiobook1'

# MRI dirs
bem_model = '.../bem_model.fif'
bem = '.../bem.fif'
mri = '.../aparc+aseg.mgz'

# noise covariance dir
noisecov_path = '.../noise_cov.fif'

# save dir
save_path = '...'

# run source estimation for subject 'P10':
stc = SourceRecon(subjects[0], subjects_dir, bem_model, bem, mri, noisecov_path, save_path)

# !save is set to False for all functions
trans = stc.coregistration(save=False)
fwd = stc.leadfield(ROI, save=False)
stc_data = stc.beamformer(ROI, meg_data_part, save=False)