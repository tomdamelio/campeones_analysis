"""
Shared configuration for modeling scripts.

EEG channel list verified from BIDS channels.tsv:
data/derivatives/campeones_preproc/sub-27/ses-vr/eeg/
sub-27_ses-vr_task-01_acq-a_run-002_desc-preproc_channels.tsv
"""

# 32 EEG channels — excludes non-EEG sensors:
# ECG, R_EYE, L_EYE, AUDIO, PHOTO, GSR, RESP, X, Y, Z,
# triggerStream, joystick_x, joystick_y
EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6',
    'FT9', 'FT10', 'TP9', 'TP10', 'FCz'
]
