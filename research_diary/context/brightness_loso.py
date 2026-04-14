"""Binary brightness decoding — Best practice LOSO pipeline.

Best config from Phase 4:
  - Time window: 0.06 – 1.26 s (1.2 s duration)
  - Classifier: LinearSVC (C=1.0)
  - CV: Leave-One-Subject-Out (LOSO)
"""

import os
import sys
import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = '/scratch/yongjieduan/eeg_moments/dataset/preprocessed_data/eeg_zscored'
OUTPUT_DIR = '/scratch/yongjieduan/eeg_moments/binary_brightness_decoding_results'
SFREQ = 100
SUBJECTS = [1, 2, 3, 4, 5, 6]

# Best parameters
WIN_START = 0.06   # seconds
WIN_END   = 1.26   # seconds (1.2 s window)
C_VALUE   = 1.0    # LinearSVC regularization

t_total = time.time()

# =============================================================================
# Load brightness labels
# =============================================================================
brightness_cache = os.path.join(OUTPUT_DIR, 'brightness_cache.npy')
brightness = np.load(brightness_cache, allow_pickle=True).item()
sorted_vids = sorted(brightness.keys(), key=lambda v: brightness[v])
vid_bright = sorted_vids[-1]
vid_dark   = sorted_vids[0]
print(f'Brightest: {vid_bright} (L={brightness[vid_bright]:.1f})')
print(f'Darkest:   {vid_dark}   (L={brightness[vid_dark]:.1f})')

# =============================================================================
# Load EEG data
# =============================================================================
print(f'\nLoading {len(SUBJECTS)} subjects...')
eeg_list, vid_list, sub_list = [], [], []
times = None

for sub in SUBJECTS:
    t0 = time.time()
    path = os.path.join(DATA_DIR,
        f'sub-{sub:02d}', 'mvnn-time', 'baseline_correction-01',
        'highpass-0.01_lowpass-100', f'sfreq-{SFREQ:04d}',
        'preprocessed_data.npy')
    d = np.load(path, allow_pickle=True).item()
    if times is None:
        times = d['times']
    eeg  = np.concatenate(d['eeg_data'], axis=0)
    stim = np.concatenate(d['stimuli_presentation_order'], axis=0)
    del d

    # Keep only brightest & darkest video trials
    mask = np.isin(stim, [vid_bright, vid_dark])
    eeg_list.append(eeg[mask])
    vid_list.append(stim[mask])
    sub_list.append(np.full(mask.sum(), sub))
    print(f'  Sub-{sub:02d}: {mask.sum()} trials ({time.time()-t0:.1f}s)')
    del eeg, stim

X_all = np.concatenate(eeg_list, axis=0)   # (trials, channels, timepoints)
y     = np.where(np.isin(np.concatenate(vid_list), [vid_bright]), 1, 0)
groups = np.concatenate(sub_list)
del eeg_list, vid_list, sub_list
print(f'  Pooled: {X_all.shape}, bright={y.sum()}, dark={len(y)-y.sum()}')

# =============================================================================
# Extract time window & flatten features
# =============================================================================
t_mask = (times >= WIN_START) & (times <= WIN_END)
n_tp = t_mask.sum()
X = X_all[:, :, t_mask].reshape(len(X_all), -1)
del X_all
print(f'\nTime window: {WIN_START:.2f} – {WIN_END:.2f} s  '
      f'({n_tp} timepoints, {X.shape[1]} features)')

# =============================================================================
# LOSO cross-validation
# =============================================================================
print('\nRunning LOSO cross-validation...')
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LinearSVC(C=C_VALUE, max_iter=10000, dual='auto')),
])
logo = LeaveOneGroupOut()
scores = cross_val_score(pipe, X, y, cv=logo, groups=groups, scoring='accuracy')

# =============================================================================
# Results
# =============================================================================
print(f'\n{"="*50}')
print('LOSO Results')
print(f'{"="*50}')
print(f'  Mean accuracy: {scores.mean()*100:.1f}%')
print(f'  Std:           {scores.std()*100:.1f}%')
print(f'  Per-subject:')
for i, (sub, acc) in enumerate(zip(SUBJECTS, scores)):
    print(f'    Sub-{sub:02d}: {acc*100:.1f}%')

# =============================================================================
# Save
# =============================================================================
results = {
    'config': {
        'win_start': WIN_START,
        'win_end': WIN_END,
        'C': C_VALUE,
        'classifier': 'LinearSVC',
        'cv': 'LOSO',
    },
    'loso_mean': scores.mean(),
    'loso_std': scores.std(),
    'loso_per_subject': dict(zip(SUBJECTS, scores)),
    'loso_folds': scores,
}
save_path = os.path.join(OUTPUT_DIR, 'best_practice_loso.npy')
np.save(save_path, results)
print(f'\nSaved: {save_path}')
print(f'Total time: {time.time()-t_total:.1f}s')
print('Done.')
