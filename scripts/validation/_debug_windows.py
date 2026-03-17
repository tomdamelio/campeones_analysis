"""Debug window sample counts."""
import numpy as np

sfreq = 250.0
tmin = -2.5

POST_WINDOWS = [(0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25)]
PRE_WINDOWS = [(-0.25, -0.20), (-0.20, -0.15), (-0.15, -0.10), (-0.10, -0.05)]

for label, windows in [("POST", POST_WINDOWS), ("PRE", PRE_WINDOWS)]:
    for t_start, t_end in windows:
        win_samples = int(np.round((t_end - t_start) * sfreq))
        s0 = int(np.round((t_start - tmin) * sfreq))
        s1 = s0 + win_samples
        print(f"  {label} [{t_start:.3f}, {t_end:.3f}]: "
              f"win_samples={win_samples}, s0={s0}, s1={s1}, "
              f"n_feat_raw={32 * win_samples}")
