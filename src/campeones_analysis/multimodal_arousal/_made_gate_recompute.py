"""Recompute the MADE/PREP channel gate with a per-run aggregation (read-only).

epochs_qc.py used the UNION of bad_channels across runs -> overcounts (a channel bad in 1 of
~7 runs counts the same as one bad throughout), flagging all 6 subjects. The 'globally bad'
notion (MADE/PREP) means channels consistently bad across the recording. Here we report, per
subject: n_runs, median & max bad channels per run, and the set bad in >=50% of runs
(consistent_bad). Apply MADE (>10% of 32 = >=4) and PREP (>25% = >=8) to the consistent count.
"""

from __future__ import annotations

import json

from src.campeones_analysis.multimodal_arousal.cohort import COHORT
from src.campeones_analysis.multimodal_arousal.erp_scr import PREP

import numpy as np

LOG = PREP / "logs_preprocessing_details_all_subjects_eeg.json"
N_CHAN = 32


def main() -> None:
    log = json.load(open(LOG, encoding="utf-8"))
    print(f"{'sub':>7} {'runs':>5} {'med/run':>8} {'max/run':>8} {'consistent(>=50%)':>18} "
          f"{'MADE_excl':>10} {'PREP_flag':>10}")
    for sub in COHORT:
        num = sub.replace("sub-", "")
        entry = log.get(num, {})
        per_run_counts = []
        ch_run_count: dict[str, int] = {}
        n_runs = 0
        for ses, runs in entry.items():
            if not isinstance(runs, dict):
                continue
            for run, info in runs.items():
                if not isinstance(info, dict) or "bad_channels" not in info:
                    continue
                n_runs += 1
                bads = info.get("bad_channels", []) or []
                per_run_counts.append(len(bads))
                for c in bads:
                    ch_run_count[c] = ch_run_count.get(c, 0) + 1
        if n_runs == 0:
            print(f"{sub:>7}  (no runs in log)")
            continue
        med = float(np.median(per_run_counts))
        mx = int(np.max(per_run_counts))
        consistent = sorted(c for c, k in ch_run_count.items() if k >= n_runs / 2)
        nc = len(consistent)
        made = nc > 0.10 * N_CHAN  # >=4
        prep = nc > 0.25 * N_CHAN  # >=8
        print(f"{sub:>7} {n_runs:>5} {med:>8.1f} {mx:>8} {nc:>4} {str(consistent):>40} "
              f"{'YES' if made else 'no':>10} {'YES' if prep else 'no':>10}")


if __name__ == "__main__":
    main()
