"""Read-only sanity check on FOOOF contrasts: sign + magnitude of the aperiodic
(offset / exponent) and periodic-alpha deltas (real - silent).

Motivated by a discrepancy: the cycle-05_04 diary summary states "Δoffset y
Δexponent ambos broadly positivos -> el piso 1/f sube globalmente", which would
read as a uniform broadband floor lift (artifact-like). This script recomputes
the per-subject and grand-average means / sign-counts straight from
fooof_scr_contrasts.csv so we interpret the actual numbers, not the summary.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal._fooof_sign_check
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.erp_scr import OUT

CSV = OUT / "qa_artifact_vs_signal" / "tables" / "fooof_scr_contrasts.csv"


def _summ(df: pd.DataFrame, col: str) -> str:
    v = df[col].to_numpy(float)
    v = v[np.isfinite(v)]
    n_pos = int((v > 0).sum())
    n_neg = int((v < 0).sum())
    return (f"mean={v.mean():+.4f}  median={np.median(v):+.4f}  "
            f"sd={v.std():.4f}  (+:{n_pos} / -:{n_neg} of {v.size})")


def main() -> None:
    df = pd.read_csv(CSV)
    ch = df[df["level"] == "channel"]

    print("=" * 78)
    print(f"FOOOF contrast sign check  ::  {CSV.name}")
    print("delta = real(SCR) - silent.  exponent<0 => spectrum FLATTENS in SCR")
    print("=" * 78)

    print("\n--- GRAND AVERAGE across 32 channels (subject==GA) ---")
    ga = ch[ch["subject"] == "GA"]
    for col in ("d_offset", "d_exponent", "d_periodic_alpha"):
        print(f"  {col:18s}: {_summ(ga, col)}")

    print("\n--- PER SUBJECT (mean across 32 channels) ---")
    print(f"  {'subject':10s} {'d_offset':>12s} {'d_exponent':>12s} "
          f"{'d_per_alpha':>12s}")
    for sub in sorted(s for s in ch['subject'].unique() if s != "GA"):
        s = ch[ch["subject"] == sub]
        print(f"  {sub:10s} {s['d_offset'].mean():+12.4f} "
              f"{s['d_exponent'].mean():+12.4f} {s['d_periodic_alpha'].mean():+12.4f}")

    print("\n--- PER-SUBJECT sign agreement on d_exponent (mean<0 ?) ---")
    flat = [sub for sub in sorted(s for s in ch['subject'].unique() if s != "GA")
            if ch[ch['subject'] == sub]['d_exponent'].mean() < 0]
    print(f"  subjects with mean d_exponent < 0 (flattening): {len(flat)}/6  -> {flat}")

    print("\n--- ROI-level GA (subject==GA, level==roi) ---")
    roi = df[(df["level"] == "roi") & (df["subject"] == "GA")]
    print(roi[["key", "d_offset", "d_exponent", "d_periodic_alpha"]].to_string(index=False))


if __name__ == "__main__":
    main()
