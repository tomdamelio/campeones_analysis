"""Sanity check (GATE E follow-up): ¿cambia el decoding de 1.5 al sacar FC1? Re-deriva los AUCs
de los 4 feature-sets (1-40, sin perms) con FC1 incluido vs excluido del cache. NO interpola
(eso es el fix real); solo cuantifica si FC1 mueve las conclusiones."""
from __future__ import annotations

import warnings
import numpy as np

from src.campeones_analysis.multimodal_arousal import decoding_panel as d
from src.campeones_analysis.multimodal_arousal.decoding_sweep import RANGES

warnings.filterwarnings("ignore")


def run(drop_fc1):
    data, freqs, ch = d.load_cache("uniform")
    ch = list(ch)
    if drop_fc1:
        keep = [i for i, c in enumerate(ch) if c != "FC1"]
        data = {s: (data[s][0][:, keep, :], data[s][1], data[s][2]) for s in data}
    out = {}
    for fset in d.FEATURE_SETS:
        r = d.eval_fset(fset, data, freqs, "1-40", 0)
        out[fset] = (r["intra_auc"], r["loso_auc"])
    return out


def main():
    print("Impacto de FC1 en el decoding de 1.5 (1-40, sin perms):")
    full = run(False)
    drop = run(True)
    print(f"\n  {'feature-set':24s} {'intra full':>10s} {'intra -FC1':>11s} {'LOSO full':>10s} {'LOSO -FC1':>10s}")
    for fset in d.FEATURE_SETS:
        i0, l0 = full[fset]; i1, l1 = drop[fset]
        print(f"  {fset:24s} {i0:10.3f} {i1:11.3f} {l0:10.3f} {l1:10.3f}   "
              f"Δintra={i1-i0:+.3f} ΔLOSO={l1-l0:+.3f}")
    print("\n  (Δ pequeño = las conclusiones de 1.5 NO dependen de FC1; el fix es por limpieza/Haufe.)")


if __name__ == "__main__":
    main()
