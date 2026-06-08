"""Diagnostico ad-hoc (GATE E follow-up): topografia del Haufe aperiodico (offset/exponent)
canal-por-canal, pooled y por-sujeto, para chequear si el offset decodable es global o se
concentra en temporal-izquierdo/edge (EMG) o frontal (ocular). NO es parte del pipeline.

Run: micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal._haufe_detail
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from src.campeones_analysis.multimodal_arousal import decoding_panel as d
from src.campeones_analysis.multimodal_arousal.decoding_sweep import RANGES, feat_aperiodic
from src.campeones_analysis.multimodal_arousal.cohort import COHORT, SUBJ_COLORS

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

EDGE = ["F7", "F8", "T7", "T8", "P7", "P8", "FT9", "FT10", "TP9", "TP10", "FC5", "FC6", "CP5", "CP6"]
LEFT_TEMP = ["F7", "T7", "P7", "FT9", "TP9", "FC5", "CP5"]
FRONTAL = ["Fp1", "Fp2", "F7", "F8", "Fz"]
CENTRAL = ["Cz", "Pz", "C3", "C4", "CP1", "CP2", "FC1", "FC2", "FCz", "P3", "P4"]


def _haufe_pooled(data, freqs, nch):
    A = d.haufe_pattern("aperiódico", data, freqs, "1-40")
    return A[:nch], A[nch:2 * nch]


def _haufe_one_subject(psd, y, freqs, nch):
    """Within-subject Haufe of the aperiodic feature-set."""
    X = feat_aperiodic(psd, freqs, RANGES["1-40"])
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    lr = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="lbfgs").fit(Xs, y)
    A = np.cov(Xs, rowvar=False) @ lr.coef_[0]
    return A[:nch], A[nch:2 * nch]


def _report(name, vals, ch):
    nch = len(ch)
    order = np.argsort(-np.abs(vals))
    tot = float(np.sum(np.abs(vals)))
    print(f"\n--- {name}: top-8 |A| ---")
    for i in order[:8]:
        print(f"  {ch[i]:5s} {vals[i]:+.3f}  ({abs(vals[i]) / tot * 100:4.1f}% of total |A|)")
    for label, group in [("edge(14)", EDGE), ("left-temp(7)", LEFT_TEMP),
                         ("frontal(5)", FRONTAL), ("central(11)", CENTRAL)]:
        idx = [ch.index(c) for c in group if c in ch]
        share = np.sum(np.abs(vals[idx])) / tot * 100
        unif = len(idx) / nch * 100
        print(f"  {label:14s} share={share:4.0f}%  (uniforme daría {unif:.0f}%)  "
              f"{'<-- CONCENTRADO' if share > 1.5 * unif else ''}")


def main():
    data, freqs, ch = d.load_cache("uniform")
    ch = list(ch); nch = len(ch)
    print(f"channels ({nch}): {ch}")

    off_p, exp_p = _haufe_pooled(data, freqs, nch)
    _report("OFFSET pooled", off_p, ch)
    _report("EXPONENT pooled", exp_p, ch)

    # per-subject consistency: is the pooled pattern driven by 1-2 subjects?
    print("\n==== per-subject OFFSET Haufe (consistency check) ====")
    offs = {}
    for s in COHORT:
        if s not in data:
            continue
        psd, y, _ = data[s]
        o, e = _haufe_one_subject(psd, y, freqs, nch)
        offs[s] = o
        order = np.argsort(-np.abs(o))
        top = ", ".join(f"{ch[i]}{o[i]:+.2f}" for i in order[:5])
        # correlation of this subject's offset-pattern with the pooled
        r = np.corrcoef(o, off_p)[0, 1]
        print(f"  {s}: top5 = {top}   | corr con pooled = {r:+.2f}")

    # mean cross-subject correlation of offset patterns
    subs = list(offs)
    rs = [np.corrcoef(offs[a], offs[b])[0, 1] for i, a in enumerate(subs) for b in subs[i + 1:]]
    print(f"\n  corr media inter-sujeto de la topografia OFFSET = {np.mean(rs):+.2f} "
          f"(rango {min(rs):+.2f}..{max(rs):+.2f})")
    print("  (alto = topografia consistente; bajo = cada sujeto carga canales distintos = no es una fuente comun)")

    # figure: per-subject offset topomaps
    info = d._make_info(ch)
    fig, axes = plt.subplots(1, len(subs) + 1, figsize=(2.3 * (len(subs) + 1), 3.0))
    d._topo(axes[0], off_p, info, "POOLED")
    for j, s in enumerate(subs):
        d._topo(axes[j + 1], offs[s], info, s)
    fig.suptitle("Haufe OFFSET (aperiódico) — pooled vs por-sujeto. ¿Consistente o 1-2 sujetos?",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = d.FIG_DIR / "haufe_offset_persubject.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
