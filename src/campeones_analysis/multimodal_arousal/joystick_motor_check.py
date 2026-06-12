"""Bloque 4.x (opción A) — ¿qué porta la señal post-shift? Confirmación de que es MOVIMIENTO.

La ventana post-shift `[-0.5,3]` decodifica evento-vs-calma a AUC ~0.73 (arousal). Este script
caracteriza QUÉ la porta, para confirmar la lectura "motor + EMG, no afecto":

  1. edge-vs-central: decodificar con SOLO canales de borde (EMG_EDGE, firma muscular craneal)
     vs SOLO centrales (CENTRAL, corteza sensoriomotora). Si edge solo ya da AUC alto -> EMG.
  2. por-banda: ¿qué banda porta el decoding? gamma (30-40) = EMG; alfa/beta = ERD sensoriomotor.
  3. Haufe (periódico): topografía del decoder -> se espera central (mu/beta) + borde (EMG).

Lee el cache de `joystick_lag_sweep.py` (lag 4 = post-shift). Reusa intra/loso/feature-builders.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_motor_check --dim arousal
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.cohort import EMG_EDGE, CENTRAL
from src.campeones_analysis.multimodal_arousal.decoding_sweep import (
    RANGES, BANDS_40, intra_auc, loso_auc, feat_periodic,
)
from src.campeones_analysis.multimodal_arousal.matched_banda import feat_band
from src.campeones_analysis.multimodal_arousal.decoding_panel import (
    haufe_pattern, _make_info, _topo, PRIMARY_RANGE,
)
from src.campeones_analysis.multimodal_arousal.joystick_lag_sweep import (
    CACHE as LAG_CACHE, load_lag, LAGS, LAG_LABELS,
)
from src.campeones_analysis.multimodal_arousal.joystick_panel import OUT_DIR

warnings.filterwarnings("ignore")
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
POST_LAG = 4  # [-0.5, 3.0]
RNG = RANGES[PRIMARY_RANGE]


def _decode(data, freqs, builder):
    Xs = [builder(psd, freqs, RNG) for psd, y, tn in data.values()]
    ys = [y for psd, y, tn in data.values()]
    Xall = np.concatenate(Xs, 0); yall = np.concatenate(ys)
    groups = np.concatenate([[i] * len(y) for i, y in enumerate(ys)])
    intra, _ = intra_auc(Xs, ys)
    loso, _ = loso_auc(Xall, yall, groups)
    return round(intra, 3), round(loso, 3), Xall.shape[1]


def _restrict(data, idx):
    return {s: (psd[:, idx, :], y, tn) for s, (psd, y, tn) in data.items()}


def run(dim_tag):
    dim = None if dim_tag == "combined" else dim_tag
    z = np.load(LAG_CACHE, allow_pickle=True)
    data, freqs, ch = load_lag(z, POST_LAG, dim=dim, min_per_class=5)
    e_idx = [ch.index(c) for c in EMG_EDGE if c in ch]
    c_idx = [ch.index(c) for c in CENTRAL if c in ch]
    print(f"[A] {dim_tag} post-shift {LAG_LABELS[POST_LAG]}: {len(data)} suj, {len(ch)} ch")
    print(f"    EMG_EDGE({len(e_idx)})={[ch[i] for i in e_idx]}")
    print(f"    CENTRAL ({len(c_idx)})={[ch[i] for i in c_idx]}", flush=True)

    rows = []
    # 1. full / edge / central (periódico)
    for name, d in [("full(29)", data), (f"edge({len(e_idx)})", _restrict(data, e_idx)),
                    (f"central({len(c_idx)})", _restrict(data, c_idx))]:
        i, l, nf = _decode(d, freqs, feat_periodic)
        rows.append(dict(test="region", name=name, intra=i, loso=l, nfeat=nf))
        print(f"  periódico {name:12s}: intra={i:.3f}  LOSO={l:.3f}  nfeat={nf}", flush=True)
    # 2. por banda (periódico de UNA banda, todos los canales)
    for bn, band in BANDS_40.items():
        i, l, nf = _decode(data, freqs, lambda p, f, r, b=band: feat_band(p, f, r, b))
        rows.append(dict(test="band", name=bn, intra=i, loso=l, nfeat=nf))
        print(f"  banda     {bn:12s}: intra={i:.3f}  LOSO={l:.3f}  nfeat={nf}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / f"motor_check_{dim_tag}.csv", index=False)

    # 3. Haufe (periódico) topo
    info = _make_info(ch); nch = len(ch)
    A = haufe_pattern("periódico", data, freqs, PRIMARY_RANGE)
    bnames = list(BANDS_40.keys())
    fig, axes = plt.subplots(1, len(bnames), figsize=(2.4 * len(bnames), 3.4))
    for j, bn in enumerate(bnames):
        _topo(axes[j], A[j * nch:(j + 1) * nch], info, bn)
    fig.suptitle(f"A. Haufe periódico post-shift {LAG_LABELS[POST_LAG]} — {dim_tag} "
                 "(central=ERD motor; borde=EMG)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG_DIR / f"motor_check_haufe_{dim_tag}.png", dpi=120); plt.close(fig)

    # bar figure: region + band
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, test, ttl in zip(axes, ["region", "band"], ["región (periódico)", "banda (periódico)"]):
        sub = df[df.test == test]
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["intra"], 0.4, label="intra", color="C0", alpha=0.7)
        ax.bar(x + 0.2, sub["loso"], 0.4, label="LOSO", color="C1", alpha=0.7)
        ax.axhline(0.5, color="0.5", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(sub["name"], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0.3, 0.85); ax.set_ylabel("AUC"); ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle(f"A. ¿Qué porta la señal post-shift? — joystick {dim_tag}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / f"motor_check_{dim_tag}.png", dpi=120); plt.close(fig)
    print(f"\n-> {FIG_DIR}/motor_check_{dim_tag}.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", choices=["arousal", "valence", "combined"], default="arousal")
    args = ap.parse_args()
    run(args.dim)


if __name__ == "__main__":
    main()
