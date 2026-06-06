"""¿La firma de acoplamiento EEG-banda<->SMNA es específica de la EDA, o es AROUSAL AUTONÓMICO GENERAL?

Test decisivo que el propio INFORME propone (§12 item 3): comparar la firma de acoplamiento de la
EEG-banda con DISTINTAS señales periféricas (SMNA, EDA-phasic, HR, RVT-respiración). Si HR y RVT
muestran la MISMA firma (gamma>delta, PO≈edge) que la SMNA -> es arousal autonómico/EMG general,
la EDA NO es privilegiada (refuerza el veredicto). Si HR/RVT difieren (p.ej. focal o nulo) -> la
relación con la EDA tiene algo propio (reabre).

Dominio CONTINUO (reusa continuous_band.build_subject_continuous: envelopes Hilbert log por banda a
12.5 Hz, alineados por run). HR/RVT continuos se interpolan sobre el mismo timebase (`times`) y se
z-scorean por run. Spearman por sujeto (runs concatenados, z intra-run), agregado cross-subject.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.coupling_periph_multitarget
  ... --subjects sub-23   (smoke)
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, NPZ_DIR, OUT
from src.campeones_analysis.multimodal_arousal.continuous_band import (
    BANDS,
    build_subject_continuous,
)

warnings.filterwarnings("ignore")

EDGE = ["FT9", "TP9", "T7", "T8", "P7", "P8"]   # proxy EMG (bordes temporales)
TARGETS = ["smna", "phasic", "hr", "rvt"]
USE_BANDS = list(BANDS)  # delta, alpha, gamma


def _z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def subject_series(sub):
    """Devuelve por (target, band, roi) las series concatenadas (z intra-run) para un sujeto."""
    runs = build_subject_continuous(sub)
    if not runs:
        return None
    cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
    acc = {}  # (target,band,roi) -> list of arrays
    for label, r in runs.items():
        times, ch = r["times"], r["ch"]
        edge_idx = [ch.index(c) for c in EDGE if c in ch]
        # targets en el timebase `times`
        tgt = {"smna": r["eda"], "phasic": r["eda_phasic"]}
        try:
            hr_t = np.asarray(cont[f"{label}__hr_t"], float)
            hr = np.asarray(cont[f"{label}__hr_continuous"], float)
            tgt["hr"] = _z(np.interp(times, hr_t, hr))
        except Exception:
            pass
        try:
            rsp_t = np.asarray(cont[f"{label}__rsp_t"], float)
            rvt = np.asarray(cont[f"{label}__rvt_continuous"], float)
            tgt["rvt"] = _z(np.interp(times, rsp_t, rvt))
        except Exception:
            pass
        for b in USE_BANDS:
            po = r["bands"][b]["po"]
            allch = r["bands"][b]["all"]
            edge = allch[:, edge_idx].mean(axis=1) if edge_idx else np.full(len(po), np.nan)
            for roi, sig in (("PO", po), ("edge", edge)):
                for tname, tv in tgt.items():
                    if len(tv) != len(sig):
                        continue
                    acc.setdefault((tname, b, roi), []).append((_z(sig), _z(tv)))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=None)
    args = ap.parse_args()
    subs = args.subjects if args.subjects else list(COHORT)
    print("=" * 78)
    print(f"coupling_periph_multitarget :: targets={TARGETS} bandas={USE_BANDS}  subjects={subs}")
    print("=" * 78, flush=True)

    per_sub_rows = []
    for sub in subs:
        print(f"\n=== {sub} ===", flush=True)
        acc = subject_series(sub)
        if acc is None:
            print(f"  {sub}: no data", flush=True)
            continue
        for (tname, b, roi), pairs in acc.items():
            sig = np.concatenate([p[0] for p in pairs])
            tv = np.concatenate([p[1] for p in pairs])
            m = np.isfinite(sig) & np.isfinite(tv)
            if m.sum() < 50:
                continue
            rho = spearmanr(sig[m], tv[m])[0]
            per_sub_rows.append(dict(subject=sub, target=tname, band=b, roi=roi, rho=float(rho)))

    df = pd.DataFrame(per_sub_rows)
    sdir = OUT / "eeg_smna_coupling" / "tables"
    sdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(sdir / "periph_multitarget_perSubject.csv", index=False)

    # resumen: target x band x roi -> mean rho, n_pos
    summ = {}
    for tname in TARGETS:
        summ[tname] = {}
        for b in USE_BANDS:
            summ[tname][b] = {}
            for roi in ("PO", "edge"):
                s = df[(df.target == tname) & (df.band == b) & (df.roi == roi)]
                if len(s):
                    summ[tname][b][roi] = dict(mean_rho=round(float(s["rho"].mean()), 4),
                                               n_pos=int((s["rho"] > 0).sum()), n=len(s))
    with open(sdir / "periph_multitarget_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=2)
    print("\n=== RESUMEN (mean rho PO) por target x banda ===")
    for tname in TARGETS:
        line = " | ".join(
            f"{b}: {summ.get(tname, {}).get(b, {}).get('PO', {}).get('mean_rho', float('nan')):+.3f}"
            f"(edge {summ.get(tname, {}).get(b, {}).get('edge', {}).get('mean_rho', float('nan')):+.3f})"
            for b in USE_BANDS)
        print(f"  {tname:7s}: {line}", flush=True)
    print(f"\nOutputs -> {sdir}", flush=True)


if __name__ == "__main__":
    main()
