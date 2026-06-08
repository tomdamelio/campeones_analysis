"""Auditoría de calidad de canales de la cohorte (GATE E follow-up, pedido del usuario).
Decide qué canales son crónicamente malos (candidatos a interpolación global / exclusión) con
criterios del estado del arte, cruzando:
  (a) logs del preproc: % de runs donde PyPREP marcó cada canal como bad (por sujeto de la cohorte).
  (b) cache post-preproc (panel_psd.npz): anomalía de potencia (z) + correlación con vecinos, por
      canal y sujeto -> detecta canales que quedan raros incluso tras la interpolación per-run.
NO es parte del pipeline.

Run: micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal._channel_audit
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict

import mne
import numpy as np

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, REPO
from src.campeones_analysis.multimodal_arousal import decoding_panel as d

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

LOG = REPO / "data" / "derivatives" / "campeones_preproc" / "logs_preprocessing_details_all_subjects_eeg.json"

# SOTA-ish thresholds for "chronically bad" -> handle globally (interpolate-all / exclude)
PCT_FLAG_CHRONIC = 50.0   # flagged bad in >=50% of a subject's runs
Z_OUTLIER = 2.0           # |z| of mean log-power within subject


def _collect_log_bads():
    """Return (per_sub_runs, per_sub_bad). The log is keyed by SUBJECT NUMBER (e.g. '19'),
    then 'vr', then run records each with 'bad_channels'. We restrict to the cohort numbers."""
    data = json.loads(LOG.read_text(encoding="utf-8"))
    cohort_num = {s.split("-")[1]: s for s in COHORT}   # '19' -> 'sub-19'
    per_sub_runs = defaultdict(int)
    per_sub_bad = defaultdict(lambda: defaultdict(int))

    def walk(obj, sub):
        if isinstance(obj, dict):
            if "bad_channels" in obj and isinstance(obj["bad_channels"], list):
                per_sub_runs[sub] += 1
                for c in obj["bad_channels"]:
                    per_sub_bad[sub][c] += 1
            for v in obj.values():
                walk(v, sub)
        elif isinstance(obj, list):
            for it in obj:
                walk(it, sub)

    for num, sub in cohort_num.items():
        if num in data:
            walk(data[num], sub)
    return per_sub_runs, per_sub_bad


def _cache_channel_stats():
    """Per subject, per channel: z of mean log-power, and corr with k nearest neighbors."""
    data, freqs, ch = d.load_cache("uniform")
    ch = list(ch)
    info = d._make_info(ch)
    pos = np.array([info["chs"][i]["loc"][:3] for i in range(len(ch))])
    # nearest 6 neighbors by 3D distance
    neigh = {}
    for i in range(len(ch)):
        dist = np.linalg.norm(pos - pos[i], axis=1)
        order = np.argsort(dist)
        neigh[i] = [j for j in order if j != i][:6]
    stats = {}  # (sub) -> dict ch -> (z, corr)
    for s in COHORT:
        if s not in data:
            continue
        psd = data[s][0]
        logp = np.log10(psd.mean(axis=2) + 1e-30)   # (n_ep, n_ch)
        chmean = logp.mean(axis=0)
        z = (chmean - chmean.mean()) / chmean.std()
        row = {}
        for i, c in enumerate(ch):
            r = np.corrcoef(logp[:, i], logp[:, neigh[i]].mean(axis=1))[0, 1]
            row[c] = (float(z[i]), float(r))
        stats[s] = row
    return ch, stats


def main():
    print("=" * 78)
    print("AUDITORÍA DE CALIDAD DE CANALES — cohorte N=6")
    print("=" * 78)

    per_sub_runs, per_sub_bad = _collect_log_bads()
    print("\n(a) % de runs marcado BAD por PyPREP, por sujeto (solo cohorte):")
    print(f"    runs por sujeto: " + ", ".join(f"{s}={per_sub_runs.get(s,0)}" for s in COHORT))
    # all channels seen
    all_ch = sorted({c for s in COHORT for c in per_sub_bad.get(s, {})})
    # build a table channel -> per-subject pct + mean pct
    print(f"\n    {'canal':6s} " + " ".join(f"{s.split('-')[1]:>4s}" for s in COHORT) + "  mean%  n_sub≥50%")
    chronic = []
    rows = []
    for c in all_ch:
        pcts = []
        for s in COHORT:
            nr = per_sub_runs.get(s, 0)
            nb = per_sub_bad.get(s, {}).get(c, 0)
            pcts.append(100 * nb / nr if nr else 0.0)
        meanp = float(np.mean(pcts))
        n_hi = int(np.sum(np.array(pcts) >= PCT_FLAG_CHRONIC))
        rows.append((meanp, c, pcts, n_hi))
    rows.sort(reverse=True)
    for meanp, c, pcts, n_hi in rows:
        if meanp < 5 and n_hi == 0:
            continue
        mark = "  <-- CRÓNICO" if (meanp >= PCT_FLAG_CHRONIC or n_hi >= 3) else ""
        print(f"    {c:6s} " + " ".join(f"{p:4.0f}" for p in pcts) + f"  {meanp:5.0f}  {n_hi:>6d}{mark}")
        if meanp >= PCT_FLAG_CHRONIC or n_hi >= 3:
            chronic.append(c)

    ch, stats = _cache_channel_stats()
    print("\n(b) Anomalía POST-preproc en el cache (z de log-power | corr con 6 vecinos), por sujeto:")
    print("    (resumen: nº sujetos con |z|>2  y  corr-vecinos media)")
    print(f"    {'canal':6s} {'n|z|>2':>6s} {'corr_med':>9s} {'z_med':>7s}  flags")
    summ = []
    for c in ch:
        zs = [stats[s][c][0] for s in COHORT if s in stats]
        rs = [stats[s][c][1] for s in COHORT if s in stats]
        n_out = int(np.sum(np.abs(zs) > Z_OUTLIER))
        summ.append((n_out, -np.mean(rs), c, np.mean(rs), np.median(zs)))
    summ.sort(reverse=True)
    for n_out, _, c, rmean, zmed in summ:
        if n_out == 0 and rmean > 0.5:
            continue
        flags = []
        if n_out >= 2:
            flags.append("z-outlier")
        if rmean < 0.5:
            flags.append("low-neigh-corr")
        print(f"    {c:6s} {n_out:>6d} {rmean:9.2f} {zmed:+7.2f}  {','.join(flags)}")

    print("\n" + "=" * 78)
    print("RESUMEN / RECOMENDACIÓN PRELIMINAR")
    print("=" * 78)
    print(f"  Canales CRÓNICOS por logs (≥{PCT_FLAG_CHRONIC:.0f}% runs o ≥3 sujetos): "
          f"{chronic if chronic else '(ninguno salvo abajo)'}")
    print("  -> candidatos a interpolación global en TODOS los runs.")
    print("  (cruzar con la anomalía post-preproc de (b) para confirmar.)")


if __name__ == "__main__":
    main()
