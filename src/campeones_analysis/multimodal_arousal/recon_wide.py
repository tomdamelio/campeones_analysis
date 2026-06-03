"""Re-derivación broadband (1-100 Hz) reusando el ICA ya ajustado, para el test EMG-vs-neural
en altas frecuencias (60-90 Hz). Tarea QA artefacto-vs-señal, deliverable highfreq (2026-06-03).

Motivación: el contraste SCR-vs-silent muestra un APLANAMIENTO del 1/f (exponent ↓). Ese
aplanamiento es ambiguo: tanto el EMG (potencia broadband que pico ~70-150 Hz) como un
aumento de excitación cortical (Gao-Voytek 2017, pendiente 1/f = proxy del balance E/I)
lo predicen en 1.5-40 Hz. Las dos hipótesis SOLO divergen por encima de ~40 Hz. Los
`*_desc-preproc_eeg` están lowpasseados a 48 Hz -> sin contenido >48 Hz. Hay que re-derivar
un stream broadband DESDE EL RAW, pero SIN re-correr PyPREP ni re-ajustar ICA: se reusa el
objeto ICA guardado (`*_desc-ica_ica.fif`, con `ica.exclude` congelado), que Step 4 ajustó
sobre una copia 1-100 Hz (`raw_for_ica`).

Receta (espeja scripts/preprocessing/04_preprocessing_eeg.py, secciones 3A/3B/7 y el bloque
raw_for_ica, líneas ~285-297, 828-850, 1034):
  read raw BIDS (data/raw, 500 Hz, sin filtrar) -> pick EEG (31 ch, sin FCz)
  -> notch [50,100] FIR zero-phase -> bandpass 1-100 FIR zero-phase
  -> add FCz (zero) -> set montage BC-32_FCz_modified.bvef
  -> info['bads'] = <bads de Step 4 desde channels.tsv> -> interpolate_bads(reset_bads=False)
  -> set_eeg_reference('average') -> ica.apply (objeto guardado) -> resample 250 Hz.

FIDELIDAD: `ica.apply()` es independiente de las anotaciones (las anotaciones solo afectaron
el FIT del ICA, ya congelado en el .fif) -> NO se transfieren anotaciones. La reconstrucción
es fiel usando solo 3 inputs congelados: raw BIDS, lista de bads guardada, objeto ICA guardado.

Run (gate de fidelidad sobre el cohort):
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.recon_wide
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.recon_wide --subjects sub-27
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, REPO
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    EDA_FS,
    NPZ_DIR,
    PREP,
    detect_scr_onsets_s,
    epoch_one_run,
    filter_clean_onsets,
    run_label,
    runs_for,
    sample_silent_controls,
)
import src.campeones_analysis.multimodal_arousal.erp_scr as _erp
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import (
    build_subject_epochs,
    compute_psd,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

RAW_ROOT = REPO / "data" / "raw"
BVEF = REPO / "scripts" / "preprocessing" / "BC-32_FCz_modified.bvef"

# Step-4 canonical filter params (04_preprocessing_eeg.py:276-296, 823-835)
NOTCH_FREQS = [50.0, 100.0]
ICA_HPASS = 1.0
ICA_LPASS = 100.0
RNG_SEED = 20260513  # = erp_scr.RNG seed, so silent controls reproduce a fresh cohort6 run


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------
def _eeg_dir_preproc(sub: str) -> Path:
    return PREP / sub / "ses-vr" / "eeg"


def _raw_vhdr(sub: str, label: str) -> Path:
    return RAW_ROOT / sub / "ses-vr" / "eeg" / f"{sub}_ses-vr_{label}_eeg.vhdr"


def _ica_fif(sub: str, label: str) -> Path:
    return _eeg_dir_preproc(sub) / f"{sub}_ses-vr_{label}_desc-ica_ica.fif"


def _channels_tsv(sub: str, label: str) -> Path:
    return _eeg_dir_preproc(sub) / f"{sub}_ses-vr_{label}_desc-preproc_channels.tsv"


def read_channel_table(sub: str, label: str) -> tuple[list[str], list[str]]:
    """Return (eeg_names_in_preproc_order, bad_channel_names) from the preproc channels.tsv.

    The status column is the on-disk record of raw_filtered.info['bads'] that Step 4 copied
    verbatim into raw_for_ica before fitting the ICA. utf-8-sig strips the BOM on 'name'.
    """
    df = pd.read_csv(_channels_tsv(sub, label), sep="\t", encoding="utf-8-sig")
    eeg = df[df["type"].astype(str).str.upper() == "EEG"]
    eeg_names = eeg["name"].astype(str).tolist()
    bads = eeg.loc[eeg["status"].astype(str).str.lower() == "bad", "name"].astype(str).tolist()
    return eeg_names, bads


# -----------------------------------------------------------------------------
# Re-derivation
# -----------------------------------------------------------------------------
def reconstruct_wide_run(sub: str, label: str, *, highpass: float = ICA_HPASS,
                         lowpass: float = ICA_LPASS, apply_ica: bool = True,
                         resample_hz: float = 250.0) -> mne.io.BaseRaw:
    """Re-derive a broadband (default 1-100 Hz) spatially-identical stream for one run,
    reusing the saved ICA. Mirrors Step-4 raw_for_ica + ica.apply. See module docstring."""
    eeg_names, bads = read_channel_table(sub, label)  # eeg_names incl FCz, preproc order
    recorded = [c for c in eeg_names if c != "FCz"]    # FCz is the (absent) online reference

    raw = mne.io.read_raw_brainvision(_raw_vhdr(sub, label), preload=True, verbose="ERROR")
    missing = [c for c in recorded if c not in raw.ch_names]
    if missing:
        raise RuntimeError(f"{sub} {label}: raw missing EEG channels {missing}")
    raw.pick(recorded)
    raw.set_channel_types({c: "eeg" for c in recorded})  # ensure 'eeg' so notch/filter pick them

    # 3A + 3B: notch then wide bandpass (reproduces raw_notched -> raw_for_ica band)
    raw.notch_filter(freqs=NOTCH_FREQS, picks="eeg", method="fir", phase="zero", verbose="ERROR")
    raw.filter(l_freq=highpass, h_freq=lowpass, picks="eeg", method="fir", phase="zero",
               verbose="ERROR")

    # Section 7 spatial state (identical to what the ICA saw): FCz -> montage -> bads ->
    # interpolate -> CAR. NO annotations (ica.apply is annotation-independent).
    raw = mne.add_reference_channels(raw.load_data(), ref_channels=["FCz"])
    raw.set_montage(mne.channels.read_custom_montage(BVEF), on_missing="ignore", verbose="ERROR")
    raw.info["bads"] = [b for b in bads if b in raw.ch_names]
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=False, verbose="ERROR")
    raw, _ = mne.set_eeg_reference(raw, ref_channels="average", copy=False, verbose="ERROR")

    if apply_ica:
        ica = mne.preprocessing.read_ica(_ica_fif(sub, label), verbose="ERROR")
        ica.apply(raw, verbose="ERROR")

    raw.resample(resample_hz, verbose="ERROR")
    # Clear bads for downstream: channels are interpolated (data fine) and the CAR above
    # already ran with bads set (excluding them from the reference, as Step 4 did). Leaving
    # per-run bads would break concatenate_epochs across runs with differing bad sets, and
    # the existing build_subject_epochs reads preproc via read_raw_brainvision (no bads).
    raw.info["bads"] = []
    raw.reorder_channels([c for c in eeg_names if c in raw.ch_names])
    return raw


def compute_psd_wide(epochs: mne.Epochs, fmin: float = 1.0, fmax: float = 95.0,
                     n_fft: int = 512) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Welch PSD up to fmax (default 95 Hz). Same n_fft as tfr_psd_scr (2 s @ 250 Hz)."""
    spectrum = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft,
                                  verbose="ERROR")
    return spectrum.get_data(), spectrum.freqs, spectrum.ch_names


# -----------------------------------------------------------------------------
# Wide epoching (mirror of tfr_psd_scr.build_subject_epochs with the wide loader)
# -----------------------------------------------------------------------------
def build_subject_epochs_wide(sub: str, *, lowpass: float = ICA_LPASS, resample: float = 250.0,
                              apply_ica: bool = True, reset_rng: bool = True
                              ) -> tuple[mne.Epochs | None, mne.Epochs | None]:
    """(real, silent) epochs for one subject from the re-derived broadband stream.

    Real-SCR onsets are deterministic (no RNG). Silent controls use erp_scr.RNG; reset_rng
    re-seeds it so a fresh process reproduces the cohort6 sampling sequence.
    """
    if reset_rng:
        _erp.RNG = np.random.default_rng(RNG_SEED)
    cont_path = NPZ_DIR / f"{sub}_continuous.npz"
    if not cont_path.exists():
        return None, None
    cont = np.load(cont_path, allow_pickle=True)
    runs_in_npz = list(cont["runs"])

    real_list: list[mne.Epochs] = []
    silent_list: list[mne.Epochs] = []
    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs_in_npz:
            continue
        try:
            raw = reconstruct_wide_run(sub, label, lowpass=lowpass, apply_ica=apply_ica,
                                       resample_hz=resample)
            duration = float(raw.times[-1])
            eda_phasic = np.asarray(cont[f"{label}__eda_phasic"], float)
            onsets_all = detect_scr_onsets_s(eda_phasic, EDA_FS)
            onsets_all = onsets_all[onsets_all < duration]
            onsets_clean = filter_clean_onsets(onsets_all, eda_phasic, EDA_FS)
            silent_t = sample_silent_controls(
                n_target=len(onsets_clean), duration_s=duration,
                phasic=eda_phasic, fs=EDA_FS, rng=_erp.RNG, avoid_onsets_s=onsets_clean,
            )
            ep_real = epoch_one_run(raw, onsets_clean, code=1)
            ep_silent = epoch_one_run(raw, silent_t, code=2)
            if ep_real is not None:
                real_list.append(ep_real)
            if ep_silent is not None:
                silent_list.append(ep_silent)
        except Exception as e:
            print(f"  {label}: FAILED -- {e}", flush=True)

    if not real_list or not silent_list:
        return None, None
    return (mne.concatenate_epochs(real_list, verbose="ERROR"),
            mne.concatenate_epochs(silent_list, verbose="ERROR"))


# -----------------------------------------------------------------------------
# Fidelity gate (HARD GATE)
# -----------------------------------------------------------------------------
def validate_against_preproc(sub: str, *, tol_median_db: float = 0.5,
                             tol_max_db: float = 2.0) -> dict:
    """Compare 1-40 Hz REAL-epoch PSD of the reconstruction vs the existing analysis stream.

    Real epochs are deterministic (onsets, no RNG) -> 1:1 comparable. The recon (ICA on
    1-100) and the preproc (ICA on 0.1-48) share the same unmixing matrix; within 1-40 Hz
    they must match. Returns a row dict with median/max |ΔdB| and pass flag.
    """
    real_pre, _ = build_subject_epochs(sub)
    real_wide, _ = build_subject_epochs_wide(sub, lowpass=ICA_LPASS, apply_ica=True)
    if real_pre is None or real_wide is None:
        return dict(subject=sub, status="no_epochs", passed=False)

    p_pre, f_pre, ch_pre = compute_psd(real_pre)                         # fmax=40
    p_wide, f_wide, ch_wide = compute_psd_wide(real_wide, fmax=40.0)     # restrict to 1-40
    # shared channels, shared freq bins (both 1-40 @ same n_fft/sfreq -> identical grid)
    shared = [c for c in ch_pre if c in ch_wide]
    ip = [ch_pre.index(c) for c in shared]
    iw = [ch_wide.index(c) for c in shared]
    nf = min(len(f_pre), len(f_wide))
    db_pre = 10.0 * np.log10(p_pre.mean(axis=0)[ip][:, :nf] + 1e-30)     # (n_ch, n_freq)
    db_wide = 10.0 * np.log10(p_wide.mean(axis=0)[iw][:, :nf] + 1e-30)
    d = np.abs(db_pre - db_wide)
    med, mx = float(np.median(d)), float(np.max(d))
    passed = (med < tol_median_db) and (mx < tol_max_db)
    print(f"  {sub}: n_real pre={len(real_pre)} wide={len(real_wide)}  "
          f"median|ΔdB|={med:.3f}  max|ΔdB|={mx:.3f}  -> {'PASS' if passed else 'FAIL'}",
          flush=True)
    return dict(subject=sub, n_real_pre=len(real_pre), n_real_wide=len(real_wide),
                n_shared_ch=len(shared), median_abs_db=round(med, 4),
                max_abs_db=round(mx, 4), passed=bool(passed))


def main() -> None:
    p = argparse.ArgumentParser(description="Fidelity gate for the broadband re-derivation.")
    p.add_argument("--subjects", nargs="+", default=None)
    args = p.parse_args()
    subs = args.subjects if args.subjects else list(COHORT)

    out_dir = NPZ_DIR.parent / "qa_artifact_vs_signal" / "highfreq" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"recon_wide :: fidelity gate (recon 1-40 vs preproc)  subjects={subs}")
    print(f"  tol: median|ΔdB|<0.5, max|ΔdB|<2.0  (HARD GATE before any 60-90 Hz number)")
    print("=" * 78, flush=True)
    rows = [validate_against_preproc(s) for s in subs]
    df = pd.DataFrame(rows)
    csv = out_dir / "recon_fidelity.csv"
    df.to_csv(csv, index=False)
    print("\n" + df.to_string(index=False), flush=True)
    n_pass = int(df.get("passed", pd.Series(dtype=bool)).sum())
    print(f"\n{n_pass}/{len(df)} subjects PASS. -> {csv}", flush=True)
    if n_pass < len(df):
        print("WARNING: not all subjects passed the fidelity gate. Do NOT trust 60-90 Hz "
              "numbers until the reconstruction matches.", flush=True)


if __name__ == "__main__":
    main()
