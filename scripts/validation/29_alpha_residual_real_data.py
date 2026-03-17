#!/usr/bin/env python
"""Verify alpha residual decay with increasing N in REAL EEG data.

Implements Task 8.3 from the research diary.

Generates a large pool of random epochs from the preprocessed EEG,
then for each target N subsamples N epochs, averages them, and measures
the RMS of the alpha-band (8-12 Hz) residual in the ERP. Repeats with
multiple random seeds to get confidence intervals.

Expected result: RMS decays as ~1/√N, confirming that the alpha residual
in averaged ERPs is an artifact of finite-N averaging (consistent with
simulations in Task 8.1).

Usage
-----
    micromamba run -n campeones python scripts/validation/29_alpha_residual_real_data.py --subject 27
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "alpha_residual_real"

SESSION = "vr"

# Epoch parameters (same as 23b)
TMIN = -4.5
TMAX = 3.0
BASELINE = (-4.5, -4.0)

# Use shorter minimum spacing to fit more epochs per run
# With EPOCH_SPAN=4.0s we can fit ~2x more epochs than with 7.5s
# The epochs still don't overlap (TMIN=-4.5, TMAX=3.0 → 7.5s window,
# but we only need non-overlapping ONSETS, not non-overlapping windows)
# Actually we DO need non-overlapping windows for independence.
# Keep EPOCH_SPAN = 7.5s for proper independence.
EPOCH_SPAN = abs(TMIN) + TMAX  # 7.5s

# Target: generate as many random epochs as possible per run
TARGET_EPOCHS_PER_RUN = 100  # will be capped by recording length

# N values to test
N_VALUES = [20, 50, 74, 100, 150, 200, 280]

# Number of random subsamples per N (for confidence intervals)
N_REPETITIONS = 50

# Alpha band for filtering
ALPHA_LOW = 8.0
ALPHA_HIGH = 12.0

# ROIs
ROIS = {
    "Frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
                "FC1", "FC2", "FC5", "FC6", "FCz"],
    "Temporal": ["T7", "T8", "FT9", "FT10", "TP9", "TP10"],
    "Parietal": ["P3", "P4", "P7", "P8", "Pz", "CP1", "CP2", "CP5", "CP6"],
    "Occipital": ["O1", "O2"],
}

RANDOM_SEED = 42
EVENT_ID = {"RANDOM": 99}

RUNS_CONFIG: dict[str, list[dict]] = {
    "27": [
        {"run": "002", "acq": "a", "task": "01"},
        {"run": "003", "acq": "a", "task": "02"},
        {"run": "004", "acq": "a", "task": "03"},
        {"run": "006", "acq": "a", "task": "04"},
        {"run": "007", "acq": "b", "task": "01"},
        {"run": "009", "acq": "b", "task": "03"},
        {"run": "010", "acq": "b", "task": "04"},
    ],
}

EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6',
    'FT9', 'FT10', 'TP9', 'TP10', 'FCz',
]


def generate_random_onsets(
    duration_s: float, n_target: int, epoch_span: float,
    margin: float, rng: np.random.Generator,
) -> np.ndarray:
    """Generate non-overlapping random onsets."""
    lo = margin
    hi = duration_s - margin
    usable = hi - lo
    if usable <= 0:
        return np.array([])
    max_epochs = int(usable // epoch_span)
    n = min(n_target, max_epochs)
    if n <= 0:
        return np.array([])
    slack = usable - (n - 1) * epoch_span
    raw_pts = np.sort(rng.uniform(0, slack, size=n))
    onsets = lo + raw_pts + np.arange(n) * epoch_span
    return onsets


def generate_epoch_pool(subject: str, seed: int) -> mne.Epochs | None:
    """Generate a large pool of random epochs across all runs."""
    runs = RUNS_CONFIG.get(subject)
    if runs is None:
        return None

    rng = np.random.default_rng(seed)
    all_epochs: list[mne.Epochs] = []

    for rc in runs:
        run_id, task, acq = rc["run"], rc["task"], rc["acq"]
        label = f"task-{task}_acq-{acq}_run-{run_id}"

        eeg_dir = PREPROC_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"
        vhdr = eeg_dir / (
            f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}"
            f"_run-{run_id}_desc-preproc_eeg.vhdr"
        )
        if not vhdr.exists():
            print(f"  SKIP {label}")
            continue

        raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
        sfreq = raw.info["sfreq"]
        duration_s = raw.n_times / sfreq
        available_chs = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        margin = max(abs(TMIN), TMAX) + 0.5

        onsets = generate_random_onsets(
            duration_s, TARGET_EPOCHS_PER_RUN, EPOCH_SPAN, margin, rng,
        )
        if len(onsets) == 0:
            continue

        samples = np.round(onsets * sfreq).astype(int) + raw.first_samp
        mne_events = np.column_stack([
            samples,
            np.zeros(len(samples), dtype=int),
            np.full(len(samples), EVENT_ID["RANDOM"], dtype=int),
        ])

        try:
            epochs = mne.Epochs(
                raw, events=mne_events, event_id=EVENT_ID,
                tmin=TMIN, tmax=TMAX, picks=available_chs,
                baseline=BASELINE, preload=True, verbose=False,
            )
            all_epochs.append(epochs)
            print(f"  {label}: {len(epochs)} random epochs")
        except Exception as exc:
            print(f"  {label}: error — {exc}")

    if not all_epochs:
        return None

    grand = mne.concatenate_epochs(all_epochs)
    print(f"  Total pool: {len(grand)} random epochs")
    return grand


def bandpass_filter_alpha(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Apply 8-12 Hz bandpass filter."""
    sos = butter(4, [ALPHA_LOW, ALPHA_HIGH], btype='band', fs=sfreq, output='sos')
    return sosfiltfilt(sos, data, axis=-1)


def compute_alpha_rms(erp: np.ndarray, sfreq: float) -> float:
    """Compute RMS of alpha-filtered ERP signal. Returns value in µV."""
    filtered = bandpass_filter_alpha(erp, sfreq)
    # MNE data is in Volts, convert to µV
    return float(np.sqrt(np.mean((filtered * 1e6) ** 2)))


def compute_rms_vs_n(
    epochs_data: np.ndarray, sfreq: float, ch_indices: list[int],
    n_values: list[int], n_reps: int, rng: np.random.Generator,
) -> dict:
    """For each N, subsample N epochs, average, measure alpha RMS.

    Returns dict with keys: n_values, rms_mean, rms_std, rms_all
    """
    n_total = epochs_data.shape[0]
    results = {"n_values": [], "rms_mean": [], "rms_std": [], "rms_all": []}

    for n_val in n_values:
        if n_val > n_total:
            print(f"    Skipping N={n_val} (only {n_total} epochs available)")
            continue

        rms_list = []
        for _ in range(n_reps):
            idx = rng.choice(n_total, size=n_val, replace=False)
            subset = epochs_data[idx]
            # Average across selected epochs, then average across ROI channels
            erp = subset.mean(axis=0)  # (n_channels, n_times)
            roi_erp = erp[ch_indices].mean(axis=0)  # (n_times,)
            rms = compute_alpha_rms(roi_erp, sfreq)
            rms_list.append(rms)

        results["n_values"].append(n_val)
        results["rms_mean"].append(np.mean(rms_list))
        results["rms_std"].append(np.std(rms_list))
        results["rms_all"].append(rms_list)
        print(f"    N={n_val:4d}: RMS = {np.mean(rms_list):.4f} ± {np.std(rms_list):.4f}")

    return results


def plot_rms_vs_n(all_roi_results: dict, output_dir: Path, subject: str) -> None:
    """Plot RMS vs N for all ROIs, with 1/√N reference curve."""
    n_rois = len(all_roi_results)
    fig, axes = plt.subplots(1, n_rois, figsize=(5 * n_rois, 5), sharey=True)
    if n_rois == 1:
        axes = [axes]

    for ax, (roi_name, res) in zip(axes, all_roi_results.items()):
        n_arr = np.array(res["n_values"])
        mean_arr = np.array(res["rms_mean"])
        std_arr = np.array(res["rms_std"])

        # Data points with error bars
        ax.errorbar(n_arr, mean_arr, yerr=std_arr, fmt='o-', color='C0',
                     capsize=4, label='Datos reales (media ± SD)')

        # 1/√N reference curve, scaled to match the first data point
        if len(n_arr) > 0:
            n_ref = np.linspace(n_arr.min(), n_arr.max(), 100)
            # Scale: RMS(N) ≈ k / √N → k = RMS(N0) * √N0
            k = mean_arr[0] * np.sqrt(n_arr[0])
            ax.plot(n_ref, k / np.sqrt(n_ref), '--', color='C3',
                    alpha=0.7, label=f'Referencia 1/√N (k={k:.2f})')

        ax.set_xlabel('N (épocas promediadas)')
        if ax == axes[0]:
            ax.set_ylabel('RMS alfa (8-12 Hz) del ERP [µV]')
        ax.set_title(roi_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Residuo alfa en datos reales vs N — sub-{subject}\n'
        f'({N_REPETITIONS} repeticiones por N, épocas random)',
        fontsize=12,
    )
    fig.tight_layout()
    out = output_dir / f"sub-{subject}_alpha_rms_vs_n.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out}")


def plot_example_erps(
    epochs_data: np.ndarray, sfreq: float, times: np.ndarray,
    ch_indices: list[int], roi_name: str,
    output_dir: Path, subject: str,
) -> None:
    """Plot example ERPs for N=20, 74, 280 to visually show residual decay."""
    rng = np.random.default_rng(123)
    n_total = epochs_data.shape[0]
    example_ns = [n for n in [20, 74, max(N_VALUES)] if n <= n_total]

    fig, axes = plt.subplots(len(example_ns), 1, figsize=(12, 3 * len(example_ns)),
                              sharex=True, sharey=True)
    if len(example_ns) == 1:
        axes = [axes]

    # Crop to visualization window
    vis_mask = times >= -3.0
    vis_times = times[vis_mask]

    for ax, n_val in zip(axes, example_ns):
        idx = rng.choice(n_total, size=n_val, replace=False)
        erp = epochs_data[idx].mean(axis=0)
        roi_erp = erp[ch_indices].mean(axis=0)
        roi_erp_vis = roi_erp[vis_mask]

        # Also show alpha-filtered version
        alpha_erp = bandpass_filter_alpha(roi_erp, sfreq)
        alpha_erp_vis = alpha_erp[vis_mask]

        rms = compute_alpha_rms(roi_erp, sfreq)

        ax.plot(vis_times * 1000, roi_erp_vis * 1e6, color='C0', alpha=0.5,
                lw=0.8, label='ERP')
        ax.plot(vis_times * 1000, alpha_erp_vis * 1e6, color='C3', lw=1.5,
                label='Alfa filtrado (8-12 Hz)')
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='black', ls='--', lw=0.8)
        ax.set_ylabel('µV')
        ax.set_title(f'N={n_val} épocas — RMS alfa = {rms:.2f} µV')
        ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel('Tiempo (ms)')
    fig.suptitle(
        f'ERP promediado de épocas random — {roi_name} — sub-{subject}',
        fontsize=12,
    )
    fig.tight_layout()
    out = output_dir / f"sub-{subject}_example_erps_{roi_name.lower()}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Example ERPs plot saved: {out}")


def run_pipeline(subject: str) -> None:
    print("=" * 60)
    print(f"29 — Alpha residual in real data vs N — sub-{subject}")
    print(f"     N values: {N_VALUES}")
    print(f"     Repetitions per N: {N_REPETITIONS}")
    print("=" * 60)

    # Generate large pool of random epochs
    print("\nGenerating random epoch pool...")
    pool = generate_epoch_pool(subject, seed=RANDOM_SEED)
    if pool is None:
        print("ERROR: No epochs generated.")
        return

    n_total = len(pool)
    sfreq = pool.info["sfreq"]
    ch_names = pool.ch_names
    times = pool.times
    data = pool.get_data()  # (n_epochs, n_channels, n_times)
    print(f"\n  Pool: {n_total} epochs, {len(ch_names)} channels, "
          f"sfreq={sfreq} Hz")

    output_dir = RESULTS_ROOT / f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute RMS vs N for each ROI
    rng = np.random.default_rng(RANDOM_SEED + 1)
    all_roi_results = {}

    for roi_name, roi_chs in ROIS.items():
        ch_indices = [ch_names.index(ch) for ch in roi_chs if ch in ch_names]
        if not ch_indices:
            print(f"\n  {roi_name}: no channels found, skipping")
            continue

        print(f"\n  ROI: {roi_name} ({len(ch_indices)} channels)")
        results = compute_rms_vs_n(
            data, sfreq, ch_indices, N_VALUES, N_REPETITIONS, rng,
        )
        all_roi_results[roi_name] = results

    # Plot RMS vs N
    plot_rms_vs_n(all_roi_results, output_dir, subject)

    # Plot example ERPs for Occipital (main ROI of interest)
    occ_chs = [ch_names.index(ch) for ch in ROIS["Occipital"] if ch in ch_names]
    if occ_chs:
        plot_example_erps(data, sfreq, times, occ_chs, "Occipital",
                          output_dir, subject)

    # Save numerical results
    import json
    results_dict = {}
    for roi_name, res in all_roi_results.items():
        results_dict[roi_name] = {
            "n_values": res["n_values"],
            "rms_mean": [float(x) for x in res["rms_mean"]],
            "rms_std": [float(x) for x in res["rms_std"]],
        }
    json_path = output_dir / f"sub-{subject}_alpha_rms_vs_n.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n  Results JSON saved: {json_path}")

    print(f"\nDone. Output in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpha residual decay with N in real EEG data",
    )
    parser.add_argument("--subject", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(subject=args.subject)
