"""Diagnose whether L_EYE and R_EYE are functional EOG channels.

Run:
    micromamba run -n campeones python -m src.campeones_analysis.utils.diagnose_eog_channels

Context. R_EYE is the lateral electrode (right of the right eye) and should
capture saccades. L_EYE is expected to be the infraorbital electrode (below
the right eye) and should capture blinks, but tapping it during setup
produced no visible response, so its functionality is in doubt. This
diagnostic compares both channels on a single run to decide whether
find_bads_eog should receive one or both.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_bids import BIDSPath, read_raw_bids

SUBJECT = "27"
SESSION = "vr"
TASK = "01"
ACQUISITION = "a"
RUN = "002"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "data", "derivatives", "diagnostics", "eog_channels")
os.makedirs(OUT_DIR, exist_ok=True)


def load_raw() -> mne.io.BaseRaw:
    bids_path = BIDSPath(
        subject=SUBJECT,
        session=SESSION,
        task=TASK,
        acquisition=ACQUISITION,
        run=RUN,
        datatype="eeg",
        suffix="eeg",
        extension=".vhdr",
        root=os.path.join(REPO_ROOT, "data", "raw"),
    )
    raw = read_raw_bids(bids_path, verbose="ERROR")
    raw.load_data()
    raw.filter(l_freq=1.0, h_freq=40.0, picks=["eeg", "eog"], verbose="ERROR")
    return raw


def channel_stats(raw: mne.io.BaseRaw, ch: str) -> dict:
    data = raw.get_data(picks=[ch])[0] * 1e6
    return {
        "std_uv": float(np.std(data)),
        "ptp_uv": float(np.ptp(data)),
        "abs_median_uv": float(np.median(np.abs(data - np.median(data)))),
    }


def correlate_with_frontal(raw: mne.io.BaseRaw, eog_ch: str, frontal_ch: str) -> float:
    eog = raw.get_data(picks=[eog_ch])[0]
    frontal = raw.get_data(picks=[frontal_ch])[0]
    return float(np.corrcoef(eog, frontal)[0, 1])


def count_eog_events(raw: mne.io.BaseRaw, ch: str) -> int:
    try:
        events = mne.preprocessing.find_eog_events(raw, ch_name=ch, verbose="ERROR")
        return int(events.shape[0])
    except Exception as exc:
        print(f"    find_eog_events failed on {ch}: {exc}")
        return -1


def plot_traces(raw: mne.io.BaseRaw, out_path: str, window_s: float = 30.0) -> None:
    sfreq = raw.info["sfreq"]
    n_samples = int(window_s * sfreq)
    t0 = int(60 * sfreq) if raw.n_times > int(90 * sfreq) else 0
    t1 = min(raw.n_times, t0 + n_samples)
    times = raw.times[t0:t1]

    channels = ["L_EYE", "R_EYE", "Fp1", "Fp2"]
    available = [ch for ch in channels if ch in raw.ch_names]

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 2.2 * len(available)),
                             sharex=True)
    if len(available) == 1:
        axes = [axes]
    for ax, ch in zip(axes, available):
        data = raw.get_data(picks=[ch])[0, t0:t1] * 1e6
        ax.plot(times, data, linewidth=0.6)
        ax.set_ylabel(f"{ch}\n(µV)")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"sub-{SUBJECT} run-{RUN} — EOG vs frontal (1–40 Hz)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def blink_locked_ranking(raw: mne.io.BaseRaw, trigger_ch: str = "R_EYE"):
    """Rank all channels by blink-locked response.

    Uses blink events from a known-working EOG channel and computes the
    blink-locked average on every other channel. A mislabeled infraorbital
    should stand out as a non-Fp channel with (i) peak amplitude comparable to
    or larger than Fp2 and (ii) polarity inverted relative to Fp2 (negative
    correlation of the blink ERP).
    """
    events = mne.preprocessing.find_eog_events(
        raw, ch_name=trigger_ch, verbose="ERROR"
    )
    if events.shape[0] < 5:
        print(f"  Too few blink events from {trigger_ch} ({events.shape[0]}) "
              f"— skipping blink-locked ranking")
        return None

    picks = mne.pick_types(raw.info, eeg=True, eog=True, misc=True,
                           exclude=[])
    epochs = mne.Epochs(
        raw, events, event_id=None, tmin=-0.3, tmax=0.5,
        baseline=(-0.3, -0.1), picks=picks, preload=True,
        reject_by_annotation=False, verbose="ERROR",
    )
    evoked = epochs.average()
    data = evoked.data * 1e6
    times = evoked.times
    mask_peak = (times >= 0.0) & (times <= 0.3)

    fp2_idx = evoked.ch_names.index("Fp2") if "Fp2" in evoked.ch_names else None
    fp2_erp = data[fp2_idx] if fp2_idx is not None else None

    rows = []
    for i, ch in enumerate(evoked.ch_names):
        erp = data[i]
        peak_abs = float(np.max(np.abs(erp[mask_peak])))
        peak_signed = float(erp[mask_peak][np.argmax(np.abs(erp[mask_peak]))])
        corr_fp2 = (float(np.corrcoef(erp, fp2_erp)[0, 1])
                    if fp2_erp is not None else np.nan)
        rows.append((ch, peak_abs, peak_signed, corr_fp2))

    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"  Blink events used: {events.shape[0]} (trigger: {trigger_ch})")
    print("  Top 15 channels by blink-locked peak amplitude:")
    print(f"    {'channel':<10} {'|peak| µV':>10} {'signed µV':>10} {'corr vs Fp2':>12}")
    for ch, peak_abs, peak_signed, corr in rows[:15]:
        marker = ""
        if ch not in {"Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "AFz"}:
            if corr is not None and corr < -0.5 and peak_abs > 20:
                marker = "  <-- INFRAORBITAL CANDIDATE (inverted)"
            elif peak_abs > 80 and ch not in {"R_EYE", "L_EYE"}:
                marker = "  <-- unusually large non-frontal response"
        print(f"    {ch:<10} {peak_abs:>10.1f} {peak_signed:>+10.1f} "
              f"{corr:>+12.3f}{marker}")

    candidates = [r for r in rows
                  if r[0] not in {"Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8",
                                  "AFz", "R_EYE", "L_EYE"}
                  and r[3] is not None and r[3] < -0.3]
    return {"events": events, "evoked": evoked, "ranking": rows,
            "candidates": candidates}


def plot_blink_erps(evoked: mne.Evoked, channels, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    times = evoked.times * 1000
    for ch in channels:
        if ch not in evoked.ch_names:
            continue
        idx = evoked.ch_names.index(ch)
        ax.plot(times, evoked.data[idx] * 1e6, label=ch, linewidth=1.3)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Time relative to blink (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("Blink-locked average (events detected on R_EYE)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def verdict(stats_l: dict, stats_r: dict, corr_l: float, corr_r: float,
            n_events_l: int, n_events_r: int) -> str:
    l_dead = stats_l["std_uv"] < 0.5 or stats_l["std_uv"] < 0.1 * stats_r["std_uv"]
    l_weak_corr = abs(corr_l) < 0.3
    l_no_events = n_events_l <= 0 or (n_events_r > 0 and n_events_l < 0.3 * n_events_r)

    if l_dead:
        return ("L_EYE looks DEAD (near-zero variance vs R_EYE) → "
                "use only R_EYE in find_bads_eog; rely on ICLabel for blinks.")
    if l_weak_corr and l_no_events:
        return ("L_EYE is alive but does NOT track blinks (low corr with Fp2, "
                "few/no EOG events) → use only R_EYE in find_bads_eog.")
    if (not l_weak_corr) and n_events_l > 0:
        return ("L_EYE appears FUNCTIONAL as a blink channel (correlates with "
                "Fp2 and yields EOG events) → use ch_name=['R_EYE', 'L_EYE'].")
    return ("AMBIGUOUS: L_EYE is not obviously dead but signal is weak. "
            "Recommend using only R_EYE for now; reinspect on more runs.")


def main() -> None:
    print(f"Loading sub-{SUBJECT} ses-{SESSION} task-{TASK} acq-{ACQUISITION} run-{RUN}")
    raw = load_raw()
    print(f"  sfreq={raw.info['sfreq']} Hz, duration={raw.times[-1]:.1f} s")
    print(f"  channels present: L_EYE={'L_EYE' in raw.ch_names}, "
          f"R_EYE={'R_EYE' in raw.ch_names}, "
          f"Fp1={'Fp1' in raw.ch_names}, Fp2={'Fp2' in raw.ch_names}")

    print("\n[1] Per-channel amplitude stats (µV)")
    stats_l = channel_stats(raw, "L_EYE")
    stats_r = channel_stats(raw, "R_EYE")
    print(f"  L_EYE: std={stats_l['std_uv']:.2f}  ptp={stats_l['ptp_uv']:.1f}  "
          f"MAD={stats_l['abs_median_uv']:.2f}")
    print(f"  R_EYE: std={stats_r['std_uv']:.2f}  ptp={stats_r['ptp_uv']:.1f}  "
          f"MAD={stats_r['abs_median_uv']:.2f}")

    print("\n[2] Correlation with frontal channels (blinks should correlate strongly)")
    corr_l_fp2 = correlate_with_frontal(raw, "L_EYE", "Fp2")
    corr_r_fp2 = correlate_with_frontal(raw, "R_EYE", "Fp2")
    corr_l_fp1 = correlate_with_frontal(raw, "L_EYE", "Fp1")
    corr_r_fp1 = correlate_with_frontal(raw, "R_EYE", "Fp1")
    print(f"  L_EYE vs Fp2: r={corr_l_fp2:+.3f}    L_EYE vs Fp1: r={corr_l_fp1:+.3f}")
    print(f"  R_EYE vs Fp2: r={corr_r_fp2:+.3f}    R_EYE vs Fp1: r={corr_r_fp1:+.3f}")

    print("\n[3] find_eog_events count (higher = more blinks detected)")
    n_events_l = count_eog_events(raw, "L_EYE")
    n_events_r = count_eog_events(raw, "R_EYE")
    duration_min = raw.times[-1] / 60.0
    print(f"  L_EYE: {n_events_l} events  ({n_events_l / duration_min:.1f}/min)")
    print(f"  R_EYE: {n_events_r} events  ({n_events_r / duration_min:.1f}/min)")

    print("\n[4] Saving time-series plot")
    plot_path = os.path.join(
        OUT_DIR, f"sub-{SUBJECT}_run-{RUN}_eog_diagnostic.png"
    )
    plot_traces(raw, plot_path)
    print(f"  → {plot_path}")

    print("\n[5] Blink-locked ranking across ALL channels (mislabel check)")
    print("  [5a] Using R_EYE as trigger:")
    ranking = blink_locked_ranking(raw, trigger_ch="R_EYE")
    print("\n  [5b] Using Fp2 as trigger (cross-check):")
    ranking_fp2 = blink_locked_ranking(raw, trigger_ch="Fp2")

    print("\n  [5c] Sanity check on top candidate: per-trial consistency")
    if ranking is not None:
        top_non_frontal = next(
            (r for r in ranking["ranking"]
             if r[0] not in {"Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "AFz",
                             "R_EYE", "L_EYE"}),
            None,
        )
        if top_non_frontal is not None:
            top_ch = top_non_frontal[0]
            top_std = channel_stats(raw, top_ch)
            print(f"    {top_ch} overall stats: std={top_std['std_uv']:.1f} µV  "
                  f"ptp={top_std['ptp_uv']:.1f} µV  MAD={top_std['abs_median_uv']:.2f} µV")
            events = ranking["events"]
            sfreq = raw.info["sfreq"]
            data_ch = raw.get_data(picks=[top_ch])[0] * 1e6
            trial_peaks = []
            for ev in events:
                s0 = ev[0]
                s_start = max(0, s0)
                s_end = min(len(data_ch), s0 + int(0.3 * sfreq))
                if s_end > s_start:
                    seg = data_ch[s_start:s_end]
                    trial_peaks.append(float(np.max(np.abs(seg))))
            if trial_peaks:
                tp = np.array(trial_peaks)
                print(f"    per-blink peak |amp| on {top_ch}: "
                      f"median={np.median(tp):.1f}  mean={np.mean(tp):.1f}  "
                      f"max={np.max(tp):.1f}  #trials={len(tp)}")
                print(f"    → if median ≈ mean → consistent response; "
                      f"if max ≫ median → outlier-driven")
    if ranking is not None and ranking["candidates"]:
        cand_names = [c[0] for c in ranking["candidates"][:3]]
        plot_channels = ["Fp2", "Fp1", "R_EYE", "L_EYE"] + cand_names
        erp_plot = os.path.join(
            OUT_DIR, f"sub-{SUBJECT}_run-{RUN}_blink_erp_candidates.png"
        )
        plot_blink_erps(ranking["evoked"], plot_channels, erp_plot)
        print(f"\n  → ERP plot: {erp_plot}")
    elif ranking is not None:
        plot_channels = ["Fp2", "Fp1", "R_EYE", "L_EYE"]
        erp_plot = os.path.join(
            OUT_DIR, f"sub-{SUBJECT}_run-{RUN}_blink_erp_candidates.png"
        )
        plot_blink_erps(ranking["evoked"], plot_channels, erp_plot)
        print(f"  No inverted-polarity infraorbital candidate found.")
        print(f"  → ERP plot: {erp_plot}")

    print("\n[6] Verdict")
    msg = verdict(stats_l, stats_r, corr_l_fp2, corr_r_fp2, n_events_l, n_events_r)
    print(f"  {msg}")


if __name__ == "__main__":
    main()
