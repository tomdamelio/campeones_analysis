"""EEG quality assurance using Autoreject.

Pure functions for applying autoreject-based epoch rejection and generating
rejection visualisations. Separates computation from I/O so that results
can be tested independently of file system access.

Requirements: 1.1, 1.2, 1.4
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from autoreject import AutoReject


def run_autoreject_qa(
    epochs: mne.Epochs,
    random_seed: int = 42,
) -> dict:
    """Apply AutoReject to EEG epochs and return rejection statistics.

    Fits AutoReject on the provided epochs to compute data-driven rejection
    thresholds per channel. Returns a summary dict with rejection counts and
    the reject_log for downstream visualisation.

    Args:
        epochs: MNE Epochs object containing the EEG data to evaluate.
            Should already be preprocessed (filtered, re-referenced).
        random_seed: Random seed for AutoReject reproducibility.

    Returns:
        Dict with keys:
            - ``n_epochs_total`` (int): Total number of epochs.
            - ``n_epochs_rejected`` (int): Number of epochs marked as bad.
            - ``rejection_pct`` (float): Percentage of rejected epochs.
            - ``reject_log`` (autoreject.RejectLog): Full rejection log for
              heatmap generation.
            - ``channel_names`` (list[str]): Channel names in epoch order.

    References:
        Requirements 1.1, 1.2.
        Jas et al. (2017). Autoreject: Automated artifact rejection for MEG
        and EEG data. NeuroImage.
    """
    # Limit interpolation to max 8 channels (25% of 32 channels) as requested.
    # Consensus grid [0.25, 1.0] aligns with the 25% interpolation cap.
    n_interpolate = np.array([1, 4, 8])
    consensus = np.linspace(0.5, 1.0, 11)

    ar = AutoReject(
        n_interpolate=n_interpolate,
        consensus=consensus,
        random_state=random_seed,
        verbose=False,
        n_jobs=-1,  # Use all cores
    )
    ar.fit(epochs)
    reject_log = ar.get_reject_log(epochs)

    n_total = len(epochs)
    n_rejected = int(np.sum(reject_log.bad_epochs))
    rejection_pct = compute_rejection_percentage(n_total=n_total, n_rejected=n_rejected)

    labels = np.array(reject_log.labels)
    ch_names = np.array(epochs.ch_names)
    bad_epochs_idx = np.where(reject_log.bad_epochs)[0].tolist()
    interp_counts = (labels == 1).sum(axis=1).tolist()
    interp_channels = [ch_names[labels[i] == 1].tolist() for i in range(n_total)]

    return {
        "n_epochs_total": n_total,
        "n_epochs_rejected": n_rejected,
        "rejection_pct": rejection_pct,
        "bad_epochs_idx": bad_epochs_idx,
        "interp_counts": interp_counts,
        "interp_channels": interp_channels,
        "ar_n_interpolate": ar.n_interpolate_.get("eeg", ar.n_interpolate_),
        "ar_consensus": ar.consensus_.get("eeg", ar.consensus_),
        "reject_log": reject_log,
        "channel_names": epochs.ch_names,
    }


def compute_rejection_percentage(n_total: int, n_rejected: int) -> float:
    """Compute the percentage of rejected epochs.

    Pure arithmetic function extracted for testability.

    Args:
        n_total: Total number of epochs. Must be > 0.
        n_rejected: Number of rejected epochs. Must satisfy 0 <= n_rejected <= n_total.

    Returns:
        Rejection percentage as a float in [0.0, 100.0].

    References:
        Requirements 1.2.
    """
    return (n_rejected / n_total) * 100.0


def plot_rejection_heatmap(
    reject_log: object,
    channel_names: list[str],
    output_path: Path,
) -> None:
    """Generate and save a heatmap of the epoch rejection pattern.

    Visualises which channels and epochs were rejected by AutoReject as a
    2-D heatmap (channels × epochs). Useful for identifying systematic
    artefact patterns across runs.

    Args:
        reject_log: AutoReject RejectLog object containing the ``labels``
            array of shape (n_epochs, n_channels) with values 0 (good),
            1 (interpolated), or 2 (bad).
        channel_names: List of channel names corresponding to columns in
            ``reject_log.labels``.
        output_path: Full path (including filename) where the PNG figure
            will be saved.

    References:
        Requirements 1.4.
    """
    labels = np.array(reject_log.labels)  # (n_epochs, n_channels)

    fig, ax = plt.subplots(figsize=(max(8, labels.shape[0] // 5), 6))
    im = ax.imshow(
        labels.T,
        aspect="auto",
        interpolation="nearest",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=2,
    )
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Channel")
    ax.set_yticks(range(len(channel_names)))
    ax.set_yticklabels(channel_names, fontsize=6)
    ax.set_title("AutoReject: epoch rejection heatmap\n(0=good, 1=interpolated, 2=bad)")
    plt.colorbar(im, ax=ax, label="Rejection label")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
