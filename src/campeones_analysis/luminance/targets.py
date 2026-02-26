"""Target computation functions for luminance prediction models.

Pure functions for deriving alternative prediction targets from raw
luminance values: delta luminance (ΔL) and binary change/stability labels.
These targets test whether the brain responds more to luminance transitions
than to steady-state luminance levels.

Requirements: 8.1, 8.2, 9.1
"""

from collections import defaultdict
from copy import deepcopy


def compute_delta_luminance(
    epoch_entries: list[dict],
    video_key: str = "video_identifier",
    target_key: str = "y",
) -> list[dict]:
    """Compute delta luminance target (ΔL = L_i − L_{i-1}) per video segment.

    For each video group, computes the first-difference of the luminance
    target. The first epoch of each video group is discarded because it has
    no preceding epoch to compute a delta from.

    This target tests whether EEG encodes luminance *transitions* rather
    than absolute luminance levels (Requirements 8.1, 8.2).

    Args:
        epoch_entries: List of epoch dicts, each containing at minimum
            ``video_key`` (str) and ``target_key`` (float).
        video_key: Key used to group epochs by video segment.
            Defaults to ``"video_identifier"``.
        target_key: Key containing the raw luminance target value.
            Defaults to ``"y"``.

    Returns:
        New list of epoch dicts with ``target_key`` replaced by the delta
        value (L_i − L_{i-1}). The first epoch of each video group is
        excluded. All other fields are preserved unchanged.
    """
    # Group epoch indices by video, preserving original order
    video_groups: dict[str, list[int]] = defaultdict(list)
    for idx, entry in enumerate(epoch_entries):
        video_groups[entry[video_key]].append(idx)

    delta_entries: list[dict] = []
    for video_id in video_groups:
        indices = video_groups[video_id]
        # Skip first epoch — no previous epoch available
        for position in range(1, len(indices)):
            current_idx = indices[position]
            previous_idx = indices[position - 1]
            new_entry = deepcopy(epoch_entries[current_idx])
            new_entry[target_key] = (
                epoch_entries[current_idx][target_key]
                - epoch_entries[previous_idx][target_key]
            )
            delta_entries.append(new_entry)

    return delta_entries


def compute_change_labels(
    epoch_entries: list[dict],
    threshold: float,
    target_key: str = "y",
) -> list[dict]:
    """Generate binary change/stability labels from delta luminance targets.

    Assigns label 1 (change detected) when the absolute delta luminance
    exceeds the threshold, and label 0 (stable) otherwise. Intended to be
    applied after ``compute_delta_luminance``.

    This target tests whether EEG can discriminate luminance transition
    events from stable periods (Requirement 9.1).

    Args:
        epoch_entries: List of epoch dicts with delta luminance in
            ``target_key``. Typically the output of
            ``compute_delta_luminance``.
        threshold: Absolute delta luminance value above which an epoch is
            labeled as a change event (1). Must be > 0.
        target_key: Key containing the delta luminance target.
            Defaults to ``"y"``.

    Returns:
        New list of epoch dicts with ``target_key`` replaced by binary
        integer labels (1 = change, 0 = stable). All other fields are
        preserved unchanged.
    """
    labeled_entries: list[dict] = []
    for entry in epoch_entries:
        new_entry = deepcopy(entry)
        new_entry[target_key] = int(abs(entry[target_key]) > threshold)
        labeled_entries.append(new_entry)
    return labeled_entries
