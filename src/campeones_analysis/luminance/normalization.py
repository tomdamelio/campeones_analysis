"""Z-score normalization of luminance targets per video group.

Normalizes the target variable within each video segment so that
cross-video scale differences (e.g., video 3 mean=86.5 vs video 12
mean=111.0) do not bias the regression model.  Applied *before* the
train/test split so that statistics are computed per video independently.

Public API:
    zscore_per_video – pure function, no I/O
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def zscore_per_video(
    epoch_entries: list[dict],
    video_key: str = "video_identifier",
    target_key: str = "y",
) -> list[dict]:
    """Normalize target values to zero mean, unit variance within each video group.

    Args:
        epoch_entries: List of epoch dicts, each containing at minimum
            the keys specified by *video_key* and *target_key*.
        video_key: Key used to group epochs by video.
        target_key: Key containing the luminance target value.

    Returns:
        New list of epoch dicts with *target_key* replaced by the z-scored
        value.  If a video group has std == 0, all targets in that group
        are set to 0.0.
    """
    if not epoch_entries:
        return []

    # Group indices by video
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, entry in enumerate(epoch_entries):
        groups[entry[video_key]].append(idx)

    # Deep-copy so the original list is never mutated
    normalized: list[dict] = [copy.copy(entry) for entry in epoch_entries]

    for video_id, indices in groups.items():
        values = [epoch_entries[i][target_key] for i in indices]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance**0.5

        if std == 0.0:
            logger.warning(
                "Video group '%s' has zero standard deviation; "
                "setting all normalized targets to 0.0.",
                video_id,
            )
            for i in indices:
                normalized[i][target_key] = 0.0
        else:
            for i in indices:
                normalized[i][target_key] = (
                    epoch_entries[i][target_key] - mean
                ) / std

    return normalized
