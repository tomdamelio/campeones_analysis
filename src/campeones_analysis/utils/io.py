"""I/O utility functions for campeones_analysis.

Includes helpers for generic file operations and BIDS-compliant data handling.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file from disk.

    Args:
        path: Path to the JSON file.

    Returns:
        Dictionary with the loaded data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save a dictionary as a JSON file.

    Args:
        data: Dictionary to save.
        path: Path to the output JSON file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't exist.

    Args:
        path: Directory path to create.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def find_bids_files(
    bids_root: str,
    suffix: str,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    datatype: Optional[str] = None,
) -> List[Path]:
    """Find BIDS files matching a pattern.

    Args:
        bids_root: Root directory of the BIDS dataset.
        suffix: File suffix (e.g., 'eeg', 'beh', 'physio').
        subject: Optional subject label (e.g., '01').
        session: Optional session label.
        datatype: Optional BIDS datatype (e.g., 'eeg', 'beh', 'physio').

    Returns:
        List of matching file paths.
    """
    root = Path(bids_root)
    pattern = f"sub-{subject or '*'}"
    if session:
        pattern += f"/ses-{session}"
    if datatype:
        pattern += f"/{datatype}"
    else:
        pattern += "/*"
    pattern += f"/*_{suffix}.*"
    return list(root.glob(pattern))
