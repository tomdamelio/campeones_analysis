"""
Configuration for luminance prediction pipeline (scripts 08–12).

Extends the shared EEG channel list from config.py with luminance-specific
parameters: epoch windowing, spectral bands, ROI definition, TDE settings,
ML pipeline hyperparameters, and file path mappings.

EEG channels verified from:
    data/derivatives/campeones_preproc/sub-27/ses-vr/eeg/
    sub-27_ses-vr_task-01_acq-a_run-002_desc-preproc_channels.tsv
"""

from pathlib import Path

from config import EEG_CHANNELS

# ---------------------------------------------------------------------------
# Epoch windowing
# ---------------------------------------------------------------------------
EPOCH_DURATION_S: float = 1.0  # 1 s
EPOCH_OVERLAP_S: float = 0.9  # 900 ms overlap
EPOCH_STEP_S: float = 0.1  # 100 ms step

# ---------------------------------------------------------------------------
# ROI – Posterior / Occipital electrodes
# Validated against EEG_CHANNELS; all 11 channels are present in the montage.
# ---------------------------------------------------------------------------
POSTERIOR_CHANNELS: list[str] = [
    "O1",
    "O2",
    "P3",
    "P4",
    "P7",
    "P8",
    "Pz",
    "CP1",
    "CP2",
    "CP5",
    "CP6",
]

# ---------------------------------------------------------------------------
# Spectral bands (Hz)
# ---------------------------------------------------------------------------
SPECTRAL_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# ---------------------------------------------------------------------------
# Time Delay Embedding
# ---------------------------------------------------------------------------
TDE_WINDOW_HALF: int = 10  # ±10 time-points → 21 total
TDE_PCA_COMPONENTS: int = 50

# ---------------------------------------------------------------------------
# ML pipeline
# ---------------------------------------------------------------------------
PCA_COMPONENTS: int = 100
RIDGE_ALPHA: float = 1.0
RIDGE_ALPHA_GRID: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0]
N_PERMUTATIONS: int = 0
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Paths – all relative to project root
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DERIVATIVES_PATH: Path = (
    PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
)
XDF_PATH: Path = PROJECT_ROOT / "data" / "sourcedata" / "xdf"
STIMULI_PATH: Path = PROJECT_ROOT / "stimuli" / "luminance"
RESULTS_PATH: Path = PROJECT_ROOT / "results" / "modeling" / "luminance"

# ---------------------------------------------------------------------------
# Luminance CSV mapping: video_id → filename
# ---------------------------------------------------------------------------
LUMINANCE_CSV_MAP: dict[int, str] = {
    3: "green_intensity_video_3.csv",
    7: "green_intensity_video_7.csv",
    9: "green_intensity_video_9.csv",
    12: "green_intensity_video_12.csv",
}

# ---------------------------------------------------------------------------
# Experimental videos (excluding video 1 = practice)
# ---------------------------------------------------------------------------
EXPERIMENTAL_VIDEOS: list[int] = [3, 7, 9, 12]

# ---------------------------------------------------------------------------
# Subject / session / runs
# ---------------------------------------------------------------------------
SUBJECT: str = "27"
SESSION: str = "vr"
RUNS_CONFIG: list[dict] = [
    {"id": "002", "acq": "a", "block": "block1", "task": "01"},
    {"id": "003", "acq": "a", "block": "block2", "task": "02"},
    {"id": "004", "acq": "a", "block": "block3", "task": "03"},
    {"id": "006", "acq": "a", "block": "block4", "task": "04"},
    {"id": "007", "acq": "b", "block": "block1", "task": "01"},
    {"id": "009", "acq": "b", "block": "block3", "task": "03"},
    {"id": "010", "acq": "b", "block": "block4", "task": "04"},
]
