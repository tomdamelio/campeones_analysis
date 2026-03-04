"""Compara canales disponibles entre dos sujetos."""
import sys
from pathlib import Path
import glob, re

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from mne_bids import BIDSPath, read_raw_bids

bids_root = repo_root / "data" / "raw"

for sub in ["37", "39"]:
    pattern = f"sub-{sub}/ses-vr/eeg/sub-{sub}_ses-vr_task-02_acq-b_run-*_eeg.vhdr"
    matches = glob.glob(str(bids_root / pattern))
    if not matches:
        print(f"\nsub-{sub}: NO FILE FOUND")
        continue
    run = re.search(r"_run-(\d+)_", Path(matches[0]).name).group(1)
    bp = BIDSPath(subject=sub, session="vr", task="02", run=run,
                  acquisition="b", datatype="eeg", root=bids_root, extension=".vhdr")
    raw = read_raw_bids(bp, verbose=False)
    print(f"\n=== sub-{sub} (run={run}) - {len(raw.ch_names)} canales ===")
    for ch in raw.ch_names:
        print(f"  {ch} ({raw.get_channel_types([ch])[0]})")
