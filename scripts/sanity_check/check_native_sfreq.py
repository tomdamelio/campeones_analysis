"""Quick diagnostic: report the NATIVE sampling rate declared in each XDF
header (no data load) for a list of subjects.

Uses pyxdf.resolve_streams, which only parses stream headers, so it is fast
and safe to run on many files. For each EEG stream it prints the nominal_srate
as written by the acquisition software. This answers whether sub-27's
250 Hz in the raw BIDS is the true native rate or an artifact of an old run
(read_xdf.py never resamples; it preserves the native rate).

Run:
    micromamba run -n campeones python scripts/sanity_check/check_native_sfreq.py
    micromamba run -n campeones python scripts/sanity_check/check_native_sfreq.py --subjects 19 27 28 30 31
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyxdf


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=["19", "27", "28", "30", "31"])
    parser.add_argument("--session", default="VR")
    args = parser.parse_args()

    xdf_root = project_root() / "data" / "sourcedata" / "xdf"

    print(f"{'file':<58} {'eeg_stream':<18} {'nominal_srate':>14}")
    print("-" * 92)

    for sub in args.subjects:
        physio_dir = xdf_root / f"sub-{sub}" / f"ses-{args.session}" / "physio"
        if not physio_dir.is_dir():
            print(f"[WARN] no physio dir for sub-{sub}: {physio_dir}")
            continue
        for xdf in sorted(physio_dir.glob(f"sub-{sub}_*_eeg.xdf")):
            try:
                streams = pyxdf.resolve_streams(str(xdf))
            except Exception as exc:  # noqa: BLE001
                print(f"{xdf.name:<58} ERROR resolve_streams: {exc}")
                continue
            eeg = [s for s in streams if str(s.get("type", "")).upper() == "EEG"]
            if not eeg:
                print(f"{xdf.name:<58} {'(no EEG stream)':<18}")
                continue
            for s in eeg:
                name = str(s.get("name", "?"))[:18]
                srate = s.get("nominal_srate", "n/a")
                print(f"{xdf.name:<58} {name:<18} {str(srate):>14}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
