"""Sequential driver around src.campeones_analysis.physio.read_xdf.

Runs ``read_xdf --subject X --continue-on-error`` for a list of subjects,
writing a combined log file under data/derivatives/campeones_preproc/batch_logs/
with progress markers the Monitor tool can grep for.

Example:
    micromamba run -n campeones python -m scripts.preprocessing.run_read_xdf_batch \\
        --subjects 19 24 30 33
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = project_root()
    log_dir = root / "data" / "derivatives" / "campeones_preproc" / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"read_xdf_{stamp}.log"

    print(f"PHASE1_START {datetime.now().isoformat()}", flush=True)
    print(f"Subjects : {args.subjects}", flush=True)
    print(f"Log      : {log_path}", flush=True)

    any_fail = False
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"PHASE1_START {datetime.now().isoformat()}\n")
        f.flush()
        for s in args.subjects:
            header = f"=== sub-{s} START ===\n"
            f.write(header)
            f.flush()
            print(header.rstrip(), flush=True)
            cmd = [
                sys.executable,
                "-m",
                "src.campeones_analysis.physio.read_xdf",
                "--subject",
                s,
                "--continue-on-error",
            ]
            if args.force:
                cmd.append("--force")
            proc = subprocess.run(
                cmd,
                cwd=str(root),
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            footer = f"=== sub-{s} END exit={proc.returncode} ===\n"
            f.write(footer)
            f.flush()
            print(footer.rstrip(), flush=True)
            if proc.returncode != 0:
                any_fail = True
        f.write(f"PHASE1_DONE {datetime.now().isoformat()}\n")
    print(f"PHASE1_DONE {datetime.now().isoformat()}", flush=True)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
