"""Batch wrapper around scripts/preprocessing/04_preprocessing_eeg.py.

Discovers all raw BIDS runs for a list of subjects, excludes task-practice,
and invokes the preprocessing script with --auto for each tuple. Continues
on error and writes a summary JSON at the end.

Example:
    micromamba run -n campeones python -m scripts.preprocessing.run_batch_preprocessing \\
        --subjects 19 23 24 30 33 --session vr
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RUN_REGEX = re.compile(
    r"^sub-(?P<sub>\d+)_ses-(?P<ses>[a-zA-Z0-9]+)_task-(?P<task>[a-zA-Z0-9]+)"
    r"_acq-(?P<acq>[a-zA-Z0-9]+)_run-(?P<run>\d+)_eeg\.vhdr$"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_runs(subjects: list[str], session: str, raw_root: Path) -> list[dict]:
    """Walk raw BIDS for each subject and return a list of run tuples.

    Each item: {"sub": "19", "ses": "vr", "task": "01", "acq": "a", "run": "002"}
    task-practice is excluded.
    """
    tuples: list[dict] = []
    for sub in subjects:
        eeg_dir = raw_root / f"sub-{sub}" / f"ses-{session}" / "eeg"
        if not eeg_dir.is_dir():
            print(f"[WARN] raw BIDS not found for sub-{sub}: {eeg_dir}", flush=True)
            continue
        for vhdr in sorted(eeg_dir.glob("*.vhdr")):
            m = RUN_REGEX.match(vhdr.name)
            if not m:
                print(f"[WARN] Skipping unrecognized filename: {vhdr.name}", flush=True)
                continue
            if m["task"] == "practice":
                continue
            tuples.append(
                dict(sub=m["sub"], ses=m["ses"], task=m["task"], acq=m["acq"], run=m["run"])
            )
    return tuples


def already_preprocessed(t: dict, deriv_root: Path) -> bool:
    out = (
        deriv_root
        / f"sub-{t['sub']}"
        / f"ses-{t['ses']}"
        / "eeg"
        / f"sub-{t['sub']}_ses-{t['ses']}_task-{t['task']}_acq-{t['acq']}_run-{t['run']}_desc-preproc_eeg.vhdr"
    )
    return out.exists()


def run_one(t: dict, force: bool) -> tuple[bool, str, float]:
    """Invoke 04_preprocessing_eeg.py for a single tuple.

    Returns (success, stderr_tail, elapsed_seconds).
    """
    cmd = [
        "micromamba",
        "run",
        "-n",
        "campeones",
        "python",
        "-m",
        "scripts.preprocessing.04_preprocessing_eeg",
        "--subject",
        t["sub"],
        "--session",
        t["ses"],
        "--task",
        t["task"],
        "--run",
        t["run"],
        "--acquisition",
        t["acq"],
        "--auto",
    ]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root()),
            capture_output=True,
            text=True,
            timeout=60 * 60,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"TIMEOUT after 60 min: {exc}", time.time() - start
    elapsed = time.time() - start
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        return False, tail, elapsed
    return True, "", elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        required=True,
        help="Subject IDs without 'sub-' prefix (e.g. 19 23 24 30 33)",
    )
    parser.add_argument("--session", default="vr", help="Session label (default: vr)")
    parser.add_argument(
        "--force", action="store_true", help="Re-process runs already in derivatives"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered tuples and exit without running preproc",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Run labels to skip, e.g. sub-19_task-04_acq-a_run-005 (aborted/short takes)",
    )
    args = parser.parse_args()

    root = project_root()
    raw_root = root / "data" / "raw"
    deriv_root = root / "data" / "derivatives" / "campeones_preproc"

    print(f"=== Batch preprocessing wrapper ===", flush=True)
    print(f"Subjects : {args.subjects}", flush=True)
    print(f"Session  : {args.session}", flush=True)
    print(f"Force    : {args.force}", flush=True)
    print(f"Raw root : {raw_root}", flush=True)

    tuples = discover_runs(args.subjects, args.session, raw_root)
    if not tuples:
        print("[ERROR] No runs discovered. Check --subjects and raw BIDS paths.", flush=True)
        return 2

    if args.exclude:
        excl = set(args.exclude)
        before = len(tuples)
        tuples = [
            t
            for t in tuples
            if f"sub-{t['sub']}_task-{t['task']}_acq-{t['acq']}_run-{t['run']}" not in excl
        ]
        print(f"Excluded {before - len(tuples)} run(s) via --exclude: {sorted(excl)}", flush=True)

    to_run: list[dict] = []
    skipped: list[dict] = []
    for t in tuples:
        if not args.force and already_preprocessed(t, deriv_root):
            skipped.append(t)
        else:
            to_run.append(t)

    print(f"\nDiscovered {len(tuples)} non-practice runs total.", flush=True)
    print(f"  To run   : {len(to_run)}", flush=True)
    print(f"  Skipped  : {len(skipped)} (already processed; use --force to redo)", flush=True)
    for t in to_run:
        print(
            f"  -> sub-{t['sub']} ses-{t['ses']} task-{t['task']} acq-{t['acq']} run-{t['run']}",
            flush=True,
        )

    if args.dry_run:
        print("\n[dry-run] exiting without running preproc.", flush=True)
        return 0

    summary = {
        "started_at": datetime.now().isoformat(),
        "subjects": args.subjects,
        "session": args.session,
        "force": args.force,
        "total": len(to_run),
        "skipped": [
            f"sub-{t['sub']}_task-{t['task']}_acq-{t['acq']}_run-{t['run']}" for t in skipped
        ],
        "successes": [],
        "failures": [],
    }

    for i, t in enumerate(to_run, start=1):
        label = f"sub-{t['sub']}_task-{t['task']}_acq-{t['acq']}_run-{t['run']}"
        print(f"\n[{i}/{len(to_run)}] RUNNING {label}", flush=True)
        ok, err_tail, elapsed = run_one(t, args.force)
        if ok:
            print(f"[{i}/{len(to_run)}] OK      {label}  ({elapsed:.0f}s)", flush=True)
            summary["successes"].append({"run": label, "elapsed_sec": round(elapsed, 1)})
        else:
            print(
                f"[{i}/{len(to_run)}] FAIL    {label}  ({elapsed:.0f}s)\n---stderr tail---\n{err_tail}\n---",
                flush=True,
            )
            summary["failures"].append(
                {"run": label, "elapsed_sec": round(elapsed, 1), "stderr_tail": err_tail}
            )

    summary["finished_at"] = datetime.now().isoformat()
    log_dir = root / "data" / "derivatives" / "campeones_preproc" / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"batch_{stamp}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== BATCH SUMMARY ===", flush=True)
    print(f"Total attempted: {len(to_run)}", flush=True)
    print(f"Successes      : {len(summary['successes'])}", flush=True)
    print(f"Failures       : {len(summary['failures'])}", flush=True)
    print(f"Log            : {log_path}", flush=True)
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
