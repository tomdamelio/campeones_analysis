"""QA analysis of batch preprocessing outputs.

Reads the preprocessing JSON log and generates a QA report with per-run
metrics and cross-subject aggregations.

Usage:
    micromamba run -n campeones python -m scripts.preprocessing.qa_batch_analysis \
        --subjects 19 23 24 30 33
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument(
        "--batch-log",
        help="Path to a specific batch_*.json log (default: latest in batch_logs/)",
    )
    parser.add_argument(
        "--output",
        help="Path to write the QA report (default: batch_logs/qa_report_<stamp>.md)",
    )
    args = parser.parse_args()

    root = project_root()
    log_dir = root / "data" / "derivatives" / "campeones_preproc" / "batch_logs"
    preproc_log_path = (
        root
        / "data"
        / "derivatives"
        / "campeones_preproc"
        / "logs_preprocessing_details_all_subjects_eeg.json"
    )

    # --- Load batch summary ---
    if args.batch_log:
        batch_path = Path(args.batch_log)
    else:
        batch_logs = sorted(log_dir.glob("batch_*.json"))
        if not batch_logs:
            print("[ERROR] No batch_*.json found in", log_dir)
            return 2
        batch_path = batch_logs[-1]

    with open(batch_path, encoding="utf-8") as f:
        batch = json.load(f)

    print(f"Batch log : {batch_path.name}")
    print(f"Started   : {batch.get('started_at', '?')}")
    print(f"Finished  : {batch.get('finished_at', '?')}")
    print(f"Successes : {len(batch.get('successes', []))}")
    print(f"Failures  : {len(batch.get('failures', []))}")
    print()

    # --- Load per-run preprocessing details ---
    with open(preproc_log_path, encoding="utf-8") as f:
        all_data = json.load(f)

    # JSON structure: {subject: {session: {task_run_key: {details...}}}}
    subs_wanted = set(args.subjects)
    entries: list[dict] = []
    for sub, sessions in all_data.items():
        if sub not in subs_wanted:
            continue
        if not isinstance(sessions, dict):
            continue
        for ses, runs in sessions.items():
            if not isinstance(runs, dict):
                continue
            for run_key, details in runs.items():
                if not isinstance(details, dict):
                    continue
                details["_subject"] = sub
                details["_session"] = ses
                details["_run_key"] = run_key
                entries.append(details)

    entries.sort(key=lambda e: (e.get("_subject", ""), e.get("_run_key", "")))
    print(f"Preprocessing log entries for requested subjects: {len(entries)}")
    print()

    # --- Per-run metrics table ---
    lines: list[str] = []
    lines.append("# QA Report — Batch Preprocessing")
    lines.append(f"")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Batch log: `{batch_path.name}`")
    lines.append(f"Subjects: {', '.join(sorted(subs_wanted))}")
    lines.append(f"Successes: {len(batch.get('successes', []))} / {batch.get('total', '?')}")
    lines.append(f"Failures: {len(batch.get('failures', []))}")
    lines.append("")

    # Failures detail
    if batch.get("failures"):
        lines.append("## Failed Runs")
        lines.append("")
        for fail in batch["failures"]:
            lines.append(f"### {fail['run']}")
            lines.append(f"- Elapsed: {fail.get('elapsed_sec', '?')}s")
            stderr = fail.get("stderr_tail", "")
            if len(stderr) > 500:
                stderr = stderr[-500:]
            lines.append(f"- Error tail:\n```\n{stderr}\n```")
            lines.append("")

    # Per-run table
    lines.append("## Per-Run Metrics")
    lines.append("")
    lines.append(
        "| Run | sfreq | Bad Ch | Bad Ch Names | Excl | Veto | Pattern Cand | ICLabel Cand | Jitter max (ms) | Jitter mean (ms) | Two-copy | ICA LP |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )

    # Aggregation accumulators
    all_bads_count: list[int] = []
    all_excl_count: list[int] = []
    all_veto_count: list[int] = []
    all_jitter_max: list[float] = []
    all_sfreq: list[float] = []
    bads_by_criterion: defaultdict[str, int] = defaultdict(int)
    bads_by_channel: defaultdict[str, int] = defaultdict(int)
    per_subject_excl: defaultdict[str, list[int]] = defaultdict(list)
    per_subject_bads: defaultdict[str, list[int]] = defaultdict(list)

    for e in entries:
        sub = e.get("_subject", "?")
        run_key = e.get("_run_key", "?")
        label = f"sub-{sub} {run_key}"

        sfreq = e.get("event_onset_sfreq", "?")
        bads = e.get("bad_channels", [])
        excl = e.get("ica_components_excluded", [])
        n_ica = len(excl) + 15  # approximate; overwrite below if detail available
        excl_detail = e.get("ica_exclusions_detail", [])
        if excl_detail:
            all_comp_nums = set()
            for d in excl_detail:
                comp_str = d.get("component", "")
                try:
                    all_comp_nums.add(int(comp_str.replace("ICA", "")))
                except ValueError:
                    pass
        veto = e.get("ica_exclusion_vetoed_by_brain_floor", [])
        pat_cand = e.get("ica_exclusion_pattern_candidates", [])
        icl_cand = e.get("ica_exclusion_iclabel_candidates", [])
        jmax = e.get("event_onset_jitter_max_ms", "?")
        jmean = e.get("event_onset_jitter_mean_ms", "?")
        tcp = e.get("two_copy_pattern", "?")
        lpass = e.get("ica_fit_lpass", "?")

        bads_str = ", ".join(bads) if bads else "none"
        n_pat = len(pat_cand) if isinstance(pat_cand, list) else pat_cand
        n_icl = len(icl_cand) if isinstance(icl_cand, list) else icl_cand
        jmax_str = f"{jmax:.3f}" if isinstance(jmax, (int, float)) else str(jmax)
        jmean_str = f"{jmean:.3f}" if isinstance(jmean, (int, float)) else str(jmean)
        sfreq_str = f"{sfreq:.1f}" if isinstance(sfreq, (int, float)) else str(sfreq)
        lines.append(
            f"| {label} | {sfreq_str} | {len(bads)} | {bads_str} | {len(excl)} | {len(veto)} | {n_pat} | {n_icl} | {jmax_str} | {jmean_str} | {tcp} | {lpass} |"
        )

        # Accumulate
        all_bads_count.append(len(bads))
        all_excl_count.append(len(excl))
        all_veto_count.append(len(veto))
        if isinstance(sfreq, (int, float)):
            all_sfreq.append(float(sfreq))
        if isinstance(jmax, (int, float)):
            all_jitter_max.append(float(jmax))
        per_subject_excl[sub].extend([len(excl)])
        per_subject_bads[sub].extend([len(bads)])
        for b in bads:
            bads_by_channel[b] += 1

        # Per-criterion bad channels
        by_crit = e.get("bad_channels_by_criterion", {})
        for crit, ch_list in by_crit.items():
            if isinstance(ch_list, list):
                bads_by_criterion[crit] += len(ch_list)

    lines.append("")

    # --- Aggregated statistics ---
    lines.append("## Aggregate Statistics")
    lines.append("")

    if all_sfreq:
        lines.append(f"- **sfreq range**: {min(all_sfreq):.1f} – {max(all_sfreq):.1f} Hz")
    if all_bads_count:
        lines.append(
            f"- **Bad channels per run**: mean={statistics.mean(all_bads_count):.1f}, "
            f"median={statistics.median(all_bads_count):.1f}, "
            f"range=[{min(all_bads_count)}, {max(all_bads_count)}]"
        )
    if all_excl_count:
        lines.append(
            f"- **ICA components excluded per run**: mean={statistics.mean(all_excl_count):.1f}, "
            f"median={statistics.median(all_excl_count):.1f}, "
            f"range=[{min(all_excl_count)}, {max(all_excl_count)}]"
        )
    if all_veto_count:
        lines.append(
            f"- **Brain-floor vetos per run**: mean={statistics.mean(all_veto_count):.1f}, "
            f"median={statistics.median(all_veto_count):.1f}, "
            f"range=[{min(all_veto_count)}, {max(all_veto_count)}]"
        )
    if all_jitter_max:
        lines.append(
            f"- **Max jitter (ms)**: mean={statistics.mean(all_jitter_max):.3f}, "
            f"max={max(all_jitter_max):.3f}"
        )
    lines.append("")

    # --- PyPREP criteria breakdown ---
    lines.append("## PyPREP Bad Channel Criteria Breakdown (R-17)")
    lines.append("")
    if bads_by_criterion:
        lines.append("| Criterion | Total flagged channels (across all runs) |")
        lines.append("|---|---|")
        for crit in sorted(bads_by_criterion):
            lines.append(f"| {crit} | {bads_by_criterion[crit]} |")
    else:
        lines.append("No per-criterion data found in logs.")
    lines.append("")

    # --- Most frequently bad channels ---
    lines.append("## Most Frequently Bad Channels")
    lines.append("")
    if bads_by_channel:
        lines.append("| Channel | Times marked bad (across all runs) |")
        lines.append("|---|---|")
        for ch, count in sorted(bads_by_channel.items(), key=lambda x: -x[1]):
            lines.append(f"| {ch} | {count} |")
    else:
        lines.append("No bad channels found.")
    lines.append("")

    # --- Per-subject summary ---
    lines.append("## Per-Subject Summary")
    lines.append("")
    lines.append("| Subject | Runs OK | Mean bads | Mean ICA excl | Mean vetos |")
    lines.append("|---|---|---|---|---|")
    for sub in sorted(subs_wanted):
        n_runs = len(per_subject_excl.get(sub, []))
        if n_runs == 0:
            lines.append(f"| sub-{sub} | 0 | — | — | — |")
            continue
        mb = statistics.mean(per_subject_bads[sub])
        me = statistics.mean(per_subject_excl[sub])
        # Compute mean vetos per subject
        sub_entries = [
            e for e in entries if e.get("_subject") == sub
        ]
        vetos = [
            len(e.get("ica_exclusion_vetoed_by_brain_floor", []))
            for e in sub_entries
        ]
        mv = statistics.mean(vetos) if vetos else 0
        lines.append(f"| sub-{sub} | {n_runs} | {mb:.1f} | {me:.1f} | {mv:.1f} |")
    lines.append("")

    # --- Outlier flags ---
    lines.append("## Outlier Flags")
    lines.append("")
    outliers_found = False

    # High bad channel count
    for e in entries:
        bads = e.get("bad_channels", [])
        if len(bads) >= 5:
            label = f"sub-{e.get('_subject')} {e.get('_run_key')}"
            lines.append(f"- **HIGH BAD CHANNELS** ({len(bads)}): {label} — {bads}")
            outliers_found = True

    # High exclusion count (>20 components)
    for e in entries:
        excl = e.get("ica_components_excluded", [])
        if len(excl) > 20:
            label = f"sub-{e.get('_subject')} {e.get('_run_key')}"
            lines.append(f"- **HIGH ICA EXCLUSION** ({len(excl)} components): {label}")
            outliers_found = True

    # Max jitter > 1.0 ms
    for e in entries:
        jmax = e.get("event_onset_jitter_max_ms", 0)
        if isinstance(jmax, (int, float)) and jmax > 1.0:
            label = f"sub-{e.get('_subject')} {e.get('_run_key')}"
            lines.append(f"- **HIGH JITTER** (max={jmax:.3f} ms): {label}")
            outliers_found = True

    # sfreq anomaly
    for e in entries:
        sf = e.get("event_onset_sfreq", 500)
        if isinstance(sf, (int, float)) and sf < 400:
            label = f"sub-{e.get('_subject')} {e.get('_run_key')}"
            lines.append(f"- **LOW SFREQ** ({sf} Hz): {label}")
            outliers_found = True

    if not outliers_found:
        lines.append("No outliers detected.")
    lines.append("")

    # --- R-18 picard-o check ---
    lines.append("## R-18 Picard-o Verification")
    lines.append("")
    picard_o_runs = 0
    picard_default_runs = 0
    for e in entries:
        lpass = e.get("ica_fit_lpass")
        tcp = e.get("two_copy_pattern", "")
        if lpass == 100.0 or lpass == 100:
            picard_o_runs += 1
        else:
            picard_default_runs += 1
    lines.append(
        f"- Runs with ICA fit LP=100 Hz (R-12 wide-band): {picard_o_runs}/{len(entries)}"
    )
    lines.append(
        "- R-18 (`fit_params=dict(ortho=False, extended=True)`) was applied at script level."
    )
    lines.append(
        "- To verify the ICLabel warning was eliminated, grep the preprocessing stdout for "
        "'picard' + 'extended infomax' — if no matches, R-18 is effective."
    )
    lines.append("")

    # --- Write report ---
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = log_dir / f"qa_report_{stamp}.md"

    report_text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("=" * 60)
    print(report_text)
    print("=" * 60)
    print(f"\nQA report written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
