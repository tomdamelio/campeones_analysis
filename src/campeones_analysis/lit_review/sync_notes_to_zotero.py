"""Sync `notes_*.md` files into Zotero as child-notes.

For each markdown file in a directory, parses identifiers (DOI/PMID/arXiv/
creator+year) from the header, locates the matching item in the user's
Zotero library, and attaches the markdown as a child-note (HTML).

Idempotent: re-runs update existing notes (matched by sentinel HTML
comment), they don't pile up duplicates.

Usage:
    micromamba run -n campeones python -m \\
        src.campeones_analysis.lit_review.sync_notes_to_zotero \\
        --dir research_diary/context/05_03/lit_review/ \\
        --tag-prefix CAMPEONES/lit-review/05_03 \\
        [--dry-run]

Behavior:
    - Files matching `notes_*.md` (default glob) in `--dir`.
    - Header parsed for DOI / PMID / arXiv / first-author / year / title.
    - Resolution order: citekey > DOI > PMID > title > creator+year.
    - If item not found in Zotero -> NO_ITEM (skipped, no creation).
    - If item found -> upsert child-note + apply tags.
    - Re-run safe: existing notes with same slug get updated, not duplicated.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .zotero_client import ZoteroClient, ZoteroWriter

# ----------------------------------------------------------------------------
# Header parsing
# ----------------------------------------------------------------------------

HEADER_SCAN_CHARS = 3000  # only look at the first N chars of each note

_RE_DOI = re.compile(r"\*\*DOI:\*\*\s*\[?(10\.\d{4,9}/[^\s\)\]]+)")
_RE_PMID = re.compile(r"\*\*PMID:\*\*\s*(\d+)")
_RE_ARXIV = re.compile(r"\*\*arXiv:\*\*\s*\[?(\d{4}\.\d{4,5})")
# # Notas profundas: <Lastname>[ & <Lastname>] <Year> — <subtitle>
_RE_TITLE_LINE = re.compile(
    r"^#\s+Notas profundas:\s+"
    r"(?P<lastname>[A-Za-zÀ-ÿ\-]+)"
    r"(?:\s*&\s*[A-Za-zÀ-ÿ\-]+)?"
    r"\s+(?P<year>\d{4})"
    r"(?:\s*[—\-:]\s*(?P<subtitle>[^\n]+))?",
    re.MULTILINE,
)


def parse_header(md_content: str) -> dict:
    """Extract identifiers from the header of a notes_*.md file."""
    head = md_content[:HEADER_SCAN_CHARS]
    out: dict = {}

    if m := _RE_DOI.search(head):
        out["doi"] = m.group(1).rstrip(".,;)")
    if m := _RE_PMID.search(head):
        out["pmid"] = m.group(1)
    if m := _RE_ARXIV.search(head):
        out["arxiv"] = m.group(1)
    if m := _RE_TITLE_LINE.search(head):
        out["lastname"] = m.group("lastname")
        out["year"] = m.group("year")
        if (st := m.group("subtitle")) is not None:
            out["title_substring"] = st.strip()
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.campeones_analysis.lit_review.sync_notes_to_zotero",
        description="Attach notes_*.md files as Zotero child-notes (idempotent).",
    )
    p.add_argument("--dir", required=True,
                   help="Directory containing notes_*.md files.")
    p.add_argument("--tag-prefix", default="CAMPEONES/lit-review",
                   help="Tag prefix applied to each note (suffix=<slug>) + parent item.")
    p.add_argument("--glob", default="notes_*.md",
                   help="Glob pattern for note files (default: notes_*.md).")
    p.add_argument("--dry-run", action="store_true",
                   help="Show resolution + actions without writing to Zotero.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print debug info to stderr.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    notes_dir = Path(args.dir).resolve()
    if not notes_dir.is_dir():
        print(f"error: not a directory: {notes_dir}", file=sys.stderr)
        return 2

    files = sorted(notes_dir.glob(args.glob))
    if not files:
        print(
            f"warning: no files matching {args.glob!r} in {notes_dir}",
            file=sys.stderr,
        )
        return 1

    print(f"Scanning {len(files)} notes file(s) in {notes_dir}")
    print(f"Tag prefix: {args.tag_prefix}")
    if args.dry_run:
        print("DRY-RUN: no writes will be made")
    print()

    client = ZoteroClient()
    writer = None if args.dry_run else ZoteroWriter()

    stats = {"created": 0, "updated": 0, "no_item": 0, "errored": 0}

    for f in files:
        slug = f.stem  # e.g., "notes_hofmann_2021"
        try:
            content = f.read_text(encoding="utf-8")
        except OSError as e:
            stats["errored"] += 1
            print(f"  ERROR   {f.name}  read failed: {e}")
            continue

        h = parse_header(content)
        if args.verbose:
            print(f"  [parse] {f.name}: {h}", file=sys.stderr)

        resolved = client.resolve_paper(
            doi=h.get("doi"),
            pmid=h.get("pmid"),
            title=h.get("title_substring"),
            creator=h.get("lastname"),
            year=h.get("year"),
        )

        if not resolved:
            stats["no_item"] += 1
            tag_hint = f"doi={h.get('doi')} pmid={h.get('pmid')} creator={h.get('lastname')} year={h.get('year')}"
            print(f"  NO_ITEM  {f.name}  ({tag_hint})")
            continue

        tags = [args.tag_prefix, f"{args.tag_prefix}/{slug}"]

        if args.dry_run:
            print(
                f"  WOULD    {f.name}  ->  item {resolved.item_key}  "
                f"({resolved.source})  '{resolved.title[:60]}'  tags={tags}"
            )
            continue

        try:
            action, note_key = writer.upsert_child_note(
                parent_key=resolved.item_key,
                note_md=content,
                slug=slug,
                tags=tags,
            )
            writer.add_tags_to_item(resolved.item_key, [args.tag_prefix])
            stats[action] += 1
            print(
                f"  {action.upper():9s}  {f.name}  ->  note {note_key}  "
                f"on item {resolved.item_key}  ({resolved.source})"
            )
        except Exception as e:
            stats["errored"] += 1
            print(f"  ERROR    {f.name}  ->  {type(e).__name__}: {e}")

    print()
    print("=" * 64)
    print(f"  Created:   {stats['created']}")
    print(f"  Updated:   {stats['updated']}")
    print(f"  No item:   {stats['no_item']}  (paper not in Zotero — add it first)")
    print(f"  Errored:   {stats['errored']}")
    print("=" * 64)

    if stats["errored"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
