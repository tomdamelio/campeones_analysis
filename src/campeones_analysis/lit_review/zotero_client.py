"""Resolve papers from the local Zotero library.

This module is read-only and uses Zotero's Local API (port 23119), which
requires Zotero desktop to be running with "Allow other applications on
this computer to communicate with Zotero" enabled (Edit > Settings >
Advanced > General).

Public API:
    ZoteroClient: stateful client with cached top-items
    ResolvedPaper: dataclass returned by resolve_paper()

CLI usage (preferred for sub-agents via Bash):
    micromamba run -n campeones python -m \
        src.campeones_analysis.lit_review.zotero_client --doi 10.1145/3803808
    # stdout = absolute path to PDF, or exit code != 0 if not found
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from pyzotero import zotero
from pyzotero.zotero_errors import PyZoteroError

BBT_URL = "http://localhost:23119/better-bibtex/json-rpc"


def _detect_zotero_data_dir() -> Path:
    """Locate the Zotero profile data directory."""
    if "ZOTERO_DATA_DIR" in os.environ:
        return Path(os.environ["ZOTERO_DATA_DIR"])
    # Default on Windows / Mac / Linux: ~/Zotero
    return Path.home() / "Zotero"


ZOTERO_DATA_DIR = _detect_zotero_data_dir()


@dataclass
class ResolvedPaper:
    """Outcome of a paper-resolution attempt against the local Zotero library."""

    item_key: str
    citekey: str | None
    doi: str | None
    pmid: str | None
    title: str
    pdf_path: Path | None
    source: str  # "by_citekey" | "by_doi" | "by_pmid" | "by_title"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pdf_path"] = str(self.pdf_path) if self.pdf_path else None
        return d


class ZoteroClient:
    """Read-only client over Zotero Local API. Caches top-level items."""

    def __init__(self, zotero_data_dir: Path = ZOTERO_DATA_DIR) -> None:
        self.zot = zotero.Zotero(library_id="0", library_type="user", local=True)
        self.zotero_data_dir = zotero_data_dir
        self._top_cache: list[dict] | None = None

    # ----- internal helpers ------------------------------------------------

    def _ensure_top_cache(self) -> list[dict]:
        if self._top_cache is None:
            self._top_cache = self.zot.everything(self.zot.top())
        return self._top_cache

    @staticmethod
    def _normalize_doi(doi: str) -> str:
        d = (doi or "").strip().lower()
        d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
        d = re.sub(r"^doi:\s*", "", d)
        return d

    @staticmethod
    def _bbt_call(method: str, params: list, timeout: float = 5.0) -> dict | None:
        """POST to BBT JSON-RPC. Returns parsed JSON or None on any failure."""
        payload = json.dumps(
            {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
        ).encode("utf-8")
        req = urllib.request.Request(
            BBT_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
            return None

    # ----- lookup ----------------------------------------------------------

    def find_by_doi(self, doi: str) -> dict | None:
        """Match by DOI: exact in `DOI`, substring in `url`, substring in `extra`."""
        target = self._normalize_doi(doi)
        if not target:
            return None
        items = self._ensure_top_cache()
        for item in items:
            if self._normalize_doi(item["data"].get("DOI", "")) == target:
                return item
        for item in items:
            if target in (item["data"].get("url", "") or "").lower():
                return item
        for item in items:
            if target in (item["data"].get("extra", "") or "").lower():
                return item
        return None

    def find_by_pmid(self, pmid: str) -> dict | None:
        """Match by PMID in `extra` field (Zotero convention: 'PMID: 12345')."""
        target = str(pmid).strip()
        if not target:
            return None
        pattern = re.compile(rf"PMID[:\s]+{re.escape(target)}\b", re.IGNORECASE)
        for item in self._ensure_top_cache():
            if pattern.search(item["data"].get("extra", "") or ""):
                return item
        return None

    def find_by_citekey(self, citekey: str) -> dict | None:
        """Look up via BBT JSON-RPC; returns None if BBT not available."""
        if not citekey:
            return None
        resp = self._bbt_call("item.search", [citekey])
        if not resp or "result" not in resp:
            return None
        results = resp["result"]
        if not isinstance(results, list):
            return None
        for r in results:
            ck = r.get("citationKey") or r.get("citekey")
            if ck != citekey:
                continue
            item_key = r.get("itemKey") or r.get("key")
            if not item_key:
                continue
            for item in self._ensure_top_cache():
                if item["key"] == item_key:
                    return item
            try:
                return self.zot.item(item_key)
            except PyZoteroError:
                return None
        return None

    def find_by_title_substring(self, title_substring: str) -> dict | None:
        """Last-resort: case-insensitive substring match on `title`."""
        target = (title_substring or "").strip().lower()
        if not target:
            return None
        for item in self._ensure_top_cache():
            if target in (item["data"].get("title", "") or "").lower():
                return item
        return None

    def find_by_creator_year(self, lastname: str, year: int | str) -> dict | None:
        """Match items by first-author lastName + year-in-date.

        Useful when DOI/PMID are absent (e.g., webpage items, arXiv preprints
        without DOI populated).
        """
        target_last = (lastname or "").lower().strip()
        target_year = str(year).strip()
        if not target_last or not target_year:
            return None
        for item in self._ensure_top_cache():
            d = item["data"]
            creators = d.get("creators", []) or []
            if not creators:
                continue
            first = creators[0]
            first_last = (
                first.get("lastName") or first.get("name", "").split()[-1] or ""
            )
            if first_last.lower().strip() != target_last:
                continue
            if target_year not in (d.get("date", "") or ""):
                continue
            return item
        return None

    # ----- attachment / PDF resolution ------------------------------------

    def get_pdf_path(self, item_key: str) -> Path | None:
        """Locate a PDF attachment for the item via Zotero storage on disk."""
        try:
            children = self.zot.children(item_key)
        except PyZoteroError:
            return None

        for child in children:
            d = child["data"]
            if d.get("itemType") != "attachment":
                continue
            if d.get("contentType") != "application/pdf":
                continue
            link_mode = d.get("linkMode", "")
            stored_path = d.get("path", "") or ""
            attachment_key = child["key"]

            # linked_file -> absolute path on disk
            if link_mode == "linked_file" and stored_path:
                p = Path(stored_path)
                if p.is_absolute() and p.exists():
                    return p.resolve()

            # imported_file / imported_url -> Zotero/storage/<key>/<filename>
            if link_mode in ("imported_file", "imported_url"):
                storage_dir = self.zotero_data_dir / "storage" / attachment_key
                if not storage_dir.is_dir():
                    continue
                hinted = None
                if stored_path.startswith("storage:"):
                    hinted = stored_path.replace("storage:", "", 1).strip() or None
                if hinted:
                    candidate = storage_dir / hinted
                    if candidate.exists():
                        return candidate.resolve()
                pdfs = list(storage_dir.glob("*.pdf"))
                if pdfs:
                    return pdfs[0].resolve()
        return None

    def get_citekey(self, item_key: str) -> str | None:
        """Reverse lookup: BBT citekey for a Zotero item. None if BBT missing."""
        resp = self._bbt_call("item.citationkey", [[item_key]])
        if not resp or "result" not in resp:
            return None
        result = resp["result"]
        if isinstance(result, dict):
            return result.get(item_key)
        return None

    # ----- top-level resolver ---------------------------------------------

    def resolve_paper(
        self,
        *,
        doi: str | None = None,
        pmid: str | None = None,
        citekey: str | None = None,
        title: str | None = None,
        creator: str | None = None,
        year: str | int | None = None,
    ) -> ResolvedPaper | None:
        """Try lookups in order: citekey -> doi -> pmid -> title -> creator+year."""
        item: dict | None = None
        source: str | None = None
        if citekey:
            item = self.find_by_citekey(citekey)
            if item:
                source = "by_citekey"
        if not item and doi:
            item = self.find_by_doi(doi)
            if item:
                source = "by_doi"
        if not item and pmid:
            item = self.find_by_pmid(pmid)
            if item:
                source = "by_pmid"
        if not item and title:
            item = self.find_by_title_substring(title)
            if item:
                source = "by_title"
        if not item and creator and year:
            item = self.find_by_creator_year(creator, year)
            if item:
                source = "by_creator_year"

        if not item:
            return None

        item_key = item["key"]
        pdf_path = self.get_pdf_path(item_key)
        ck = self.get_citekey(item_key)

        extra = item["data"].get("extra", "") or ""
        m = re.search(r"PMID[:\s]+(\d+)", extra, re.IGNORECASE)
        item_pmid = m.group(1) if m else None

        return ResolvedPaper(
            item_key=item_key,
            citekey=ck,
            doi=(item["data"].get("DOI") or None),
            pmid=item_pmid,
            title=item["data"].get("title", ""),
            pdf_path=pdf_path,
            source=source or "unknown",
        )


# ---------- Writer (Cloud API) -------------------------------------------


class ZoteroWriter:
    """Cloud API client for writes (creates notes, updates tags).

    Loads credentials from .env at project root (ZOTERO_LIBRARY_ID, _TYPE,
    _API_KEY). Use this for Phase 2: writing notes back to items.

    Local API doesn't support writes (read-only), so this is unavoidable.
    """

    NOTE_SENTINEL_PREFIX = "<!-- claude-sync-notes:"

    def __init__(self) -> None:
        try:
            from dotenv import load_dotenv
        except ImportError as e:
            raise RuntimeError("python-dotenv not installed in env") from e
        load_dotenv()
        try:
            lib_id = os.environ["ZOTERO_LIBRARY_ID"]
            lib_type = os.environ.get("ZOTERO_LIBRARY_TYPE", "user")
            api_key = os.environ["ZOTERO_API_KEY"]
        except KeyError as e:
            raise RuntimeError(
                f"missing env var {e!s}; check .env at project root"
            ) from e
        self.zot = zotero.Zotero(
            library_id=lib_id, library_type=lib_type, api_key=api_key
        )

    @staticmethod
    def _md_to_html(md_text: str) -> str:
        try:
            import markdown
        except ImportError as e:
            raise RuntimeError("markdown lib not installed in env") from e
        return markdown.markdown(
            md_text,
            extensions=["tables", "fenced_code", "attr_list", "sane_lists"],
        )

    def find_child_note_by_slug(self, parent_key: str, slug: str) -> dict | None:
        """Locate a child-note whose HTML carries the sentinel for `slug`. None if absent."""
        try:
            children = self.zot.children(parent_key)
        except PyZoteroError:
            return None
        sentinel = f"{self.NOTE_SENTINEL_PREFIX}{slug} -->"
        for c in children:
            if c["data"].get("itemType") != "note":
                continue
            if sentinel in (c["data"].get("note", "") or ""):
                return c
        return None

    def upsert_child_note(
        self,
        parent_key: str,
        note_md: str,
        slug: str,
        tags: list[str] | None = None,
    ) -> tuple[str, str]:
        """Idempotently attach (create or update) a child-note to parent_key.

        Args:
            parent_key: Zotero item key of the parent paper.
            note_md: Markdown content of the note.
            slug: stable identifier used in the sentinel comment (e.g.,
                  "notes_hofmann_2021"). Re-runs with the same slug update
                  the existing note instead of creating duplicates.
            tags: list of tag strings to apply to the note.

        Returns:
            (action, note_key) where action is "created" or "updated".
        """
        sentinel = f"{self.NOTE_SENTINEL_PREFIX}{slug} -->"
        body_html = self._md_to_html(note_md)
        note_html = f"{sentinel}\n{body_html}"

        existing = self.find_child_note_by_slug(parent_key, slug)
        tag_payload = [{"tag": t} for t in (tags or [])]

        if existing:
            existing["data"]["note"] = note_html
            current = {t.get("tag") for t in existing["data"].get("tags") or []}
            for t in tags or []:
                if t not in current:
                    existing["data"].setdefault("tags", []).append({"tag": t})
            ok = self.zot.update_item(existing)
            if not ok:
                raise RuntimeError(f"update_item returned falsy for {existing['key']}")
            return ("updated", existing["key"])

        payload = {
            "itemType": "note",
            "parentItem": parent_key,
            "note": note_html,
            "tags": tag_payload,
            "relations": {},
        }
        resp = self.zot.create_items([payload])
        if resp.get("failed"):
            raise RuntimeError(f"create_items failed: {resp['failed']}")
        if not resp.get("successful"):
            raise RuntimeError(f"create_items unexpected response: {resp}")
        return ("created", resp["successful"]["0"]["key"])

    def add_tags_to_item(self, item_key: str, tags: list[str]) -> bool:
        """Append tags (idempotently) to an existing item. Returns True if any added."""
        if not tags:
            return False
        try:
            item = self.zot.item(item_key)
        except PyZoteroError as e:
            raise RuntimeError(f"item {item_key} not on Cloud (sync first?): {e}") from e
        current = {t.get("tag") for t in item["data"].get("tags") or []}
        new = [t for t in tags if t not in current]
        if not new:
            return False
        for t in new:
            item["data"].setdefault("tags", []).append({"tag": t})
        ok = self.zot.update_item(item)
        if not ok:
            raise RuntimeError(f"update_item returned falsy for {item_key}")
        return True


# ---------- CLI -----------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.campeones_analysis.lit_review.zotero_client",
        description="Resolve a paper from the local Zotero library and print its PDF path.",
    )
    p.add_argument("--doi", help="DOI (e.g., 10.7554/eLife.64812)")
    p.add_argument("--pmid", help="PubMed ID")
    p.add_argument("--citekey", help="BetterBibTeX citation key")
    p.add_argument("--title", help="Title substring (loose match; last resort)")
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit full resolved metadata as JSON (default: PDF path only).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true", help="Log debug info to stderr."
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not any([args.doi, args.pmid, args.citekey, args.title]):
        print(
            "error: at least one of --doi, --pmid, --citekey, --title is required",
            file=sys.stderr,
        )
        return 2

    if args.verbose:
        print(f"[zotero_client] data dir: {ZOTERO_DATA_DIR}", file=sys.stderr)
        print(
            f"[zotero_client] lookup: doi={args.doi} pmid={args.pmid} "
            f"citekey={args.citekey} title={args.title!r}",
            file=sys.stderr,
        )

    try:
        client = ZoteroClient()
        resolved = client.resolve_paper(
            doi=args.doi, pmid=args.pmid, citekey=args.citekey, title=args.title
        )
    except PyZoteroError as e:
        print(f"error: pyzotero: {e}", file=sys.stderr)
        print(
            "hint: is Zotero desktop running with local API enabled?",
            file=sys.stderr,
        )
        return 2
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    if resolved is None:
        if args.verbose:
            print("[zotero_client] no match in local Zotero", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(resolved.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if resolved.pdf_path is None:
        if args.verbose:
            print(
                f"[zotero_client] item found ({resolved.item_key}) but no PDF",
                file=sys.stderr,
            )
        return 1

    print(resolved.pdf_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
