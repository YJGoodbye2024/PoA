#!/usr/bin/env python3
"""
Scan pipeline_state.json and papers metadata to identify mismatched records.

Outputs a JSON report listing every citation whose stored metadata fails
the validation checks (titles/authors/year) used in build_semantic_scholar_archive.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from build_semantic_scholar_archive import (
    PAPERS_DIR,
    PIPELINE_STATE_PATH,
    metadata_matches_citation,
    parse_citation,
)

OUTPUT_PATH = Path("Dataset/papers_info/logs/metadata_mismatches.json")


def load_pipeline_state(path: Path) -> Dict[str, Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(paper_key: str) -> Dict:
    path = PAPERS_DIR / f"{paper_key}.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def identify_mismatches(state: Dict[str, Dict]) -> Tuple[List[Dict], List[str]]:
    mismatches: List[Dict] = []
    missing_metadata: List[str] = []
    for record in state.values():
        paper_key = record.get("paper_key")
        citation_raw = record.get("citation")
        principle = record.get("principle") or "unknown"
        index = int(record.get("citation_index") or 0)
        if not paper_key or not citation_raw:
            continue
        try:
            metadata = load_metadata(paper_key)
        except FileNotFoundError:
            missing_metadata.append(paper_key)
            continue

        info = parse_citation(principle, citation_raw, index)

        if not metadata_matches_citation(metadata, info):
            mismatches.append(
                {
                    "paper_key": paper_key,
                    "principle": principle,
                    "citation": citation_raw,
                    "metadata_title": metadata.get("title"),
                    "metadata_year": metadata.get("year"),
                    "metadata_source": metadata.get("source"),
                    "metadata_doi": metadata.get("doi"),
                    "pdf_url": (metadata.get("openAccessPdf") or {}).get("url"),
                }
            )
    return mismatches, missing_metadata


def main() -> None:
    state = load_pipeline_state(PIPELINE_STATE_PATH)
    mismatches, missing = identify_mismatches(state)
    result = {
        "mismatch_count": len(mismatches),
        "missing_metadata_count": len(missing),
        "missing_metadata_keys": missing,
        "entries": mismatches,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Wrote mismatch report to {OUTPUT_PATH} "
        f"(mismatches={result['mismatch_count']}, missing_metadata={len(missing)})"
    )


if __name__ == "__main__":
    main()
