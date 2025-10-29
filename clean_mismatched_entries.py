#!/usr/bin/env python3
"""
Remove mismatched metadata entries before rerunning Stage 1.

This script:
  * Scans pipeline_state.json and papers/*.json using the same
    metadata_matches_citation logic as the main pipeline.
  * Deletes any entry whose stored metadata does not match the citation
    (or whose metadata file is missing).
  * Cleans up corresponding records in principles.json, papers/, pdfs/, texts/.
  * Writes the trimmed pipeline_state.json back to disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from build_semantic_scholar_archive import (
    PAPERS_DIR,
    PDF_DIR,
    PIPELINE_STATE_PATH,
    TEXT_DIR,
    metadata_matches_citation,
    parse_citation,
)

PRINCIPLES_PATH = Path("Dataset/papers_info/principles.json")
RESPONSES_PATH = Path("Dataset/gemini_references/responses_parsed.json")


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False,
                    indent=2), encoding="utf-8")


def remove_files_for_key(paper_key: str) -> None:
    targets = [
        PAPERS_DIR / f"{paper_key}.json",
        PDF_DIR / f"{paper_key}.pdf",
        TEXT_DIR / f"{paper_key}.txt",
    ]
    for path in targets:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


def main() -> None:
    pipeline_state: Dict[str, Dict] = load_json(PIPELINE_STATE_PATH, {})
    principles: Dict[str, List[Dict]] = load_json(PRINCIPLES_PATH, {})
    responses: Dict[str, List[str]] = load_json(RESPONSES_PATH, {})

    # (state_key, reason, principle, index)
    removed_entries: List[Tuple[str, str, str, int]] = []
    removed_paper_keys: List[str] = []
    # (principle, citation_index)
    removed_citations: List[Tuple[str, int]] = []

    REMOVE_RESPONSE_REASONS = {"missing_citation_info"}

    for state_key, record in list(pipeline_state.items()):
        citation = record.get("citation")
        principle = record.get("principle")
        citation_index = record.get("citation_index")

        principle_str = principle or "<unknown>"
        index_int = int(citation_index) if citation_index is not None else -1

        reason = None
        should_remove_response = False

        if not citation or principle is None or citation_index is None:
            reason = "missing_citation_info"
            should_remove_response = True
        else:
            info = parse_citation(principle, citation, index_int)

            paper_key = record.get("paper_key") or (
                state_key if not state_key.startswith("missing_") else None
            )
            if not paper_key:
                reason = "no_paper_key"
            else:
                paper_path = PAPERS_DIR / f"{paper_key}.json"
                if not paper_path.exists():
                    reason = "metadata_file_missing"
                else:
                    try:
                        metadata = json.loads(
                            paper_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        reason = "metadata_file_corrupt"
                    else:
                        if not metadata_matches_citation(metadata, info):
                            reason = "metadata_mismatch"

        if reason is None:
            continue

        removed_entries.append((state_key, reason, principle_str, index_int))
        pipeline_state.pop(state_key, None)

        paper_key = record.get("paper_key") or (
            state_key if not state_key.startswith("missing_") else None
        )
        if paper_key:
            removed_paper_keys.append(paper_key)

        if reason in REMOVE_RESPONSE_REASONS:
            should_remove_response = True

        if should_remove_response and principle is not None and citation_index is not None:
            removed_citations.append((principle, index_int))

    # Deduplicate paper keys before deleting files
    for paper_key in sorted(set(removed_paper_keys)):
        remove_files_for_key(paper_key)

    # Update principles.json by removing any entry with removed paper_key
    if principles and removed_paper_keys:
        key_set = set(removed_paper_keys)
        for principle, entries in list(principles.items()):
            new_entries = [
                entry
                for entry in entries
                if entry.get("paper_key") not in key_set
            ]
            if new_entries:
                principles[principle] = new_entries
            else:
                principles.pop(principle, None)

    # Update responses_parsed.json by removing affected citations
    if responses and removed_citations:
        grouped_indices: Dict[str, set] = {}
        for principle, index in removed_citations:
            if principle is None or index < 1:
                continue
            grouped_indices.setdefault(principle, set()).add(index)
        for principle, indices in grouped_indices.items():
            entries = responses.get(principle)
            if not entries:
                continue
            responses[principle] = [
                citation_text
                for pos, citation_text in enumerate(entries, start=1)
                if pos not in indices
            ]
        # Drop empty principles
        for principle in list(responses.keys()):
            if not responses[principle]:
                responses.pop(principle)

    write_json(PIPELINE_STATE_PATH, pipeline_state)
    if principles:
        write_json(PRINCIPLES_PATH, principles)
    if responses:
        write_json(RESPONSES_PATH, responses)

    print(f"Removed {len(removed_entries)} mismatched entries.")
    for state_key, reason, principle, index in removed_entries[:10]:
        print(
            f"  - {state_key} (principle={principle}, citation_index={index}): {reason}")
    if len(removed_entries) > 10:
        print(f"  ... and {len(removed_entries) - 10} more.")


if __name__ == "__main__":
    main()
