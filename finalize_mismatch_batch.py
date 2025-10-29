#!/usr/bin/env python3
"""
Merge the repaired pipeline subset back into the full dataset.

Usage example:
    python finalize_mismatch_batch.py --batch-dir Dataset/repair_batches/batch_20240712_101530

Assumes:
 - prepare_mismatch_batch.py created the batch directory with good/repair copies.
 - build_semantic_scholar_archive.py has been run against the repair subset (i.e., the current
   Dataset/papers_info/logs/pipeline_state.json contains only repaired entries).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

from build_semantic_scholar_archive import (
    PIPELINE_STATE_PATH,
    RESPONSES_PATH,
    build_principle_entries_from_state,
    update_principles_index,
)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge repaired mismatch batch back into the main pipeline."
    )
    parser.add_argument(
        "--batch-dir",
        required=True,
        type=Path,
        help="Batch directory created by prepare_mismatch_batch.py.",
    )
    parser.add_argument(
        "--updated-pipeline",
        type=Path,
        default=PIPELINE_STATE_PATH,
        help="Path to the recently repaired pipeline_state.json (default: working file).",
    )
    parser.add_argument(
        "--updated-responses",
        type=Path,
        default=RESPONSES_PATH,
        help="Path to the repaired responses_parsed.json (default: working file).",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backups of current full pipeline_state.json and responses_parsed.json before overwriting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_dir = args.batch_dir

    good_state_path = batch_dir / "good_pipeline_state.json"
    good_responses_path = batch_dir / "good_responses_parsed.json"
    repaired_state = load_json(args.updated_pipeline)
    repaired_responses = load_json(args.updated_responses)

    good_state = load_json(good_state_path)
    good_responses = load_json(good_responses_path)

    merged_state = dict(good_state)
    merged_state.update(repaired_state)

    merged_responses = dict(good_responses)
    for principle, citations in repaired_responses.items():
        existing = merged_responses.get(principle, [])
        merged_responses[principle] = existing + citations

    if args.backup:
        shutil.copy2(PIPELINE_STATE_PATH, PIPELINE_STATE_PATH.with_suffix(".json.bak"))
        shutil.copy2(RESPONSES_PATH, RESPONSES_PATH.with_suffix(".json.bak"))

    write_json(PIPELINE_STATE_PATH, merged_state)
    write_json(RESPONSES_PATH, merged_responses)

    entries = build_principle_entries_from_state(merged_state)
    update_principles_index(entries)

    write_json(batch_dir / "final_merged_pipeline_state.json", merged_state)
    write_json(batch_dir / "final_merged_responses_parsed.json", merged_responses)

    print("Merged repaired subset back into main dataset.")
    print(f"Updated pipeline_state.json and responses_parsed.json; principles.json rebuilt.")


if __name__ == "__main__":
    main()
