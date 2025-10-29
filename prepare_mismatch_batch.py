#!/usr/bin/env python3
"""
Prepare a repair batch containing only the mismatched citations.

The script:
1. Reads Dataset/papers_info/logs/metadata_mismatches.json (produced by identify_metadata_mismatches.py)
2. Splits Dataset/papers_info/logs/pipeline_state.json and Dataset/gemini_references/responses_parsed.json
   into a "good" portion (kept intact) and a "repair" portion (mismatched entries only)
3. Writes both portions into the specified batch directory for later processing
4. (Optional) With --activate, swaps the working pipeline_state.json / responses_parsed.json to the repair subset
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from build_semantic_scholar_archive import (
    PIPELINE_STATE_PATH,
    RESPONSES_PATH,
)

MISMATCH_REPORT_PATH = Path("Dataset/papers_info/logs/metadata_mismatches.json")


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Missing required file: {path}")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_response_subsets(
    responses: Dict[str, List[str]],
    repair_state: Dict[str, Dict],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    # Build mapping: principle -> list of (citation_index, citation)
    repair_by_principle: Dict[str, List[Tuple[int, str]]] = {}
    for record in repair_state.values():
        principle = record.get("principle")
        citation = record.get("citation")
        idx = int(record.get("citation_index") or 0)
        if not principle or not citation:
            continue
        repair_by_principle.setdefault(principle, []).append((idx, citation))

    repair_responses: Dict[str, List[str]] = {}
    good_responses: Dict[str, List[str]] = {}
    for principle, citations in responses.items():
        to_remove = {
            idx for idx, _ in repair_by_principle.get(principle, [])
        }
        if to_remove:
            # rebuild list skipping removed indices (1-based in pipeline_state)
            rebuilt: List[str] = []
            removed: List[str] = []
            for pos, entry in enumerate(citations, start=1):
                if pos in to_remove:
                    removed.append(entry)
                else:
                    rebuilt.append(entry)
            good_responses[principle] = rebuilt
            # sort repair citations by citation_index to maintain order
            ordered = sorted(repair_by_principle[principle], key=lambda item: item[0])
            repair_responses[principle] = [entry for _, entry in ordered]
        else:
            good_responses[principle] = list(citations)

    return repair_responses, good_responses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a mismatch repair batch without disturbing correct data."
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        help="Directory to write the batch (default: Dataset/repair_batches/<timestamp>).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=MISMATCH_REPORT_PATH,
        help=f"Path to mismatch report (default: {MISMATCH_REPORT_PATH}).",
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="After preparing the batch, overwrite pipeline_state.json and responses_parsed.json with the repair subset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = load_json(args.report)
    entries = report.get("entries") or []
    mismatch_keys = {entry.get("paper_key") for entry in entries if entry.get("paper_key")}

    if not mismatch_keys:
        print("No mismatched entries found in the report; nothing to do.")
        return

    pipeline_state = load_json(PIPELINE_STATE_PATH)
    responses = load_json(RESPONSES_PATH)

    repair_state = {
        key: record for key, record in pipeline_state.items() if key in mismatch_keys
    }
    if not repair_state:
        print("Mismatch report keys were not found in pipeline_state.json; aborting.")
        return

    good_state = {
        key: record for key, record in pipeline_state.items() if key not in mismatch_keys
    }

    repair_responses, good_responses = build_response_subsets(responses, repair_state)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = args.batch_dir or Path(f"Dataset/repair_batches/batch_{timestamp}")
    batch_dir.mkdir(parents=True, exist_ok=True)

    write_json(batch_dir / "repair_pipeline_state.json", repair_state)
    write_json(batch_dir / "repair_responses_parsed.json", repair_responses)
    write_json(batch_dir / "good_pipeline_state.json", good_state)
    write_json(batch_dir / "good_responses_parsed.json", good_responses)

    (batch_dir / "paper_keys.txt").write_text(
        "\n".join(sorted(repair_state.keys())), encoding="utf-8"
    )
    write_json(
        batch_dir / "manifest.json",
        {
            "repair_count": len(repair_state),
            "good_count": len(good_state),
            "report": str(args.report),
            "activated": args.activate,
        },
    )

    if args.activate:
        shutil.copy2(PIPELINE_STATE_PATH, batch_dir / "original_pipeline_state.json")
        shutil.copy2(RESPONSES_PATH, batch_dir / "original_responses_parsed.json")
        shutil.copy2(batch_dir / "repair_pipeline_state.json", PIPELINE_STATE_PATH)
        shutil.copy2(batch_dir / "repair_responses_parsed.json", RESPONSES_PATH)
        print("Working pipeline_state.json / responses_parsed.json replaced with repair subset.")

    print(f"Prepared repair batch at {batch_dir}")
    print(
        "Next steps:\n"
        "1. (Optional) Use --activate to swap the working files.\n"
        "2. Run build_semantic_scholar_archive.py on the repair subset.\n"
        "3. Afterwards, run finalize_mismatch_batch.py to merge results."
    )


if __name__ == "__main__":
    main()
