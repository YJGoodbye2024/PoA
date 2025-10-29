#!/usr/bin/env python3
"""
Prepare a retry batch for Stage2 tasks (PDF download / deep search).

Example:
    python prepare_retry_batch.py --output-dir Dataset/retry_batch \
        --statuses pdf_download_failed deep_search_failed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from build_semantic_scholar_archive import PIPELINE_STATE_PATH, RESPONSES_PATH


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_response_subset(
    responses: Dict[str, List[str]],
    selected: List[Tuple[str, int]],
) -> Dict[str, List[str]]:
    subset: Dict[str, List[str]] = {}
    for principle, citation_index in selected:
        citations = responses.get(principle)
        if not citations:
            continue
        if 1 <= citation_index <= len(citations):
            subset.setdefault(principle, []).append(citations[citation_index - 1])
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect pipeline entries that need Stage2 retry and prepare a standalone batch."
    )
    parser.add_argument(
        "--pipeline-state",
        type=Path,
        default=PIPELINE_STATE_PATH,
        help=f"Path to pipeline_state.json (default: {PIPELINE_STATE_PATH}).",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=RESPONSES_PATH,
        help=f"Path to responses_parsed.json (default: {RESPONSES_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the retry batch files.",
    )
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=["pdf_download_failed", "deep_search_failed"],
        help="Pipeline statuses to include in the retry batch.",
    )
    parser.add_argument(
        "--require-pdf",
        action="store_true",
        help="Only include entries that still have open access PDF links.",
    )

    args = parser.parse_args()

    state = load_json(args.pipeline_state)
    responses = load_json(args.responses)

    selected_keys: Dict[str, Dict] = {}
    response_refs: List[Tuple[str, int]] = []

    for key, record in state.items():
        status = record.get("status")
        if status not in args.statuses:
            continue
        if args.require_pdf and not record.get("pdf_available"):
            continue
        principle = record.get("principle")
        citation_index = record.get("citation_index")
        if not principle or citation_index is None:
            continue
        selected_keys[key] = record
        response_refs.append((principle, int(citation_index)))

    if not selected_keys:
        print("No entries matched the specified criteria; nothing to prepare.")
        return

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "pipeline_state.json", selected_keys)

    subset_responses = build_response_subset(responses, response_refs)
    write_json(output_dir / "responses_parsed.json", subset_responses)

    tasks_path = output_dir / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8") as handle:
        for key, record in selected_keys.items():
            payload = {
                "paper_key": key if not key.startswith("missing_") else record.get("paper_key"),
                "principle": record.get("principle"),
                "citation_index": record.get("citation_index"),
                "citation": record.get("citation"),
                "status": record.get("status"),
                "pdf_url": record.get("pdf_url"),
                "needs_pdf_download": record.get("needs_pdf_download"),
                "needs_deep_search": record.get("needs_deep_search"),
                "pdf_available": record.get("pdf_available"),
                "pdf_saved": record.get("pdf_saved"),
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    principles = sorted(subset_responses.keys())
    (output_dir / "principles.txt").write_text(
        "\n".join(principles),
        encoding="utf-8",
    )

    manifest = {
        "split_id": output_dir.name,
        "task_count": len(selected_keys),
        "principle_count": len(principles),
        "statuses": args.statuses,
        "require_pdf": args.require_pdf,
        "pipeline_state_file": "pipeline_state.json",
        "responses_file": "responses_parsed.json",
        "tasks_file": "tasks.jsonl",
        "principles_file": "principles.txt",
    }
    write_json(output_dir / "manifest.json", manifest)

    print(
        json.dumps(
            {
                "tasks": len(selected_keys),
                "principles": len(principles),
                "output": output_dir.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
