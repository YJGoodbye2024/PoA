#!/usr/bin/env python3
"""
Split responses_parsed.json into multiple subsets for pre-Stage1 processing.

Example:
    python split_responses.py --output-dir Dataset/task_splits --splits 3 --archive
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

RESPONSES_PATH_DEFAULT = Path("Dataset/gemini_references/responses_parsed.json")


def load_responses(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"responses_parsed.json not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc


def split_round_robin(entries: List[Tuple[str, int, str]], parts: int) -> List[List[Tuple[str, int, str]]]:
    buckets: List[List[Tuple[str, int, str]]] = [[] for _ in range(parts)]
    for idx, entry in enumerate(entries):
        buckets[idx % parts].append(entry)
    return buckets


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_split(
    entries: List[Tuple[str, int, str]],
    split_dir: Path,
) -> Dict[str, int]:
    split_dir.mkdir(parents=True, exist_ok=True)

    subset_responses: Dict[str, List[str]] = {}
    principles_set = set()

    tasks_path = split_dir / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8") as handle:
        for principle, citation_index, citation in entries:
            principles_set.add(principle)
            subset_responses.setdefault(principle, []).append(citation)
            payload = {
                "principle": principle,
                "citation_index": citation_index,
                "citation": citation,
                "status": "pending_stage1",
                "needs_pdf_download": False,
                "needs_deep_search": False,
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    write_json(split_dir / "responses_parsed.json", subset_responses)
    write_json(split_dir / "pipeline_state.json", {})

    (split_dir / "principles.txt").write_text(
        "\n".join(sorted(principles_set)),
        encoding="utf-8",
    )

    manifest = {
        "split_id": split_dir.name,
        "split_path": split_dir.as_posix(),
        "task_count": len(entries),
        "principle_count": len(principles_set),
        "pipeline_state_file": "pipeline_state.json",
        "responses_file": "responses_parsed.json",
        "tasks_file": "tasks.jsonl",
        "principles_file": "principles.txt",
        "mode": "responses_only",
    }
    write_json(split_dir / "manifest.json", manifest)

    return {
        "tasks": len(entries),
        "principles": len(principles_set),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split responses_parsed.json into multiple subsets for Stage1 processing."
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=RESPONSES_PATH_DEFAULT,
        help=f"Path to responses_parsed.json (default: {RESPONSES_PATH_DEFAULT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where split outputs will be written.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=3,
        help="Number of partitions to produce (default: 3).",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Create .tar.gz archive for each split directory.",
    )
    args = parser.parse_args()

    if args.splits < 1:
        raise ValueError("Number of splits must be at least 1.")

    responses = load_responses(args.responses)
    entries: List[Tuple[str, int, str]] = []
    for principle, citations in responses.items():
        for idx, citation in enumerate(citations, start=1):
            entries.append((principle, idx, citation))

    if not entries:
        print("No citations found; nothing to split.")
        return

    entries.sort()
    buckets = split_round_robin(entries, args.splits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, int]] = {}

    for idx, bucket in enumerate(buckets, start=1):
        split_dir = args.output_dir / f"split_{idx}"
        counts = write_split(bucket, split_dir)
        if args.archive:
            archive_path = args.output_dir / f"{split_dir.name}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(split_dir, arcname=split_dir.name)
            counts["archive"] = str(archive_path)
        summary[f"split_{idx}"] = counts

    summary["totals"] = {
        "pending_tasks": len(entries),
        "splits": args.splits,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
