#!/usr/bin/env python3
"""
Utility for partitioning pending PDF downloads / deep-search tasks across machines.

Example:
    python split_pipeline_tasks.py --output-dir Dataset/task_splits --splits 3 \
        --include-deep --include-failed-pdf
"""
from __future__ import annotations

import argparse
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

README_TEMPLATE = """\
Split ID: {split_id}
Tasks assigned: {task_count}
Principles included ({principle_count}): {principles}

步骤：
1. 将本目录与主仓库的 `Dataset/papers_info` 一起复制到目标机器，保持目录结构不变。
2. 在目标机器上用此目录中的 `pipeline_state.json` 覆盖 `Dataset/papers_info/logs/pipeline_state.json`，
   用 `responses_parsed.json` 覆盖 `Dataset/gemini_references/responses_parsed.json`。
3. （可选）参考 `principles.txt` 或 `paper_keys.txt` 确认分到的任务范围。
4. 运行 `python build_semantic_scholar_archive.py --pdf-workers 8 --deep-workers 10`（按需调整并发）。
5. 完成后，在目标机器上执行：`python merge_pipeline_results.py prepare --dataset-root Dataset/papers_info --split-dir {split_dir} --output {split_dir}/results`
6. 将 {split_dir}/results 目录打包带回主机，使用 `python merge_pipeline_results.py apply --dataset-root Dataset/papers_info --results {split_dir}/results [...]` 合并成果。
"""


DEFAULT_PIPELINE_STATE = Path("Dataset/papers_info/logs/pipeline_state.json")


@dataclass(frozen=True)
class PendingTask:
    paper_key: str
    record: Dict
    reasons: Sequence[str]
    principle: str
    citation: str
    citation_index: int


def load_pipeline_state(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Pipeline state not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc


def collect_pending_tasks(
    state: Dict[str, Dict],
    include_deep: bool,
    include_failed_pdf: bool,
) -> List[PendingTask]:
    pending: List[PendingTask] = []
    for paper_key, record in state.items():
        reasons: List[str] = []

        needs_pdf = bool(record.get("needs_pdf_download"))
        pdf_available = bool(record.get("pdf_available"))
        status = record.get("status")

        if include_failed_pdf and not needs_pdf:
            if status == "pdf_download_failed" and pdf_available:
                needs_pdf = True

        if needs_pdf and not pdf_available:
            needs_pdf = False

        needs_deep = include_deep and bool(record.get("needs_deep_search"))

        if not needs_pdf and not needs_deep:
            continue

        principle = record.get("principle") or ""
        citation = record.get("citation") or ""
        citation_index = int(record.get("citation_index") or 0)

        if needs_pdf:
            reasons.append("pdf")
        if needs_deep:
            reasons.append("deep")

        pending.append(
            PendingTask(
                paper_key=paper_key,
                record=record,
                reasons=tuple(reasons),
                principle=principle,
                citation=citation,
                citation_index=citation_index,
            )
        )
    return pending


def split_tasks(tasks: Sequence[PendingTask], parts: int) -> List[List[PendingTask]]:
    buckets: List[List[PendingTask]] = [[] for _ in range(parts)]
    for idx, task in enumerate(tasks):
        buckets[idx % parts].append(task)
    return buckets


def write_split(
    tasks: Sequence[PendingTask],
    dest_dir: Path,
) -> Dict[str, int]:
    dest_dir.mkdir(parents=True, exist_ok=True)

    subset_state = {task.paper_key: task.record for task in tasks}
    (dest_dir / "pipeline_state.json").write_text(
        json.dumps(subset_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tasks_path = dest_dir / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            payload = {
                "paper_key": task.paper_key,
                "principle": task.record.get("principle"),
                "citation_index": task.record.get("citation_index"),
                "citation": task.citation,
                "status": task.record.get("status"),
                "pdf_url": task.record.get("pdf_url"),
                "needs": list(task.reasons),
                "needs_pdf_download": task.record.get("needs_pdf_download"),
                "needs_deep_search": task.record.get("needs_deep_search"),
                "pdf_available": task.record.get("pdf_available"),
                "pdf_saved": task.record.get("pdf_saved"),
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    principles = sorted({task.principle for task in tasks if task.principle})
    (dest_dir / "principles.txt").write_text(
        "\n".join(principles),
        encoding="utf-8",
    )

    paper_keys = [task.paper_key for task in tasks]
    (dest_dir / "paper_keys.txt").write_text(
        "\n".join(paper_keys),
        encoding="utf-8",
    )

    return {
        "tasks": len(tasks),
        "principles": len(principles),
    }


def write_responses_subset(tasks: Sequence[PendingTask], dest_path: Path) -> None:
    principle_map: Dict[str, List[PendingTask]] = {}
    for task in tasks:
        if not task.principle or not task.citation:
            continue
        principle_map.setdefault(task.principle, []).append(task)

    responses: Dict[str, List[str]] = {}
    for principle, grouped in principle_map.items():
        ordered = sorted(grouped, key=lambda task: task.citation_index)
        responses[principle] = [task.citation for task in ordered]

    dest_path.write_text(
        json.dumps(responses, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_manifest(
    split_dir: Path,
    split_id: str,
    tasks: Sequence[PendingTask],
    counts: Dict[str, int],
) -> None:
    manifest = {
        "split_id": split_id,
        "split_path": split_dir.as_posix(),
        "task_count": counts["tasks"],
        "principle_count": counts["principles"],
        "paper_keys_file": "paper_keys.txt",
        "pipeline_state_file": "pipeline_state.json",
        "responses_file": "responses_parsed.json",
        "tasks_file": "tasks.jsonl",
        "principles_file": "principles.txt",
    }
    (split_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    principle_list = ", ".join(sorted({task.principle for task in tasks if task.principle}))
    readme_content = README_TEMPLATE.format(
        split_id=split_id,
        task_count=counts["tasks"],
        principle_count=counts["principles"],
        principles=principle_list or "（无）",
        split_dir=split_dir.as_posix(),
    )
    (split_dir / "README.txt").write_text(readme_content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split pending PDF download / deep-search work across multiple machines."
    )
    parser.add_argument(
        "--pipeline-state",
        type=Path,
        default=DEFAULT_PIPELINE_STATE,
        help=f"Path to pipeline_state.json (default: {DEFAULT_PIPELINE_STATE}).",
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
        "--include-deep",
        action="store_true",
        help="Also distribute tasks that still require deep-search text extraction.",
    )
    parser.add_argument(
        "--include-failed-pdf",
        action="store_true",
        help="Re-queue entries whose status is 'pdf_download_failed' even if needs_pdf_download is false.",
    )
    parser.add_argument(
        "--emit-responses",
        action="store_true",
        help="Also emit responses_parsed.json fragments for each split.",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Create .tar.gz archive for each split directory.",
    )

    args = parser.parse_args()

    if args.splits < 1:
        raise ValueError("Number of splits must be at least 1.")

    state = load_pipeline_state(args.pipeline_state)
    pending = collect_pending_tasks(state, include_deep=args.include_deep, include_failed_pdf=args.include_failed_pdf)

    if not pending:
        print("No pending tasks found with the requested filters.")
        return

    pending.sort(key=lambda task: (task.principle, task.citation_index))

    buckets = split_tasks(pending, args.splits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, int]] = {}

    for idx, bucket in enumerate(buckets, start=1):
        split_dir = args.output_dir / f"split_{idx}"
        counts = write_split(bucket, split_dir)
        if args.emit_responses:
            write_responses_subset(bucket, split_dir / "responses_parsed.json")
        write_manifest(split_dir, f"split_{idx}", bucket, counts)
        if args.archive:
            archive_path = args.output_dir / f"{split_dir.name}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(split_dir, arcname=split_dir.name)
            counts["archive"] = str(archive_path)
        summary[f"split_{idx}"] = counts

    summary["totals"] = {
        "pending_tasks": len(pending),
        "splits": args.splits,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
