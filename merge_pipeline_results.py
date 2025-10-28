#!/usr/bin/env python3
"""
Helper script to package and merge pipeline results across machines.

Usage examples:

  # On remote machine after processing a split:
  python merge_pipeline_results.py prepare \
      --dataset-root Dataset/papers_info \
      --split-dir Dataset/task_splits/split_1 \
      --output Dataset/task_splits/split_1/results

  # Back on the main machine, after collecting results directories:
  python merge_pipeline_results.py apply \
      --dataset-root Dataset/papers_info \
      --results Dataset/task_splits/split_1/results Dataset/task_splits/split_2/results
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


DATASET_DEFAULT = Path("Dataset/papers_info")


@dataclass(frozen=True)
class TaskSpec:
    paper_key: str
    citation: str


def read_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc


def read_jsonl(path: Path) -> Iterator[Dict]:
    if not path.exists():
        return iter(())
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_task_specs(tasks_path: Path) -> List[TaskSpec]:
    specs: List[TaskSpec] = []
    for record in read_jsonl(tasks_path):
        paper_key = record.get("paper_key")
        citation = record.get("citation") or ""
        if paper_key:
            specs.append(TaskSpec(paper_key=paper_key, citation=citation))
    if not specs:
        raise ValueError(f"No tasks found in {tasks_path}")
    return specs


def ensure_directory(path: Path, force: bool = False) -> None:
    if path.exists():
        if force:
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"Output directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=True)


def subset_pipeline_state(state_path: Path, paper_keys: Sequence[str]) -> Dict[str, Dict]:
    state = read_json(state_path)
    subset: Dict[str, Dict] = {}
    missing: List[str] = []
    for key in paper_keys:
        record = state.get(key)
        if record:
            subset[key] = record
        else:
            missing.append(key)
    if missing:
        print(f"[warn] Missing {len(missing)} entries in pipeline_state for provided keys.")
    return subset


def filter_log(log_path: Path, specs: Sequence[TaskSpec]) -> List[Dict]:
    if not log_path.exists():
        return []
    targets = {spec.paper_key for spec in specs}
    citations = {spec.citation for spec in specs if spec.citation}
    filtered: List[Dict] = []
    for record in read_jsonl(log_path):
        paper_key = record.get("paper_key")
        citation = record.get("citation")
        if paper_key and paper_key in targets:
            filtered.append(record)
        elif citation and citation in citations:
            filtered.append(record)
    return filtered


def copy_artifacts(
    dataset_root: Path,
    specs: Sequence[TaskSpec],
    output_root: Path,
) -> Tuple[int, int]:
    pdf_src = dataset_root / "pdfs"
    text_src = dataset_root / "texts"
    pdf_dest = output_root / "pdfs"
    text_dest = output_root / "texts"
    pdf_dest.mkdir(parents=True, exist_ok=True)
    text_dest.mkdir(parents=True, exist_ok=True)

    pdf_count = 0
    text_count = 0

    for spec in specs:
        pdf_file = pdf_src / f"{spec.paper_key}.pdf"
        text_file = text_src / f"{spec.paper_key}.txt"
        if pdf_file.exists():
            shutil.copy2(pdf_file, pdf_dest / pdf_file.name)
            pdf_count += 1
        if text_file.exists():
            shutil.copy2(text_file, text_dest / text_file.name)
            text_count += 1
    return pdf_count, text_count


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def prepare_results(dataset_root: Path, split_dir: Path, output_dir: Path, force: bool) -> Dict[str, int]:
    tasks_path = split_dir / "tasks.jsonl"
    specs = load_task_specs(tasks_path)
    ensure_directory(output_dir, force=force)

    logs_dir = dataset_root / "logs"
    pipeline_path = logs_dir / "pipeline_state.json"
    subset = subset_pipeline_state(pipeline_path, [spec.paper_key for spec in specs])
    write_json(output_dir / "pipeline_state.json", subset)

    errors_records = filter_log(logs_dir / "errors.jsonl", specs)
    if errors_records:
        write_jsonl(output_dir / "errors.jsonl", errors_records)
    not_found_records = filter_log(logs_dir / "not_found.jsonl", specs)
    if not_found_records:
        write_jsonl(output_dir / "not_found.jsonl", not_found_records)

    pdf_count, text_count = copy_artifacts(dataset_root, specs, output_dir)

    manifest_payload = {
        "mode": "prepare",
        "split_dir": str(split_dir),
        "task_count": len(specs),
        "pdf_copied": pdf_count,
        "text_copied": text_count,
    }
    write_json(output_dir / "manifest.json", manifest_payload)
    return {
        "tasks": len(specs),
        "pdfs": pdf_count,
        "texts": text_count,
        "errors": len(errors_records),
        "not_found": len(not_found_records),
    }


def append_jsonl(dest_path: Path, records: Iterable[Dict]) -> int:
    records_list = list(records)
    if not records_list:
        return 0
    existing_lines = set()
    if dest_path.exists():
        with dest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                existing_lines.add(line.rstrip("\n"))
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with dest_path.open("a", encoding="utf-8") as handle:
        for record in records_list:
            line = json.dumps(record, ensure_ascii=False)
            if line in existing_lines:
                continue
            handle.write(line)
            handle.write("\n")
            existing_lines.add(line)
            appended += 1
    return appended


def build_principles_index(state: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    index: Dict[str, List[Dict]] = {}
    for record in state.values():
        principle = record.get("principle")
        if not principle:
            continue
        entry = {
            "paper_key": record.get("paper_key"),
            "status": record.get("status"),
            "pdf_saved": record.get("pdf_saved"),
            "needs_pdf_download": record.get("needs_pdf_download"),
            "needs_deep_search": record.get("needs_deep_search"),
            "text_saved": record.get("text_saved"),
            "deep_search_attempted": record.get("deep_search_attempted"),
            "lookup_sources": record.get("lookup_sources"),
        }
        index.setdefault(principle, []).append(entry)
    return index


def apply_results(dataset_root: Path, result_dirs: Sequence[Path]) -> Dict[str, int]:
    logs_dir = dataset_root / "logs"
    pipeline_path = logs_dir / "pipeline_state.json"
    base_state = read_json(pipeline_path)

    pdf_dest = dataset_root / "pdfs"
    text_dest = dataset_root / "texts"
    pdf_dest.mkdir(parents=True, exist_ok=True)
    text_dest.mkdir(parents=True, exist_ok=True)

    pdf_copied = 0
    text_copied = 0
    updated_keys = set()
    errors_appended = 0
    not_found_appended = 0

    for result_dir in result_dirs:
        result_dir = result_dir.resolve()
        fragment_path = result_dir / "pipeline_state.json"
        fragment = read_json(fragment_path)
        for key, record in fragment.items():
            base_state[key] = record
            updated_keys.add(key)

        errors_path = result_dir / "errors.jsonl"
        errors_appended += append_jsonl(logs_dir / "errors.jsonl", read_jsonl(errors_path))
        not_found_path = result_dir / "not_found.jsonl"
        not_found_appended += append_jsonl(logs_dir / "not_found.jsonl", read_jsonl(not_found_path))

        src_pdfs = result_dir / "pdfs"
        if src_pdfs.exists():
            for file in src_pdfs.glob("*.pdf"):
                shutil.copy2(file, pdf_dest / file.name)
                pdf_copied += 1
        src_texts = result_dir / "texts"
        if src_texts.exists():
            for file in src_texts.glob("*.txt"):
                shutil.copy2(file, text_dest / file.name)
                text_copied += 1

    write_json(pipeline_path, base_state)

    principles_index = build_principles_index(base_state)
    write_json(dataset_root / "principles.json", principles_index)

    return {
        "updated_pipeline_keys": len(updated_keys),
        "pdf_copied": pdf_copied,
        "text_copied": text_copied,
        "errors_appended": errors_appended,
        "not_found_appended": not_found_appended,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or merge pipeline processing results across machines."
    )
    subparsers = parser.add_subparsers(dest="command")

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Collect processed artifacts on a worker machine for transfer.",
    )
    prepare_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_DEFAULT,
        help="Path to Dataset/papers_info on this machine (default: Dataset/papers_info).",
    )
    prepare_parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        help="Split directory produced by split_pipeline_tasks.py.",
    )
    prepare_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the prepared result package.",
    )
    prepare_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory if it exists.",
    )

    apply_parser = subparsers.add_parser(
        "apply",
        help="Merge result packages back into the main dataset.",
    )
    apply_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_DEFAULT,
        help="Path to Dataset/papers_info on the main machine (default: Dataset/papers_info).",
    )
    apply_parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="One or more prepared result directories to merge.",
    )

    args = parser.parse_args()
    if args.command == "prepare":
        summary = prepare_results(args.dataset_root, args.split_dir, args.output, args.force)
        print(json.dumps({"prepare_summary": summary}, ensure_ascii=False, indent=2))
    elif args.command == "apply":
        summary = apply_results(args.dataset_root, args.results)
        print(json.dumps({"apply_summary": summary}, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
