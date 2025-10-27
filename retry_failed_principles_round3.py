import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

from google_search import fetch_references, _write_outputs

OUTPUT_DIR = Path("Dataset/gemini_references")
FAILURE_PATH = OUTPUT_DIR / "failed_principles_round3.txt"
ERRORS_PATH = OUTPUT_DIR / "errors.json"


def _load_failure_list(path: Path) -> Iterable[str]:
    if not path.exists():
        print(f"[info] Failure list not found at {path}. Nothing to retry.")
        return []

    principles = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not principles:
        print(f"[info] Failure list at {path} is empty. Nothing to retry.")
    else:
        print(f"[info] Loaded {len(principles)} principles from {path}")

    return principles


def _load_json_dict(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON file at {path}") from exc


def _process_principles_sync(principles: Iterable[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    aggregated: Dict[str, str] = {}
    errors: Dict[str, str] = {}

    principles = list(principles)
    total = len(principles)
    if total == 0:
        return aggregated, errors

    for index, principle in enumerate(principles, start=1):
        print(f"[info] ({index}/{total}) Requesting {principle}...")
        try:
            content = fetch_references(principle)
        except Exception as exc:  # noqa: BLE001
            errors[principle] = str(exc)
            print(f"[error] {principle}: {exc}")
            continue

        cleaned = content.strip()
        if not cleaned:
            errors[principle] = "Empty response content"
            print(f"[error] {principle}: Empty response content")
            continue

        aggregated[principle] = cleaned
        print(f"[saved] {principle}")

    print(
        f"[info] Completed retry run. Success: {len(aggregated)} | Failed: {len(errors)}")
    return aggregated, errors


def main() -> None:
    retry_principles = list(_load_failure_list(FAILURE_PATH))
    if not retry_principles:
        return

    aggregated_existing = _load_json_dict(OUTPUT_DIR / "responses_raw.json")
    errors_existing = _load_json_dict(ERRORS_PATH)

    aggregated_new, errors_new = _process_principles_sync(retry_principles)

    aggregated_total = dict(aggregated_existing)
    aggregated_total.update(aggregated_new)

    final_errors = {principle: message for principle,
                    message in errors_existing.items()}
    for principle in aggregated_new:
        final_errors.pop(principle, None)
    final_errors.update(errors_new)

    _write_outputs(aggregated_total, final_errors, OUTPUT_DIR)

    remaining_round3 = [
        name for name in retry_principles if name in final_errors]
    if remaining_round3:
        FAILURE_PATH.write_text(
            "\n".join(remaining_round3),
            encoding="utf-8",
        )
        print(
            f"[warn] {len(remaining_round3)} principles still failed after manual retry. "
            f"Updated list saved to {FAILURE_PATH}"
        )
    else:
        FAILURE_PATH.write_text("", encoding="utf-8")
        print("[info] All previously failed principles succeeded during manual retry.")


if __name__ == "__main__":
    main()
