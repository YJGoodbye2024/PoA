import json
from pathlib import Path

"""
This script augments existing pattern information with source references
"""

INFO_PATH = Path("Dataset/patterns_info/psy_patterns_info.json")
REF_PATH = Path("Dataset/gemini_references/responses_parsed_origin.json")
OUTPUT_PATH = Path("Dataset/patterns_info/psy_patterns_info_with_source.json")


def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise RuntimeError(f"Missing required file: {path}")


def main() -> None:
    info = load_json(INFO_PATH)
    refs = load_json(REF_PATH)

    refs_lower = {key.lower(): value for key, value in refs.items()}

    combined = {}
    for pattern, payload in info.items():
        entry = dict(payload)
        source_list = refs.get(pattern)
        if source_list is None:
            source_list = refs_lower.get(pattern.lower(), [])
        entry["source"] = source_list
        combined[pattern] = entry

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(combined)} patterns with sources to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
