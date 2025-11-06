import json
import os
import time
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from principle_situaton import sd_pri_list, td_pri_list_100

PATTERNS_INFO_PATH = Path("Dataset/patterns_info/psy_patterns_info.json")
RESEARCH_INFO_PATH = Path("Dataset/principle_info_reserach.json")
SD_INFO_PATH = Path("Dataset/sd_principle_info.json")
TD_INFO_PATH = Path("Dataset/td_principle_info_100.json")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0
SUMMARY_TEMPERATURE = 0.2


def load_json_dict(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    if isinstance(data, list):
        return {
            item.get("construct_name", "").strip().lower(): item
            for item in data
            if isinstance(item, dict) and item.get("construct_name")
        }
    return {}


def gather_gemini_credentials() -> List[Tuple[str, str]]:
    credentials: List[Tuple[str, str]] = []
    base_default = os.getenv("BASE_URL_LIMIT")
    for idx in range(10):
        suffix = "" if idx == 0 else str(idx)
        api_key = os.getenv(f"API_KEY_LIMIT{suffix}")
        if not api_key:
            continue
        base_env = os.getenv(f"BASE_URL_LIMIT{suffix}") or base_default
        if not base_env:
            print(
                f"[warn] Missing BASE_URL_LIMIT{suffix or ''} for API_KEY_LIMIT{suffix or ''}; skipping."
            )
            continue
        credentials.append((base_env.rstrip("/"), api_key))
    if not credentials:
        raise RuntimeError(
            "No Gemini API credentials found. Set BASE_URL_LIMIT and API_KEY_LIMIT… env vars."
        )
    return credentials


class GeminiClient:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self.credentials = gather_gemini_credentials()
        self._cycle = cycle(self.credentials)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        backoff = INITIAL_BACKOFF
        for attempt in range(MAX_RETRIES):
            base_url, api_key = next(self._cycle)
            url = f"{base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": SUMMARY_TEMPERATURE,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=180,
                )
            except requests.RequestException as exc:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(backoff)
                    backoff = max(backoff * 2, 1.0)
                    continue
                raise RuntimeError(
                    f"Network failure calling Gemini API: {exc}"
                ) from exc
            if response.status_code in {429} or 500 <= response.status_code < 600:
                if attempt < MAX_RETRIES - 1:
                    retry_after = response.headers.get("Retry-After")
                    wait_time = backoff
                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                        except ValueError:
                            wait_time = backoff
                    time.sleep(wait_time)
                    backoff = max(backoff * 2, 1.0)
                    continue
                response.raise_for_status()
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                print(
                    f"[warn] Model response missing choices. Raw response: {json.dumps(data, ensure_ascii=False)}"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(backoff)
                    backoff = max(backoff * 2, 1.0)
                    continue
                raise RuntimeError("Model response missing choices.")
            content = choices[0].get("message", {}).get("content")
            if not content:
                raise RuntimeError("Model response missing content.")
            return strip_code_fences(content)
        raise RuntimeError("Exceeded maximum retries when calling Gemini API.")


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.lower().startswith("```json"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
    elif cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


def strip_leading_acknowledgement(text: str) -> str:
    prefixes = [
        "好的，", "好的,", "好的。", "当然，", "当然,", "当然。", "下面是", "以下是",
        "Here is", "Here are", "Sure,", "Sure.", "Sure!", "Sure—", "Sure-", "Sure –",
        "Of course,", "Of course.", "Certainly,", "Certainly.", "Absolutely,", "Absolutely."
    ]
    cleaned = text.strip()
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].lstrip()
    return cleaned


def summarize_field(
    client: GeminiClient,
    principle: str,
    field: str,
    guidance: str,
    text: str,
) -> str:
    if not text or not text.strip():
        return ""
    snippet = text.strip()
    if len(snippet) > 8000:
        snippet = snippet[:8000]
    system_prompt = (
        "You are an expert academic summarizer. Produce a neutral, formal paragraph that captures "
        "the essential psychological insights from the provided notes without adding commentary."
    )
    user_prompt = (
        f"Construct Name: {principle}\n"
        f"Target Section: {field}\n"
        f"Guidance: {guidance}\n"
        "Instructions:\n"
        "- Write exactly one paragraph in formal academic prose.\n"
        "- Produce between 4 and 10 sentences, keeping the paragraph under 200 words.\n"
        "- Retain key mechanisms, drivers, and consequences; omit anecdotes, dialogue, or citations.\n"
        "- Do NOT include introductions, acknowledgements, or closing remarks.\n\n"
        "Source Notes:\n"
        f"{snippet}\n\n"
        "Return only the final paragraph."
    )
    response = client.chat(system_prompt, user_prompt)
    return strip_leading_acknowledgement(response)


def collect_sd_fields(entry: Dict[str, str]) -> Dict[str, str]:
    return {
        "description": entry.get("description", ""),
        "core_mechanisms": entry.get("core_mechanisms", entry.get("core mechanism", "")),
        "real_world_manifestation": entry.get(
            "real_world_manifestation",
            entry.get("real world manifestation", ""),
        ),
    }


def collect_td_fields(entry: Dict) -> Dict[str, str]:
    description = entry.get("description")
    if not description:
        description = entry.get("definition", "")

    core_field = entry.get("core_mechanisms") or ""
    if isinstance(core_field, dict):
        core_parts = [
            core_field.get("cognitive_patterns", ""),
            core_field.get("emotional_signatures", ""),
            core_field.get("behavioral_tendencies", ""),
        ]
        core_field = "\n\n".join(part.strip()
                                 for part in core_parts if part and part.strip())

    real_field = entry.get("real_world_manifestation") or ""
    if isinstance(real_field, dict):
        real_parts = [
            real_field.get("under_stress", ""),
            real_field.get("in_conflict", ""),
            real_field.get("in_positive_situations", ""),
        ]
        real_field = "\n\n".join(part.strip()
                                 for part in real_parts if part and part.strip())

    return {
        "description": description or "",
        "core_mechanisms": core_field or "",
        "real_world_manifestation": real_field or "",
    }


SD_FIELD_GUIDANCE = {
    "description": (
        "Provide a crisp definition of the principle and capture its defining psychological characteristics."
    ),
    "core_mechanisms": (
        "Summarize the major cognitive, motivational, and evolutionary mechanisms that sustain the principle."
    ),
    "real_world_manifestation": (
        "Explain how the principle appears in real situations, highlighting consequences, applications, and double-edged effects."
    ),
}

TD_FIELD_GUIDANCE = {
    "description": (
        "Provide a precise definition of the trait and briefly explain how it fits within personality structure."
    ),
    "core_mechanisms": (
        "Summarize hallmark cognitive, emotional, and behavioural mechanisms that sustain the trait."
    ),
    "real_world_manifestation": (
        "Explain how the trait presents under stress, in conflict, and during positive situations."
    ),
}

PLACEHOLDER_SUBSTRINGS = [
    "provided corpus does not contain",
    "provided documents do not contain",
    "the corpus does not contain",
    "corpus does not contain",
    "corpus does not include",
    "source text does not include",
    "source material does not include",
    "no relevant content found",
    "no supporting evidence in the corpus",
    "no supporting documents in the corpus",
    "insufficient source text",
    "source corpus is empty",
    "未提供相关文本",
    "未提供任何相关语料",
    "语料不包含",
    "缺少源文本",
    "there is no information",
    "offers no information",
    "contains no information",
    "provides no information",
    "text does not contain information",
    "text does not describe",
    "text does not provide",
    "corpus does not describe",
    "corpus does not analyze",
    "corpus does not mention",
    "corpus does not provide",
    "documents do not describe",
    "documents do not provide",
    "no information available",
    "no information provided",
]

PLACEHOLDER_EXACT_VALUES = {
    "n/a",
    "na",
    "none",
    "no data",
    "not provided",
    "无相关信息",
    "暂无",
}


def field_needs_regeneration(value: Optional[str]) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return True
        lowered = cleaned.lower()
        if lowered in PLACEHOLDER_EXACT_VALUES:
            return True
        for fragment in PLACEHOLDER_SUBSTRINGS:
            if fragment in lowered:
                return True
        return False
    return False


def detect_problem_fields(principle: str, entry: Dict) -> List[str]:
    fields: List[str] = []
    if principle in sd_pri_list:
        if field_needs_regeneration(entry.get("description")):
            fields.append("description")
        if field_needs_regeneration(entry.get("core_mechanisms")):
            fields.append("core_mechanisms")
        if field_needs_regeneration(entry.get("real_world_manifestation")):
            fields.append("real_world_manifestation")
    else:
        normalized = collect_td_fields(entry)
        description_value = normalized.get("description", "")
        original_description = entry.get("description")
        if field_needs_regeneration(description_value) or not isinstance(original_description, str) or not original_description.strip():
            fields.append("description")

        normalized_core = normalized.get("core_mechanisms", "")
        original_core = entry.get("core_mechanisms")
        if isinstance(original_core, dict) or not isinstance(original_core, str):
            fields.append("core_mechanisms")
        elif field_needs_regeneration(normalized_core):
            fields.append("core_mechanisms")

        normalized_real = normalized.get("real_world_manifestation", "")
        original_real = entry.get("real_world_manifestation")
        if isinstance(original_real, dict) or not isinstance(original_real, str):
            fields.append("real_world_manifestation")
        elif field_needs_regeneration(normalized_real):
            fields.append("real_world_manifestation")
    if fields:
        fields = list(dict.fromkeys(fields))
    return fields


def summarize_sd_fields(
    client: GeminiClient,
    principle: str,
    source_entry: Dict,
    fields: List[str],
) -> Dict[str, str]:
    field_texts = collect_sd_fields(source_entry)
    updates: Dict[str, str] = {}
    for field in fields:
        text = field_texts.get(field, "")
        if not text or not text.strip():
            print(
                f"[warn] No source text available for '{principle}' field '{field}'. Skipping."
            )
            continue
        guidance = SD_FIELD_GUIDANCE[field]
        updates[field] = summarize_field(
            client, principle, field, guidance, text)
    return updates


def summarize_td_fields(
    client: GeminiClient,
    principle: str,
    source_entry: Dict,
    fields: List[str],
) -> Dict[str, str]:
    field_texts = collect_td_fields(source_entry)
    updates: Dict[str, str] = {}
    for field_key in fields:
        text = field_texts.get(field_key, "")
        if not text or not text.strip():
            print(
                f"[warn] No source text available for '{principle}' field '{field_key}'. Skipping."
            )
            continue
        guidance = TD_FIELD_GUIDANCE[field_key]
        updates[field_key] = summarize_field(
            client, principle, field_key, guidance, text)
    return updates


def copy_sd_fields(
    source_entry: Dict,
    fields: List[str],
) -> Dict[str, str]:
    field_texts = collect_sd_fields(source_entry)
    return {field: field_texts.get(field, "") for field in fields}


def copy_td_fields(
    source_entry: Dict,
    fields: List[str],
) -> Dict[str, str]:
    field_texts = collect_td_fields(source_entry)
    return {field: field_texts.get(field, "") for field in fields}


def backfill_sd_principle(
    client: GeminiClient,
    principle: str,
    source_entry: Dict[str, str],
) -> Dict[str, str]:
    field_texts = collect_sd_fields(source_entry)
    output = {"construct_name": principle}
    for field, text in field_texts.items():
        guidance = SD_FIELD_GUIDANCE[field]
        output[field] = summarize_field(
            client, principle, field, guidance, text)
    return output


def backfill_td_principle(
    client: GeminiClient,
    principle: str,
    source_entry: Dict,
) -> Dict:
    field_texts = collect_td_fields(source_entry)
    output = {"construct_name": principle}
    for field_key, text in field_texts.items():
        guidance = TD_FIELD_GUIDANCE[field_key]
        summary = summarize_field(client, principle, field_key, guidance, text)
        output[field_key] = summary
    return output


def determine_source_entry(
    principle: str,
    research_map: Dict[str, Dict],
    sd_map: Dict[str, Dict],
    td_map: Dict[str, Dict],
) -> Tuple[Optional[Dict], Optional[str]]:
    key = principle.lower()
    if key in research_map:
        return research_map[key], "research"
    if key in sd_map:
        return sd_map[key], "sd"
    if key in td_map:
        return td_map[key], "td"
    return None, None


def main() -> None:
    expected: List[str] = []
    seen: set[str] = set()
    for name in sd_pri_list + td_pri_list_100:
        if name not in seen:
            seen.add(name)
            expected.append(name)

    patterns_info = load_json_dict(PATTERNS_INFO_PATH)
    case_map = {key.lower(): key for key in patterns_info.keys()}

    missing: List[str] = []
    partial_updates: Dict[str, Dict[str, object]] = {}

    for name in expected:
        normalized = name.lower()
        actual_key = case_map.get(normalized)
        if actual_key is None:
            missing.append(name)
            continue
        entry = patterns_info.get(actual_key)
        if not isinstance(entry, dict):
            missing.append(name)
            continue
        problem_fields = detect_problem_fields(name, entry)
        if problem_fields:
            partial_updates[name] = {
                "fields": problem_fields,
                "key": actual_key,
            }

    if not missing and not partial_updates:
        print("No missing principles or problematic fields detected. Nothing to backfill.")
        return

    research_map = load_json_dict(RESEARCH_INFO_PATH)
    sd_map = load_json_dict(SD_INFO_PATH)
    td_map = load_json_dict(TD_INFO_PATH)

    client: Optional[GeminiClient] = None

    def ensure_client() -> GeminiClient:
        nonlocal client
        if client is None:
            client = GeminiClient()
        return client

    added_count = 0
    updated_field_count = 0
    failed_missing: List[str] = []
    failed_updates: Dict[str, List[str]] = {}

    for principle in missing:
        source_entry, source_type = determine_source_entry(
            principle, research_map, sd_map, td_map)
        if not source_entry:
            print(
                f"[warn] No source information found for '{principle}'. Skipping.")
            failed_missing.append(principle)
            continue
        try:
            if source_type == "research":
                if principle in sd_pri_list:
                    summary = backfill_sd_principle(
                        ensure_client(), principle, source_entry)
                else:
                    summary = backfill_td_principle(
                        ensure_client(), principle, source_entry)
            else:
                if principle in sd_pri_list:
                    summary_fields = collect_sd_fields(source_entry)
                else:
                    summary_fields = collect_td_fields(source_entry)
                summary = {"construct_name": principle, **summary_fields}
        except Exception as exc:
            print(f"[error] Failed to summarize '{principle}': {exc}")
            failed_missing.append(principle)
            continue
        patterns_info[principle] = summary
        case_map[principle.lower()] = principle
        added_count += 1
        print(f"[info] Backfilled '{principle}'.")

    for principle, payload in partial_updates.items():
        fields = payload["fields"]
        if not fields:
            continue
        store_key = payload["key"]
        entry = patterns_info.get(store_key)
        if not isinstance(entry, dict):
            print(f"[warn] Existing entry for '{principle}' is not valid. Skipping field updates.")
            failed_updates[principle] = fields
            continue
        source_entry, source_type = determine_source_entry(
            principle, research_map, sd_map, td_map)
        if not source_entry:
            print(
                f"[warn] No source information found for '{principle}'. Skipping field updates.")
            failed_updates[principle] = fields
            continue
        updates: Dict[str, str]
        if principle in sd_pri_list:
            if source_type == "research":
                updates = summarize_sd_fields(
                    ensure_client(), principle, source_entry, fields
                )
            else:
                updates = copy_sd_fields(source_entry, fields)
            if not updates:
                print(f"[warn] No updates generated for '{principle}'.")
                failed_updates[principle] = fields
                continue
            entry.setdefault("construct_name", principle)
            for field, value in updates.items():
                entry[field] = value
        else:
            if source_type == "research":
                updates = summarize_td_fields(
                    ensure_client(), principle, source_entry, fields
                )
            else:
                updates = copy_td_fields(source_entry, fields)
            if not updates:
                print(f"[warn] No updates generated for '{principle}'.")
                failed_updates[principle] = fields
                continue
            entry.setdefault("construct_name", principle)
            for field_key, value in updates.items():
                if field_key == "description":
                    entry["description"] = value
                    entry.pop("definition", None)
                elif field_key == "core_mechanisms":
                    entry["core_mechanisms"] = value
                elif field_key == "real_world_manifestation":
                    entry["real_world_manifestation"] = value
                else:
                    entry[field_key] = value
        applied_fields = sorted(updates.keys())
        updated_field_count += len(applied_fields)
        patterns_info[store_key] = entry
        print(
            f"[info] Updated '{principle}' fields: {', '.join(applied_fields)}.")

    case_map_final = {key.lower(): key for key in patterns_info.keys()}
    remaining_missing: List[str] = []
    remaining_problem_fields: Dict[str, List[str]] = {}
    for principle in expected:
        key = case_map_final.get(principle.lower())
        if key is None:
            remaining_missing.append(principle)
            continue
        entry = patterns_info.get(key)
        if not isinstance(entry, dict):
            remaining_missing.append(principle)
            continue
        problems = detect_problem_fields(principle, entry)
        if problems:
            remaining_problem_fields[principle] = problems

    pending_field_total = sum(len(fields)
                              for fields in remaining_problem_fields.values())
    summary_lines: List[str] = [
        f"[summary] Remaining missing principles: {len(remaining_missing)}.",
        f"[summary] Principles with unresolved fields: {len(remaining_problem_fields)} (total fields: {pending_field_total}).",
    ]
    if failed_missing:
        snippet = ", ".join(failed_missing[:5])
        if len(failed_missing) > 5:
            snippet += ", ..."
        summary_lines.append(
            f"[summary] Failed to backfill {len(failed_missing)} principles during this run: {snippet}.")
    if failed_updates:
        keys = list(failed_updates.keys())
        snippet = ", ".join(keys[:5])
        if len(keys) > 5:
            snippet += ", ..."
        summary_lines.append(
            f"[summary] Failed to update fields for {len(keys)} principles: {snippet}.")

    if added_count == 0 and updated_field_count == 0:
        print("No new entries or field updates were produced.")
        for line in summary_lines:
            print(line)
        return

    PATTERNS_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    PATTERNS_INFO_PATH.write_text(
        json.dumps(patterns_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_parts: List[str] = []
    if added_count:
        summary_parts.append(f"added {added_count} principle(s)")
    if updated_field_count:
        summary_parts.append(
            f"updated {updated_field_count} field(s)")
    summary_text = " and ".join(summary_parts) if summary_parts else "no changes"
    print(f"[done] {summary_text} in {PATTERNS_INFO_PATH}.")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
