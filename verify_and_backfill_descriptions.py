import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backfill_patterns_info import (
    GeminiClient,
    collect_sd_fields,
    collect_td_fields,
    determine_source_entry,
    load_json_dict,
    backfill_sd_principle,
    backfill_td_principle,
)
from principle_situaton import sd_pri_list, td_pri_list_100

PATTERNS_PATH = Path("Dataset/patterns_info/psy_patterns_info.json")
RESEARCH_INFO_PATH = Path("Dataset/principle_info_reserach.json")
SD_INFO_PATH = Path("Dataset/sd_principle_info.json")
TD_INFO_PATH = Path("Dataset/td_principle_info_100.json")

CHECK_SYS_PROMPT = (
    "你是一名心理学条目校对助手。严格按照指令，用 JSON 输出结果。"
)
CHECK_USER_PROMPT = """请判断以下描述内容是否适配所对应的心理学原则名称。

输出要求：
1. 只输出一个 JSON 对象，格式如下：
{{
  "match": "是" 或 "否",
  "reason": "一句话描述判断理由"
}}
2. 不要输出除 JSON 以外的任何内容。
3. 判定条件很宽松，只有在描述显然对应的是其他原则时，才令 "match" 为 "否"；否则，哪怕是定义不够准确，都令 "match" 为 "是"。

原则名称：{construct_name}

描述：
{description}
"""

FIELD_ORDER = (
    "construct_name",
    "description",
    "core_mechanisms",
    "real_world_manifestation",
)


def load_patterns() -> Dict[str, Dict]:
    if not PATTERNS_PATH.exists():
        raise RuntimeError(f"找不到文件：{PATTERNS_PATH}")
    with PATTERNS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError("patterns_info JSON 格式错误，应为字典。")
    return data


def description_mentions_construct(name: str, description: str) -> bool:
    if not description:
        return False
    pattern = re.compile(re.escape(name), re.IGNORECASE)
    return bool(pattern.search(description))


def detect_candidates(patterns: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
    candidates: List[Tuple[str, Dict]] = []
    for key, entry in patterns.items():
        if not isinstance(entry, dict):
            continue
        construct_name = entry.get("construct_name") or key
        description = entry.get("description", "")
        if not description_mentions_construct(construct_name, description):
            candidates.append((key, entry))
    return candidates


def ask_mismatch(
    client: GeminiClient, construct_name: str, description: str
) -> Optional[bool]:
    try:
        response = client.chat(
            CHECK_SYS_PROMPT,
            CHECK_USER_PROMPT.format(
                construct_name=construct_name,
                description=description.strip() or "(描述为空)",
            ),
        )
        # print(CHECK_USER_PROMPT.format(construct_name=construct_name,
        #       description=description.strip() or "(描述为空)",))
    except Exception as exc:
        print(
            f"[warn] 模型接口错误（construct={construct_name!r}）：{exc}"
        )
        return None
    # print(response)
    match = re.search(r'"match"\s*:\s*"([^"]+)"', response)
    if not match:
        print(
            f"[warn] 无法解析模型 JSON 回答：construct={construct_name!r}, response={response!r}"
        )
        return None
    match_value = match.group(1)
    if match_value == "否":
        return True  # 说明描述与原则不符，需要改写
    if match_value == "是":
        return False  # 描述与原则相符，无需处理
    print(
        f"[warn] JSON 中缺少有效的 'match' 字段：construct={construct_name!r}, response={response!r}"
    )
    return None


def build_ordered_entry(entry: Dict[str, str]) -> Dict[str, str]:
    ordered = OrderedDict()
    for field in FIELD_ORDER:
        if field in entry:
            ordered[field] = entry[field]
    for key, value in entry.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def regenerate_entry(
    client: GeminiClient,
    construct_name: str,
    store_key: str,
    patterns: Dict[str, Dict],
    research_map: Dict[str, Dict],
    sd_map: Dict[str, Dict],
    td_map: Dict[str, Dict],
) -> bool:
    source_entry, source_type = determine_source_entry(
        construct_name, research_map, sd_map, td_map
    )
    if not source_entry:
        print(f"[warn] 找不到 '{construct_name}' 的参考资料，无法改写。")
        return False

    try:
        if construct_name in sd_pri_list:
            if source_type == "research":
                rewritten = backfill_sd_principle(
                    client, construct_name, source_entry
                )
            else:
                fields = collect_sd_fields(source_entry)
                rewritten = {"construct_name": construct_name, **fields}
        else:
            if source_type == "research":
                rewritten = backfill_td_principle(
                    client, construct_name, source_entry
                )
            else:
                fields = collect_td_fields(source_entry)
                rewritten = {"construct_name": construct_name, **fields}
    except Exception as exc:
        print(f"[error] 改写 '{construct_name}' 失败：{exc}")
        return False

    patterns[store_key] = build_ordered_entry(rewritten)
    return True


def main(limit: Optional[int], dry_run: bool) -> None:
    patterns = load_patterns()
    candidates = detect_candidates(patterns)
    if limit is not None and limit > 0:
        candidates = candidates[:limit]
    print(f"候选条目数量：{len(candidates)}")
    if not candidates:
        return

    client = GeminiClient()
    research_map = load_json_dict(RESEARCH_INFO_PATH)
    sd_map = load_json_dict(SD_INFO_PATH)
    td_map = load_json_dict(TD_INFO_PATH)

    flagged: List[Tuple[str, str]] = []
    for key, entry in candidates:
        construct_name = entry.get("construct_name") or key
        description = entry.get("description", "")
        try:
            result = ask_mismatch(client, construct_name, description)
        except Exception as exc:
            print(
                f"[warn] 调用模型判断 '{construct_name}' 时失败：{exc}。跳过此条。"
            )
            continue
        if result is True:
            flagged.append((key, construct_name))

    print(f"模型判定需要改写的条目：{len(flagged)}")
    if not flagged:
        return

    updated = 0
    for store_key, construct_name in flagged:
        success = regenerate_entry(
            client, construct_name, store_key, patterns, research_map, sd_map, td_map
        )
        if success:
            updated += 1

    if updated == 0:
        print("没有条目被改写。")
        return

    if dry_run:
        print(f"[dry-run] 将改写 {updated} 条，但未写回文件。")
        return

    PATTERNS_PATH.write_text(
        json.dumps(patterns, ensure_ascii=False, indent=4), encoding="utf-8"
    )
    print(f"已改写 {updated} 条记录，并写回 {PATTERNS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="检测 principles description 是否与名称匹配，并在需要时改写。"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="仅处理前 N 条候选记录，用于测试。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检测并打印结果，不写回文件。",
    )
    args = parser.parse_args()

    try:
        main(limit=args.limit, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n用户中断，退出。")
