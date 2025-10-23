"""
检测并修复 Dataset/sft_data/SFT_data_without_thoughts.json 中被打断的对话。
通过--mode选择检测还是修复功能
python detect_interruptions.py --mode detect 仅仅收集对话打断样本到need_fix.json。
python detect_interruptions.py --mode repair 修复样本并合并到原数据集。

"""

import argparse
import asyncio
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm


DatasetItem = Dict[str, Any]

SOURCE_PATH = Path("Dataset/sft_data/SFT_data_without_thoughts.json")
NEED_FIX_PATH = Path("Dataset/sft_data/need_fix.json")

# 连字符或破折号后直接换行（或结尾）被视作“被打断”。
INTERRUPTION_PATTERN = re.compile(r'[-—](?:\r?\n)+(?:\"|$)')


# --- LLM 配置（与 gen_SFT_dataset.py 保持一致的风格） ---
MODELS: Dict[str, Dict[str, Optional[str]]] = {
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat",
    },
    "pumpkin_gemini_pro": {
        "api_key": os.getenv("API_KEY_FULL"),
        "base_url": os.getenv("BASE_URL_FULL"),
        "model_name": "gemini-2.5-pro-preview-05-06",
    },
    "pumpkin_claude_sonnet": {
        "api_key": os.getenv("API_KEY_FULL"),
        "base_url": os.getenv("BASE_URL_FULL"),
        "model_name": "claude-4-5-sonnet-20250929",
    },
    "pumpkin_gpt": {
        "api_key": os.getenv("API_KEY_FULL"),
        "base_url": os.getenv("BASE_URL_FULL"),
        "model_name": "gpt-5",
    },
}

INTERRUPTION_FIX_MODEL_PROFILE = "pumpkin_gpt"
MAX_CONCURRENT_FIX_REQUESTS = 8
MAX_FIX_RETRIES = 2

FIX_SYSTEM_PROMPT = (
    "You are an expert data curator who repairs truncated dialogue turns in SFT samples. "
    "Return valid JSON only. Preserve every field except for the minimal text edits needed "
    "to complete interrupted sentences."
)

FIX_USER_PROMPT_TEMPLATE = """以下是一个SFT数据样本。请沿用原有结构，仅修复 `conversations` 数组中被打断的发言，使句子语义完整。
要求：
1. 不增加或删除对话轮次，`from` 字段保持不变。
2. 除了修复 `value` 字段必要的文字，其余字段（`principle`、`situation`、`system`）不得修改。
3. 保持 `value` 原有的换行与段落格式。
4. 只返回一个 JSON 对象，包含与输入相同的四个顶层字段。

样本 JSON：
```json
{sample_json}
```

请给出修复后的 JSON："""


def load_dataset(path: Path) -> List[DatasetItem]:
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def save_dataset(path: Path, data: Sequence[DatasetItem]) -> None:
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)


def detect_interrupted_samples(dataset: Sequence[DatasetItem]) -> Tuple[List[int], List[DatasetItem]]:
    indices: List[int] = []
    samples: List[DatasetItem] = []
    for idx, sample in enumerate(dataset):
        for turn in sample.get("conversations", []):
            value = turn.get("value")
            if isinstance(value, str) and INTERRUPTION_PATTERN.search(value):
                indices.append(idx)
                samples.append(sample)
                break
    return indices, samples


def write_need_fix_file(samples: Sequence[DatasetItem]) -> None:
    with NEED_FIX_PATH.open("w", encoding="utf-8") as outfile:
        json.dump(list(samples), outfile, ensure_ascii=False, indent=2)


def read_need_fix_file() -> List[DatasetItem]:
    if not NEED_FIX_PATH.exists():
        return []
    with NEED_FIX_PATH.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def detect_and_report(dataset: Sequence[DatasetItem]) -> Tuple[List[int], List[DatasetItem]]:
    indices, samples = detect_interrupted_samples(dataset)
    print(f"数据集中共有 {len(dataset)} 个样本。")
    print(f"检测到 {len(indices)} 个样本存在被打断的对话。")
    write_need_fix_file(samples)
    print(f"已写入 {len(samples)} 个样本至 {NEED_FIX_PATH}")
    return indices, samples


def build_repair_prompt(sample: DatasetItem) -> str:
    return FIX_USER_PROMPT_TEMPLATE.format(
        sample_json=json.dumps(sample, ensure_ascii=False, indent=2)
    )


def extract_json_from_response(content: Optional[str]) -> Optional[str]:
    if not content:
        return None
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```",
                      content, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else content).strip()


def ensure_structure(original: DatasetItem, candidate: DatasetItem) -> Optional[DatasetItem]:
    if not isinstance(candidate, dict):
        return None

    conversations = candidate.get("conversations")
    if not isinstance(conversations, list):
        return None

    original_conversations = original.get("conversations", [])
    if len(conversations) != len(original_conversations):
        return None

    for orig_turn, new_turn in zip(original_conversations, conversations):
        if not isinstance(new_turn, dict):
            return None
        if orig_turn.get("from") != new_turn.get("from"):
            return None

    updated = copy.deepcopy(original)
    updated["conversations"] = conversations
    return updated


def still_interrupted(sample: DatasetItem) -> bool:
    for turn in sample.get("conversations", []):
        value = turn.get("value")
        if isinstance(value, str) and INTERRUPTION_PATTERN.search(value):
            return True
    return False


def init_client(profile: str) -> Tuple[AsyncOpenAI, str]:
    try:
        config = MODELS[profile]
    except KeyError as exc:
        raise ValueError(f"模型配置 '{profile}' 不存在") from exc

    api_key = config.get("api_key")
    base_url = config.get("base_url")
    model_name = config.get("model_name")

    if not api_key:
        raise ValueError(f"模型 '{profile}' 缺少 API Key 配置")
    if not base_url:
        raise ValueError(f"模型 '{profile}' 缺少 Base URL 配置")
    if not model_name:
        raise ValueError(f"模型 '{profile}' 缺少模型名称配置")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return client, model_name


async def repair_sample_async(
    sample: DatasetItem,
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    model_name: str,
    sample_index: int,
) -> Optional[DatasetItem]:
    prompt = build_repair_prompt(sample)
    messages = [
        {"role": "system", "content": FIX_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(1, MAX_FIX_RETRIES + 1):
        async with semaphore:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
            )
        raw_content = response.choices[0].message.content
        candidate_str = extract_json_from_response(raw_content)

        if candidate_str:
            try:
                parsed_candidate = json.loads(candidate_str)
            except json.JSONDecodeError:
                messages = [
                    {"role": "system", "content": FIX_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "上一次的回答不是有效的 JSON，请重新输出符合要求的 JSON。\n\n"
                            + prompt
                        ),
                    },
                ]
                continue

            validated = ensure_structure(sample, parsed_candidate)
            if validated and not still_interrupted(validated):
                return validated

            # 如果结构或检测未通过，则带反馈重试。
            failure_reason = "修复后仍检测到被打断" if validated else "返回结构不匹配"
            messages = [
                {"role": "system", "content": FIX_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"{failure_reason}，请再次修复并保持要求。\n\n" + prompt
                    ),
                },
            ]
        else:
            messages = [
                {"role": "system", "content": FIX_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "未收到可解析的回答，请仅返回符合要求的 JSON。\n\n" + prompt
                    ),
                },
            ]

    print(f"[WARN] 样本索引 {sample_index} 仍未成功修复。")
    return None


async def repair_all_samples(
    indices: Sequence[int],
) -> List[Tuple[int, DatasetItem]]:
    if not indices:
        return []

    client, model_name = init_client(INTERRUPTION_FIX_MODEL_PROFILE)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FIX_REQUESTS)

    need_fix_samples = read_need_fix_file()
    if len(need_fix_samples) != len(indices):
        print(
            f"[WARN] need_fix.json 中样本数量 ({len(need_fix_samples)}) 与检测到的数量 ({len(indices)}) 不一致，将按较小值处理。"
        )
    paired = list(zip(indices, need_fix_samples[: len(indices)]))

    async def run_with_position(
        position: int, idx: int, sample: DatasetItem
    ) -> Tuple[int, int, Optional[DatasetItem]]:
        repaired = await repair_sample_async(sample, semaphore, client, model_name, idx)
        return position, idx, repaired

    tasks: List[asyncio.Task[Tuple[int, int, Optional[DatasetItem]]]] = []
    for position, (idx, sample) in enumerate(paired):
        task = asyncio.create_task(run_with_position(position, idx, sample))
        tasks.append(task)

    ordered_results: List[Optional[Tuple[int, DatasetItem]]] = [
        None] * len(tasks)
    for pending in async_tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="修复进度",
    ):
        position, idx, result = await pending
        if result is not None:
            ordered_results[position] = (idx, result)

    repaired_pairs: List[Tuple[int, DatasetItem]] = []
    for entry in ordered_results:
        if entry is not None:
            idx, sample = entry
            repaired_pairs.append((idx, sample))

    return repaired_pairs


async def run_repair_mode(dataset: List[DatasetItem]) -> None:
    indices, _ = detect_and_report(dataset)

    if not indices:
        return

    repaired_pairs = await repair_all_samples(indices)

    for replace_index, new_sample in repaired_pairs:
        dataset[replace_index] = new_sample

    if repaired_pairs:
        print(f"成功修复 {len(repaired_pairs)} 个样本，正在覆盖原数据集……")
        save_dataset(SOURCE_PATH, dataset)
    else:
        print("没有样本被成功修复，原数据集未被修改。")

    remaining_indices, remaining_samples = detect_interrupted_samples(dataset)
    write_need_fix_file(remaining_samples)
    print(f"修复完成后仍有 {len(remaining_indices)} 个样本待修复。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检测或修复被打断的对话样本。")
    parser.add_argument(
        "--mode",
        choices=["detect", "repair"],
        default="detect",
        help="选择执行模式：'detect' 仅检测并输出 need_fix.json，'repair' 还会调用 LLM 尝试修复。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(SOURCE_PATH)

    if args.mode == "detect":
        detect_and_report(dataset)
    else:
        asyncio.run(run_repair_mode(dataset))


if __name__ == "__main__":
    main()
