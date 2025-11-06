import argparse
import asyncio
import json
import os
import time
from collections import deque
from itertools import cycle
from typing import Dict, List, Optional

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from prompt_all import (
    gen_checklist_sys_prompt,
    gen_checklist_prompt,
)

# --- Configuration ---
GEMINI_MODEL = "gemini-2.5-pro-cli"

# --- File Paths ---
PATTERNS_INFO_FILE = 'Dataset/patterns_info/psy_patterns_info.json'
SOURCE_DATA_FILE = 'Dataset/generated_data.json'
OUTPUT_FILE = 'Dataset/generated_data_with_checklists.json'

# --- Concurrency / Rate ---
MAX_CONCURRENT_REQUESTS = 20
REQUEST_TIMEOUT_SECONDS = 300
MAX_RETRY_ATTEMPTS = 5
RETRY_BACKOFF_SECONDS = 3
RATE_LIMIT_REQUESTS_PER_MINUTE = 4
RATE_LIMIT_WINDOW_SECONDS = 60


class ModelRunner:
    def __init__(self, name: str, client: AsyncOpenAI, model_name: str) -> None:
        self.name = name
        self.client = client
        self.model_name = model_name
        self._rate_lock = asyncio.Lock()
        self._request_timestamps: deque[float] = deque()


def load_patterns_info(file_path: str) -> Dict[str, Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"找不到模式信息文件: {file_path}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"模式信息文件 JSON 无效: {exc}")


def load_source_data(file_path: str) -> List[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("源数据不是数组")
            return data
    except FileNotFoundError:
        raise RuntimeError(f"找不到源数据文件: {file_path}")
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"源数据文件读取失败: {exc}")


def gather_gemini_credentials() -> List[ModelRunner]:
    credentials: List[ModelRunner] = []
    base_default = os.getenv("BASE_URL_LIMIT")

    for idx in range(10):
        suffix = "" if idx == 0 else str(idx)
        api_key = os.getenv(f"API_KEY_LIMIT{suffix}")
        if not api_key:
            continue
        base_env = os.getenv(f"BASE_URL_LIMIT{suffix}") or base_default
        if not base_env:
            print(
                f"[warn] Missing BASE_URL_LIMIT{suffix or ''} for available API key; skipping.")
            continue
        client = AsyncOpenAI(api_key=api_key, base_url=base_env.rstrip('/'))
        credentials.append(ModelRunner(
            f"gemini-{idx + 1}", client, GEMINI_MODEL))

    if not credentials:
        raise RuntimeError("未检测到 Gemini API 配置。")
    return credentials


async def _respect_rate_limit(runner: ModelRunner) -> None:
    while True:
        async with runner._rate_lock:
            now = time.monotonic()
            timestamps = runner._request_timestamps
            while timestamps and now - timestamps[0] >= RATE_LIMIT_WINDOW_SECONDS:
                timestamps.popleft()

            if len(timestamps) < RATE_LIMIT_REQUESTS_PER_MINUTE:
                timestamps.append(now)
                return

            wait_for = RATE_LIMIT_WINDOW_SECONDS - (now - timestamps[0])
        await asyncio.sleep(max(wait_for, 0.01))


async def get_model_answer_async(runner: ModelRunner, sys_prompt: str, user_prompt: str) -> Optional[str]:
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            await _respect_rate_limit(runner)
            response = await asyncio.wait_for(
                runner.client.chat.completions.create(
                    model=runner.model_name,
                    stream=False,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            choices = getattr(response, 'choices', None)
            if not choices:
                raise ValueError("Empty choices from API")
            return choices[0].message.content
        except asyncio.TimeoutError:
            print(
                f"[{runner.name}] Request timed out (attempt {attempt}/{MAX_RETRY_ATTEMPTS}).")
        except Exception as exc:
            print(
                f"[{runner.name}] API call failed on attempt {attempt}/{MAX_RETRY_ATTEMPTS}: {exc}")

        if attempt < MAX_RETRY_ATTEMPTS:
            await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
    return None


async def process_entry(entry: Dict, principle_info: Dict, runner: ModelRunner, semaphore: asyncio.Semaphore) -> Optional[Dict]:
    async with semaphore:
        scenario = entry.get("scenario", "").strip()
        analysis = entry.get("analysis", "").strip()
        pattern_name = entry.get("pattern") or entry.get("principle")
        if not scenario or not analysis:
            print(
                f"[warn] 条目缺少 scenario 或 analysis，跳过 pattern={pattern_name}。")
            return None

        user_prompt = gen_checklist_prompt.format(
            principle1_information=json.dumps(
                principle_info, ensure_ascii=False, indent=2),
            scenario=scenario,
            analysis=analysis,
        )
        checklist = await get_model_answer_async(runner, gen_checklist_sys_prompt, user_prompt)
        if not checklist:
            print(f"[warn] 生成 checklist 失败，pattern={pattern_name}")
            return None

        result = dict(entry)
        result["checklist"] = checklist.strip()
        return result


async def main(limit: Optional[int]) -> None:
    patterns_info_raw = load_patterns_info(PATTERNS_INFO_FILE)
    pattern_map = {}
    for key, info in patterns_info_raw.items():
        if not isinstance(info, dict):
            continue
        construct_name = info.get("construct_name") or key
        if construct_name:
            pattern_map[construct_name.strip().lower()] = info
        pattern_map[key.strip().lower()] = info

    source_entries = load_source_data(SOURCE_DATA_FILE)
    if limit is not None and limit > 0:
        source_entries = source_entries[:limit]
        print(f"[info] 仅处理前 {len(source_entries)} 条样本。")

    runners = gather_gemini_credentials()
    runner_cycle = cycle(runners)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    tasks = []
    assigned = 0
    for entry in source_entries:
        pattern_name = entry.get("pattern") or entry.get("principle")
        lookup_key = (pattern_name or "").strip().lower()
        principle_info = pattern_map.get(lookup_key)
        if not principle_info:
            print(f"[warn] 找不到模式 {pattern_name} 的信息，跳过。")
            continue
        runner = next(runner_cycle)
        assigned += 1
        tasks.append(process_entry(entry, principle_info, runner, semaphore))

    if not tasks:
        print("没有任务需要执行。")
        return

    print(
        f"Created {len(tasks)} checklist tasks. Starting processing with concurrency {MAX_CONCURRENT_REQUESTS}...")
    results = await tqdm_asyncio.gather(*tasks)
    successful = [res for res in results if res is not None]

    if not successful:
        print("所有 checklist 生成都失败。")
        return

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(successful, f, ensure_ascii=False, indent=4)
    print(f"共生成 {len(successful)} 条 checklist，已写入 {OUTPUT_FILE}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate conversation checklists from analysis fields.')
    parser.add_argument('--limit', type=int, help='仅处理前 N 条样本。')
    args = parser.parse_args()

    try:
        asyncio.run(main(limit=args.limit))
    except KeyboardInterrupt:
        print('\n用户中断，退出。')
