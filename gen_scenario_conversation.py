import argparse
import asyncio
import json
import os
import re
import time
from itertools import cycle
from typing import Dict, List, Tuple, Optional

from openai import AsyncOpenAI  # Use the asynchronous client
from tqdm.asyncio import tqdm_asyncio  # For asynchronous progress bars

# Make sure prompt_all.py and principle_situaton.py are in the same directory or accessible via PYTHONPATH
from principle_situaton import sd_pri_list, td_pri_list_100, Situation_list
from prompt_all import (
    scenario_sys_prompt,
    gen_scenario_prompt,
    gen_conversationtion_sys_prompt,
    gen_conversationtion_prompt,
    gen_conversationtion_sys_prompt_no_analysis,
    gen_conversationtion_prompt_no_analysis,
    protagonist_name_sys_prompt,
    protagonist_name_prompt,
)


# prompt for conversation (modified to include analysis input)


# --- Configuration ---
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
GEMINI_MODEL = "gemini-2.5-pro-cli"

# --- File Paths ---
PATTERNS_INFO_FILE = 'Dataset/patterns_info/psy_patterns_info.json'
OUTPUT_FILE = 'Dataset/generated_data.json'

# --- Concurrency Control ---
# 控制同时发送的API请求数量，根据你的API速率限制调整
# A good starting point is between 10 and 50.
MAX_CONCURRENT_REQUESTS = 20

# --- Retry / Timeout ---
REQUEST_TIMEOUT_SECONDS = 300
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2

# --- Functions (modified for async) ---


class ModelRunner:
    def __init__(self, name: str, client: AsyncOpenAI, model_name: str) -> None:
        self.name = name
        self.client = client
        self.model_name = model_name


async def get_model_answer_async(runner: ModelRunner, sys_prompt: str = "", user_prompt: str = ""):
    """Asynchronous function to get a response from a chat model."""
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            response = await asyncio.wait_for(
                runner.client.chat.completions.create(
                    model=runner.model_name,
                    stream=False,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return response.choices[0].message.content
        except asyncio.TimeoutError:
            print(
                f"[{runner.name}] Request timed out (attempt {attempt}/{MAX_RETRY_ATTEMPTS}).")
        except Exception as e:
            print(
                f"[{runner.name}] API call failed on attempt {attempt}/{MAX_RETRY_ATTEMPTS}: {e}")

        if attempt < MAX_RETRY_ATTEMPTS:
            backoff = RETRY_BACKOFF_SECONDS * attempt
            await asyncio.sleep(backoff)

    return None


def load_patterns_info(file_path: str) -> Dict[str, Dict]:
    """Loads principle details from the consolidated patterns info file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} contains invalid JSON.")
        return {}


def gather_gemini_credentials() -> List[Tuple[str, str]]:
    """Collect BASE_URL/API key pairs for Gemini model."""
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
                f"[warn] Missing BASE_URL_LIMIT{suffix or ''} for available API key; skipping.")
            continue
        credentials.append((base_env.rstrip("/"), api_key))

    if not credentials:
        raise RuntimeError(
            "No Gemini API credentials found in environment variables.")
    return credentials


def _clean_section_text(text: str) -> str:
    """Trim whitespace and drop common markdown separators from a section."""
    separator_lines = {"---", "***", "___"}
    lines = text.splitlines()
    start = 0
    end = len(lines)

    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1

    while start < end and lines[start].strip() in separator_lines:
        start += 1
        while start < end and not lines[start].strip():
            start += 1

    while end > start and lines[end - 1].strip() in separator_lines:
        end -= 1
        while end > start and not lines[end - 1].strip():
            end -= 1

    return "\n".join(lines[start:end]).strip()


def split_part_sections(content: str) -> Tuple[str, str]:
    """
    Split model output into Part 1 (scenario) and Part 2 (analysis) sections.
    Falls back to returning the full content as the scenario if split fails.
    """
    heading_regex = re.compile(
        r'(?im)^\s*(?:[#>*-]+\s*)?(?:\*\*|__)?\s*Part\s*(\d)\s*[:\-–—.]*.*?$',
        re.MULTILINE,
    )
    matches = list(heading_regex.finditer(content))

    part1_match = next((m for m in matches if m.group(1) == "1"), None)
    part2_match = next((m for m in matches if m.group(1) == "2"), None)

    if not part1_match or not part2_match or part2_match.start() <= part1_match.end():
        print("[warn] Unable to detect distinct Part 1/Part 2 sections; storing full output under 'scenario'.")
        return content.strip(), ""

    scenario_raw = content[part1_match.end():part2_match.start()]
    analysis_raw = content[part2_match.end():]

    scenario = _clean_section_text(scenario_raw)
    analysis = _clean_section_text(analysis_raw)

    if not analysis:
        fallback_start = part2_match.start()
        analysis = _clean_section_text(content[fallback_start:])

    return scenario, analysis


async def process_single_combination(
        principle,
        situation,
        runner: ModelRunner,
        semaphore,
        scenario_override: Optional[str] = None,
        analysis_override: Optional[str] = None,
        include_analysis: bool = True,
        conversation_sys_prompt_override: Optional[str] = None,
        conversation_prompt_override: Optional[str] = None):
    """
    Processes a single principle-situation pair asynchronously.
    This function combines scenario and conversation generation.
    """
    async with semaphore:  # Acquire a semaphore slot
        if scenario_override is None:
            # Step 1: Generate Scenario
            scenario_prompt = gen_scenario_prompt.format(
                principle1_information=json.dumps(
                    principle, ensure_ascii=False, indent=2),
                situation=situation
            )
            scenario_content_raw = await get_model_answer_async(runner, scenario_sys_prompt, scenario_prompt)
            if not scenario_content_raw:
                print(
                    f"[{runner.name}] Failed to generate scenario for '{principle['construct_name']}' with '{situation}'.")
                return None

            scenario_content, analysis_content = split_part_sections(
                scenario_content_raw)
        else:
            scenario_content = scenario_override
            analysis_content = analysis_override or ""
            if not scenario_content:
                print(
                    f"[warn] Empty scenario provided for '{principle['construct_name']}' with '{situation}'. Skipping conversation regeneration.")
                return None

        protagonist_prompt = protagonist_name_prompt.format(
            scenario=scenario_content)
        protagonist_name_raw = await get_model_answer_async(
            runner, protagonist_name_sys_prompt, protagonist_prompt)
        protagonist_name = (protagonist_name_raw or "").strip()
        if "\n" in protagonist_name:
            protagonist_name = protagonist_name.splitlines()[0].strip()

        conversation_sys = conversation_sys_prompt_override or gen_conversationtion_sys_prompt
        conversation_template = conversation_prompt_override or gen_conversationtion_prompt

        prompt_kwargs = {
            "principle1_information": json.dumps(
                principle, ensure_ascii=False, indent=2),
            "scenario": scenario_content,
        }
        if include_analysis:
            prompt_kwargs["analysis"] = analysis_content
        conversation_prompt = conversation_template.format(**prompt_kwargs)
        conversation_content = await get_model_answer_async(
            runner, conversation_sys, conversation_prompt)
        if not conversation_content:
            print(
                f"[{runner.name}] Failed to generate conversation for '{principle['construct_name']}'.")
            return None

        # Step 3: Structure and return data
        return {
            "pattern": principle['construct_name'],
            "situation": situation,
            "scenario": scenario_content,
            "analysis": analysis_content,
            "protagonist": protagonist_name,
            "conversation": conversation_content
        }


def save_data_to_json(data, file_path):
    """Saves the generated data list to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\nData successfully saved to {file_path}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")


async def main(
    limit: Optional[int] = None,
    situation_limit: Optional[int] = None,
    only_conversation: bool = False,
    conversation_limit: Optional[int] = None,
    conversation_without_analysis: bool = False,
) -> None:
    start_time = time.time()

    # Load principle data from consolidated patterns info file
    patterns_info = load_patterns_info(PATTERNS_INFO_FILE)
    if not patterns_info:
        print("No principle information loaded. Exiting.")
        return

    base_url_full = os.getenv("BASE_URL_FULL")
    api_key_full = os.getenv("API_KEY_FULL")
    if not base_url_full:
        raise RuntimeError("BASE_URL_FULL environment variable is not set.")
    if not api_key_full:
        raise RuntimeError("API_KEY_FULL environment variable is not set.")

    claude_runner = ModelRunner(
        "claude",
        AsyncOpenAI(api_key=api_key_full, base_url=base_url_full.rstrip("/")),
        CLAUDE_MODEL,
    )

    gemini_credentials = gather_gemini_credentials()
    gemini_runners = [
        ModelRunner(
            f"gemini-{idx + 1}",
            AsyncOpenAI(api_key=api_key, base_url=base_url),
            GEMINI_MODEL,
        )
        for idx, (base_url, api_key) in enumerate(gemini_credentials)
    ]
    gemini_cycle = cycle(gemini_runners)

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Create a list of all tasks to be run
    tasks = []
    claude_assignments = 0
    gemini_assignments = 0
    success_count = 0
    task_entries: List[Tuple[int, Dict]] = []
    existing_data: List[Dict] = []
    final_dataset: List[Dict]

    if only_conversation:
        if not os.path.exists(OUTPUT_FILE):
            print(
                f"[error] Dataset file '{OUTPUT_FILE}' 不存在，无法重新生成对话。")
            return
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                if not isinstance(loaded_data, list):
                    raise ValueError("Existing dataset is not a list.")
                existing_data = loaded_data
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[error] 无法读取已有数据集：{exc}")
            return

        final_dataset = list(existing_data)
        total_entries = len(existing_data)
        if total_entries == 0:
            print("[info] 数据集为空，无需重新生成对话。")
            return

        process_count = total_entries
        if conversation_limit is not None and conversation_limit > 0:
            process_count = min(conversation_limit, total_entries)
        elif limit is not None and limit > 0:
            process_count = min(limit, total_entries)
        if process_count < total_entries:
            print(
                f"[info] 仅重新生成前 {process_count} 条样本的对话（共 {total_entries} 条）。")

        include_analysis = not conversation_without_analysis
        convo_sys_prompt = (
            gen_conversationtion_sys_prompt_no_analysis
            if not include_analysis else gen_conversationtion_sys_prompt
        )
        convo_prompt_template = (
            gen_conversationtion_prompt_no_analysis
            if not include_analysis else gen_conversationtion_prompt
        )

        for idx in range(process_count):
            entry = existing_data[idx]
            principle_name = entry.get("pattern") or entry.get("principle")
            principle_info = patterns_info.get(principle_name or "")
            if not principle_info:
                print(
                    f"[warn] 找不到原理“{principle_name}”的详细信息，保留原对话。")
                continue

            principle_entry = dict(principle_info)
            principle_entry.setdefault("construct_name", principle_name)

            situation = entry.get("situation", "")
            scenario_text = entry.get("scenario", "")
            analysis_text = entry.get("analysis", "")

            task_index = len(tasks)
            if task_index % 2 == 0:
                runner = claude_runner
                claude_assignments += 1
            else:
                runner = next(gemini_cycle)
                gemini_assignments += 1

            tasks.append(process_single_combination(
                principle_entry,
                situation,
                runner,
                semaphore,
                scenario_override=scenario_text,
                analysis_override=analysis_text,
                include_analysis=include_analysis,
                conversation_sys_prompt_override=convo_sys_prompt,
                conversation_prompt_override=convo_prompt_template,
            ))
            task_entries.append((idx, entry))

        if not tasks:
            print("[warn] 没有任何样本进入重新生成流程。")
            return

        print(f"Created {len(tasks)} conversation regeneration tasks. Starting concurrent processing with a limit of {MAX_CONCURRENT_REQUESTS} requests...")
        print(
            f"Task allocation -> Claude: {claude_assignments}, Gemini: {gemini_assignments}")

        results = await tqdm_asyncio.gather(*tasks)

        for (idx, original_entry), res in zip(task_entries, results):
            if res is None:
                continue
            updated_entry = dict(original_entry)
            updated_entry.update({
                "pattern": res["pattern"],
                "situation": res["situation"],
                "scenario": res["scenario"],
                "analysis": res["analysis"],
                "protagonist": res["protagonist"],
                "conversation": res["conversation"],
            })
            final_dataset[idx] = updated_entry
            success_count += 1

        successful_results = final_dataset
        total_tasks = len(tasks)
    else:
        seen_names = set()
        principles_to_process = []
        missing_principles = []
        for principle_name in sd_pri_list + td_pri_list_100:
            if principle_name in seen_names:
                continue
            seen_names.add(principle_name)
            info = patterns_info.get(principle_name)
            if not info:
                missing_principles.append(principle_name)
                continue
            entry = dict(info)
            entry.setdefault("construct_name", principle_name)
            principles_to_process.append(entry)
        if limit is not None and limit > 0:
            principles_to_process = principles_to_process[:limit]
            print(
                f"[info] Limiting to first {len(principles_to_process)} principles for testing.")

        if missing_principles:
            print(
                f"[warn] Missing principle info for: {', '.join(missing_principles)}")

        if not principles_to_process:
            print("No valid principles to process. Exiting.")
            return

        selected_situations = list(Situation_list)
        if situation_limit is not None and situation_limit > 0:
            selected_situations = selected_situations[:situation_limit]
            print(
                f"[info] Limiting to first {len(selected_situations)} situations for testing.")

        all_combinations = [
            (principle, situation)
            for principle in principles_to_process
            for situation in selected_situations
        ]

        for idx, (principle, situation) in enumerate(all_combinations):
            if idx % 2 == 0:
                runner = claude_runner
                claude_assignments += 1
            else:
                runner = next(gemini_cycle)
                gemini_assignments += 1
            tasks.append(process_single_combination(
                principle, situation, runner, semaphore))

        print(
            f"Created {len(tasks)} tasks. Starting concurrent processing with a limit of {MAX_CONCURRENT_REQUESTS} requests...")
        print(
            f"Task allocation -> Claude: {claude_assignments}, Gemini: {gemini_assignments}")

        results = await tqdm_asyncio.gather(*tasks)

        # Filter out any failed tasks (which return None)
        successful_results = [res for res in results if res is not None]
        success_count = len(successful_results)
        total_tasks = len(tasks)

    # Save the successful results
    if successful_results:
        save_data_to_json(successful_results, OUTPUT_FILE)
    else:
        print("No data was generated successfully.")

    end_time = time.time()
    print(f"\nScript finished in {end_time - start_time:.2f} seconds.")
    print(
        f"Successfully generated {success_count} out of {total_tasks} total items.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate scenarios and conversations based on principle summaries."
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N principles from psy_patterns_info.json (testing helper).",
    )
    parser.add_argument(
        "--situation-limit",
        type=int,
        help="Only use the first M situations from Situation_list (testing helper).",
    )
    parser.add_argument(
        "--only-conversation",
        action="store_true",
        help="只重新生成已有数据集中的对话，不重新生成场景。",
    )
    parser.add_argument(
        "--conversation-limit",
        type=int,
        help="在 --only-conversation 模式下，仅重写前 N 条样本的对话。",
    )
    parser.add_argument(
        "--conversation-without-analysis",
        action="store_true",
        help="在 --only-conversation 模式下，生成对话时不传入 analysis，并改用不含 analysis 的 prompt。",
    )
    args = parser.parse_args()

    try:
        asyncio.run(
            main(
                limit=args.limit,
                situation_limit=args.situation_limit,
                only_conversation=args.only_conversation,
                conversation_limit=args.conversation_limit,
                conversation_without_analysis=args.conversation_without_analysis,
            ))
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting.")
