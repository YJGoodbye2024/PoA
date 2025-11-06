import argparse
import asyncio
import json
import os
import time
from itertools import cycle
from typing import Dict, List, Tuple, Optional

from openai import AsyncOpenAI  # Use the asynchronous client
from tqdm.asyncio import tqdm_asyncio  # For asynchronous progress bars

# Make sure prompt_all.py and principle_situaton.py are in the same directory or accessible via PYTHONPATH
from prompt_all import scenario_sys_prompt, gen_scenario_prompt, gen_conversationtion_sys_prompt, gen_conversationtion_prompt
from principle_situaton import sd_pri_list, td_pri_list_100, Situation_list

# --- Configuration ---
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
GEMINI_MODEL = "gemini-2.5-pro"

# --- File Paths ---
PATTERNS_INFO_FILE = 'Dataset/patterns_info/psy_patterns_info.json'
OUTPUT_FILE = 'Dataset/generated_data.json'

# --- Concurrency Control ---
# 控制同时发送的API请求数量，根据你的API速率限制调整
# A good starting point is between 10 and 50.
MAX_CONCURRENT_REQUESTS = 20

# --- Functions (modified for async) ---


class ModelRunner:
    def __init__(self, name: str, client: AsyncOpenAI, model_name: str) -> None:
        self.name = name
        self.client = client
        self.model_name = model_name


async def get_model_answer_async(runner: ModelRunner, sys_prompt: str = "", user_prompt: str = ""):
    """Asynchronous function to get a response from a chat model."""
    try:
        response = await runner.client.chat.completions.create(
            model=runner.model_name,
            stream=False,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[{runner.name}] An error occurred while calling the API: {e}")
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


async def process_single_combination(principle, situation, runner: ModelRunner, semaphore):
    """
    Processes a single principle-situation pair asynchronously.
    This function combines scenario and conversation generation.
    """
    async with semaphore:  # Acquire a semaphore slot
        # Step 1: Generate Scenario
        scenario_prompt = gen_scenario_prompt.format(
            principle1_information=json.dumps(
                principle, ensure_ascii=False, indent=2),
            situation=situation
        )
        scenario_content = await get_model_answer_async(runner, scenario_sys_prompt, scenario_prompt)
        if not scenario_content:
            print(
                f"[{runner.name}] Failed to generate scenario for '{principle['construct_name']}' with '{situation}'.")
            return None

        # Step 2: Generate Conversation
        conversation_prompt = gen_conversationtion_prompt.format(
            principle1_information=json.dumps(
                principle, ensure_ascii=False, indent=2),
            scenario=scenario_content
        )
        conversation_content = await get_model_answer_async(
            runner, gen_conversationtion_sys_prompt, conversation_prompt)
        if not conversation_content:
            print(
                f"[{runner.name}] Failed to generate conversation for '{principle['construct_name']}'.")
            return None

        # Step 3: Structure and return data
        return {
            "principle": principle['construct_name'],
            "situation": situation,
            "scenario": scenario_content,
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
) -> None:
    start_time = time.time()

    # Load principle data from consolidated patterns info file
    patterns_info = load_patterns_info(PATTERNS_INFO_FILE)
    if not patterns_info:
        print("No principle information loaded. Exiting.")
        return

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

    print(f"Created {len(tasks)} tasks. Starting concurrent processing with a limit of {MAX_CONCURRENT_REQUESTS} requests...")
    print(
        f"Task allocation -> Claude: {claude_assignments}, Gemini: {gemini_assignments}")

    # Run tasks concurrently and display a progress bar
    results = await tqdm_asyncio.gather(*tasks)

    # Filter out any failed tasks (which return None)
    successful_results = [res for res in results if res is not None]

    # Save the successful results
    if successful_results:
        save_data_to_json(successful_results, OUTPUT_FILE)
    else:
        print("No data was generated successfully.")

    end_time = time.time()
    print(f"\nScript finished in {end_time - start_time:.2f} seconds.")
    print(
        f"Successfully generated {len(successful_results)} out of {len(tasks)} total items.")

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
    args = parser.parse_args()

    try:
        asyncio.run(
            main(limit=args.limit, situation_limit=args.situation_limit))
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting.")
