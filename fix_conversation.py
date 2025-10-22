import asyncio
from openai import AsyncOpenAI
import json
import time
import os
import re
from tqdm.asyncio import tqdm

# --- 配置 ---
# --- Triage Model (用于分类的低成本模型) ---
TRIAGE_API_KEY = "sk-a5582064b3c444249b2cdc825c76eebc"
TRIAGE_BASE_URL = "https://api.deepseek.com"
TRIAGE_MODEL = "deepseek-chat"

# --- Correction Model (用于修正的高质量模型) ---
CORRECTION_API_KEY = "sk-WKak50Ii5K68isoe7bF316D6E7Eb44A3Aa32843eBaE4866f"
CORRECTION_BASE_URL = "https://api.pumpkinaigc.online/v1"
CORRECTION_MODEL = "claude-sonnet-4-20250514"

# --- 文件路径 ---
INPUT_FILE = 'Dataset/unfixed_generated_data.json'
CLEAN_DATA_FILE = 'Dataset/temp_clean_data.json'
NEEDS_FIXING_FILE = 'Dataset/temp_needs_fixing.json'
FIXED_DATA_FILE = 'Dataset/temp_fixed_data.json'
FINAL_OUTPUT_FILE = 'Dataset/final_clean_dataset.json'

# --- 并发控制 ---
MAX_CONCURRENT_REQUESTS = 30


# --- API 客户端 ---
TRIAGE_ACLIENT = AsyncOpenAI(api_key=TRIAGE_API_KEY, base_url=TRIAGE_BASE_URL)
CORRECTION_ACLIENT = AsyncOpenAI(
    api_key=CORRECTION_API_KEY, base_url=CORRECTION_BASE_URL)

# --- Prompts (MODIFICATION: Split into SYSTEM and USER prompts) ---
# --- Triage Prompts ---
TRIAGE_SYSTEM_PROMPT = """
**Role:** You are a text analysis tool focused on dialogue structure.
**Task:** Analyze the following dialogue from a script. Your only task is to determine if any character interrupts another character's speech. An interruption is typically marked by a sentence ending in "—" or "..." and being immediately followed by another character speaking.
Your entire response MUST be a single word:
- `YES` if an interruption exists.
- `NO` if the dialogue is strictly turn-based and no one cuts anyone off.
Do not provide any explanation or other text.
"""
TRIAGE_USER_PROMPT = """
**Dialogue to Analyze:**
{conversation_text}
"""

# --- Correction Prompts ---
CORRECTION_SYSTEM_PROMPT = """
**Role:** You are a precision-focused dialogue correction tool.

**Primary Directive:**
Your single and only task is to fix character interruptions in the provided `Conversation to Revise`.

**Step-by-Step Logic:**
1.  **Identify:** Locate turns where a speaker is cut off (their line ends in "—" or "...").
2.  **Complete:** Finish the interrupted speaker's sentence in a way that is logical and consistent with their character, using the `Scenario Context` for reference.
3.  **Adjust:** Ensure the interrupting speaker's line follows as a new, separate turn, preserving its original intent.

**ABSOLUTE REQUIREMENT:**
You are forbidden from making any other changes. The structure, content, and flow of all other dialogue turns must be preserved **exactly** as they are in the original. Do not add narrative paragraphs. Do not rephrase sentences that are not part of an interruption. 

**Strict Output Format:**
Your final output MUST be **ONLY the complete, corrected conversation text**.

"""
CORRECTION_USER_PROMPT = """
---
**[Scenario Context]**
{scenario}
---
**[Conversation to Revise]**
{conversation}
"""


# --- Core Functions ---
def load_data(file_path):
    """Loads data from a JSON file."""
    if not os.path.exists(file_path):
        print(f"Warning: Input file not found: {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return []


def save_data(data, file_path):
    """Saves data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved {len(data)} items to {file_path}")
    except IOError as e:
        print(f"Error saving data to {file_path}: {e}")


async def classify_conversation_async(item, semaphore, client, model):
    """Uses a low-cost LLM to classify if a conversation has interruptions."""
    async with semaphore:
        try:
            # MODIFICATION: Use both system and user prompts
            user_prompt = TRIAGE_USER_PROMPT.format(
                conversation_text=item.get("conversation", ""))
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=5
            )
            decision = response.choices[0].message.content.strip().upper()
            return item, decision
        except Exception as e:
            print(
                f"Error classifying item for principle '{item.get('principle')}': {e}")
            return item, "NO"


async def correct_conversation_async(item, semaphore, client, model):
    """Uses a high-quality LLM to fix a conversation, returning only the corrected text."""
    async with semaphore:
        try:
            # MODIFICATION: Use both system and user prompts
            user_prompt = CORRECTION_USER_PROMPT.format(
                scenario=item.get("scenario", ""),
                conversation=item.get("conversation", "")
            )
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=8192
            )
            corrected_conversation_text = response.choices[0].message.content
            return item, corrected_conversation_text
        except Exception as e:
            print(
                f"Error correcting item for principle '{item.get('principle')}': {e}")
            return item, None


async def main():
    """Main function to run the entire data cleaning pipeline."""
    start_time = time.time()

    source_data = load_data(INPUT_FILE)
    if not source_data:
        return
    print(f"Loaded {len(source_data)} items from {INPUT_FILE}")

    # --- Step 1 & 2: Triage and Segregate ---
    print("\n--- Step 1 & 2: Classifying conversations for interruptions... ---")
    triage_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    triage_tasks = [classify_conversation_async(
        item, triage_semaphore, TRIAGE_ACLIENT, TRIAGE_MODEL) for item in source_data]

    clean_data = []
    needs_fixing = []
    for future in tqdm(asyncio.as_completed(triage_tasks), total=len(triage_tasks), desc="Classifying"):
        item, decision = await future
        if decision == "YES":
            needs_fixing.append(item)
        else:
            clean_data.append(item)

    print(
        f"Classification complete: {len(clean_data)} items are clean, {len(needs_fixing)} items need fixing.")
    save_data(clean_data, CLEAN_DATA_FILE)
    save_data(needs_fixing, NEEDS_FIXING_FILE)

    # --- Step 3: Correct Data ---
    print("\n--- Step 3: Correcting conversations that have interruptions... ---")
    fixed_data = []
    if needs_fixing:
        correction_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        correction_tasks = [correct_conversation_async(
            item, correction_semaphore, CORRECTION_ACLIENT, CORRECTION_MODEL) for item in needs_fixing]

        for future in tqdm(asyncio.as_completed(correction_tasks), total=len(correction_tasks), desc="Correcting"):
            original_item, corrected_conversation = await future
            if corrected_conversation is not None:
                original_item["conversation"] = corrected_conversation
                fixed_data.append(original_item)

        print(
            f"Correction complete: Successfully fixed {len(fixed_data)} out of {len(needs_fixing)} items.")
        save_data(fixed_data, FIXED_DATA_FILE)
    else:
        print("No items needed fixing. Skipping correction step.")

    # --- Step 4: Merge Data ---
    print("\n--- Step 4: Merging clean and fixed data... ---")
    final_dataset = clean_data + fixed_data
    save_data(final_dataset, FINAL_OUTPUT_FILE)

    end_time = time.time()
    print(f"\nPipeline finished in {end_time - start_time:.2f} seconds.")
    print(f"Final dataset contains {len(final_dataset)} items.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Partial progress may be saved in temp files.")
