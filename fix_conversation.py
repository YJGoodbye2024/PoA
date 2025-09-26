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
CORRECTION_MODEL = "gemini-2.5-pro-preview-05-06"

# --- 文件路径 ---
INPUT_FILE = 'Dataset/generated_data_test.json'  # 您的原始数据文件
# 中间文件
CLEAN_DATA_FILE = 'Dataset/temp_clean_data.json'
NEEDS_FIXING_FILE = 'Dataset/temp_needs_fixing.json'
FIXED_DATA_FILE = 'Dataset/temp_fixed_data.json'
# 最终输出文件
FINAL_OUTPUT_FILE = 'Dataset/final_clean_dataset.json'

# --- 并发控制 ---
MAX_CONCURRENT_REQUESTS = 30  # 可根据您的API速率限制调整


# --- API 客户端 ---
TRIAGE_ACLIENT = AsyncOpenAI(api_key=TRIAGE_API_KEY, base_url=TRIAGE_BASE_URL)
CORRECTION_ACLIENT = AsyncOpenAI(
    api_key=CORRECTION_API_KEY, base_url=CORRECTION_BASE_URL)

# --- Prompts ---
TRIAGE_PROMPT = """
**Role:** You are a text analysis tool focused on dialogue structure.
**Task:** Analyze the following dialogue from a script. Your only task is to determine if any character interrupts another character's speech. An interruption is typically marked by a sentence ending in "—" or "..." and being immediately followed by another character speaking.
Your entire response MUST be a single word:
- `YES` if an interruption exists.
- `NO` if the dialogue is strictly turn-based and no one cuts anyone off.
Do not provide any explanation or other text.
**Dialogue to Analyze:**
{conversation_text}
"""

CORRECTION_PROMPT = """
**Role:** You are an expert script editor and psychologist. Your task is to revise a dialogue script to improve its structure for data processing, while meticulously preserving its original dramatic tension, character voice, and underlying psychological meaning.
**Context:** The provided dialogue contains instances where one character interrupts another. For the purpose of training an AI, this dialogue needs to be converted into a strict turn-based format where each character finishes their line before the next one speaks.
**Your Task:**
Rewrite the `conversation` value in the provided JSON object. You must follow these rules:
1.  **Identify the Interruption:** Locate the turns where a character's speech ends abruptly (often with "—" or "...") and is immediately followed by another character's line.
2.  **Complete the Thought:** Allow the interrupted character to finish their sentence or thought in a way that is natural and consistent with their profile in the `scenario`.
3.  **Adjust the Interrupter:** Modify the interrupting character's line so it follows sequentially. Their dialogue should still feel reactive and maintain its original intent (e.g., disagreement, urgency), but it must not overlap with the previous line.
4.  **Preserve Quality:** The core of your task is preservation. The revised dialogue's tone, pacing, character consistency, and the way it demonstrates the psychological `principle` must remain as close to the original as possible.
5.  **Strict Output Format:** Your final output MUST be the complete original JSON object, with ONLY the value of the "conversation" key modified. Do not change the "principle", "situation", or "scenario" keys. Do not add any explanations or notes outside the JSON structure.
**Example of Revision:**
* **Original Interruption:**
    Sarah: ...The source verification took longer than expected because—
    Tom: (cuts her off) I don't need excuses, I need copy.
* **Correctly Revised Turn-Based Version:**
    Sarah: ...The source verification took longer than expected because our main source was unreachable until an hour ago.
    Tom: (impatiently) I don't need the whole story, Sarah, I need copy.
---
**JSON Object to Revise:**
{full_json_object}
"""

# --- NEW: Added Prompts for JSON Fixing ---
format_prompt_system = '''
You are an expert-level JSON correction tool.
Your task is to receive a potentially malformed string and return a syntactically perfect JSON object.
You must follow these Core Correction Rules:
1.  **Quotation:** Ensure all string keys and values are enclosed in proper double quotes (`"`).
2.  **Internal Quotes:** If a string value contains internal, unescaped double quotes, remove them to ensure validity.
3.  **Trailing Commas:** Remove any trailing commas after the last element in objects and arrays.
4.  **Brackets & Braces:** Ensure all brackets (`[]`) and braces (`{}`) are correctly matched and closed.
'''

format_prompt_user = '''
The following string is a malformed JSON object. Please correct it according to your rules.
**String to Fix:**
{malformed_string}
**Output Requirements:**
- Your response MUST be the corrected JSON object and nothing else.
- DO NOT output any explanatory text, introductory phrases, or markdown code blocks (e.g., ```json).
- The output must be pure plain text that a standard JSON parser can process successfully.
'''

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
            prompt = TRIAGE_PROMPT.format(
                conversation_text=item.get("conversation", ""))
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5
            )
            decision = response.choices[0].message.content.strip().upper()
            return item, decision
        except Exception as e:
            print(
                f"Error classifying item for principle '{item.get('principle')}': {e}")
            return item, "NO"

# --- NEW: Added JSON fixing helper function ---


async def fix_json_async(malformed_string, client):
    """Calls the LLM with a specific prompt to fix a malformed JSON string."""
    try:
        user_prompt = format_prompt_user.format(
            malformed_string=malformed_string)
        response = await client.chat.completions.create(
            model=TRIAGE_MODEL,  # Can use the cheaper model for simple syntax fixes
            messages=[
                {"role": "system", "content": format_prompt_system},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=8192
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n  -> ❌ 在尝试自动修复JSON API调用时发生错误: {e}")
        return None

# --- MODIFIED: The main correction function now includes the robust parsing logic ---


async def correct_conversation_async(item, semaphore, client, model):
    """Uses a high-quality LLM to fix a conversation, with a retry mechanism for JSON parsing."""
    async with semaphore:
        try:
            prompt = CORRECTION_PROMPT.format(
                full_json_object=json.dumps(item, ensure_ascii=False, indent=2))
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=8192
            )
            raw_json_string = response.choices[0].message.content

            # --- Start of the robust parsing logic ---
            # 1. First, strip leading/trailing whitespace and clean markdown fences
            cleaned_string = raw_json_string.strip()
            match = re.search(
                r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned_string, re.DOTALL)
            json_to_parse = match.group(1).strip() if match else cleaned_string

            # 2. First attempt to parse
            try:
                return json.loads(json_to_parse)
            except json.JSONDecodeError:
                print(
                    f"\n警告：修正模型返回的JSON解析失败 (原则: '{item.get('principle')}')... 正在尝试自动语法修复...")

                # 3. If it fails, call the dedicated fixer LLM
                corrected_string = await fix_json_async(json_to_parse, TRIAGE_ACLIENT)

                if corrected_string:
                    # 4. Second and final attempt to parse
                    try:
                        return json.loads(corrected_string.strip())
                    except json.JSONDecodeError:
                        print(f"  -> ❌ 自动语法修复后再次解析失败。已跳过该条目。")
                        return None
                else:
                    print(f"  -> ❌ 自动语法修复API调用失败。已跳过该条目。")
                    return None
        except Exception as e:
            print(
                f"Error correcting item for principle '{item.get('principle')}': {e}")
            return None


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
            result = await future
            if result is not None:
                fixed_data.append(result)

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
