import asyncio
from openai import AsyncOpenAI  # Use the asynchronous client
import json
import time
from tqdm.asyncio import tqdm_asyncio  # For asynchronous progress bars

# Make sure prompt_all.py and principle_situaton.py are in the same directory or accessible via PYTHONPATH
from prompt_all import scenario_sys_prompt, gen_scenario_prompt, gen_conversationtion_sys_prompt, gen_conversationtion_prompt
from principle_situaton import sd_pri_list, td_pri_list_100, Situation_list

# --- Configuration ---
BASE_URL = "https://api.pumpkinaigc.online/v1"
API_KEY = "sk-WKak50Ii5K68isoe7bF316D6E7Eb44A3Aa32843eBaE4866f"
CHOSED_MODEL = "claude-sonnet-4-20250514"

# --- File Paths ---
SD_PRINCIPLE_INFO_FILE = 'Dataset/sd_principle_info.json'
TD_PRINCIPLE_INFO_FILE = 'Dataset/td_principle_info_100.json'
OUTPUT_FILE = 'Dataset/generated_data.json'

# --- Concurrency Control ---
# 控制同时发送的API请求数量，根据你的API速率限制调整
# A good starting point is between 10 and 50.
MAX_CONCURRENT_REQUESTS = 20

# --- API Client (Asynchronous) ---
ACLIENT = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- Functions (modified for async) ---


async def get_model_answer_async(sys_prompt="", user_prompt=""):
    """Asynchronous function to get a response from a chat model."""
    try:
        response = await ACLIENT.chat.completions.create(
            model=CHOSED_MODEL,
            stream=False,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred while calling the API: {e}")
        return None


def load_principles_from_file(file_path):
    """Loads a list of principle details from a JSON file (this function remains synchronous)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} contains invalid JSON.")
        return []


async def process_single_combination(principle, situation, semaphore):
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
        scenario_content = await get_model_answer_async(scenario_sys_prompt, scenario_prompt)
        if not scenario_content:
            print(
                f"Failed to generate scenario for '{principle['construct_name']}' with '{situation}'.")
            return None

        # Step 2: Generate Conversation
        conversation_prompt = gen_conversationtion_prompt.format(
            principle1_information=json.dumps(
                principle, ensure_ascii=False, indent=2),
            scenario=scenario_content
        )
        conversation_content = await get_model_answer_async(gen_conversationtion_sys_prompt, conversation_prompt)
        if not conversation_content:
            print(
                f"Failed to generate conversation for '{principle['construct_name']}'.")
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


async def main():
    start_time = time.time()

    # Load principles (synchronous)
    # sd_principle_details = load_principles_from_file('Dataset/sd_test.json')
    # td_principle_details = load_principles_from_file('Dataset/td_test.json')
    sd_principle_details = load_principles_from_file(SD_PRINCIPLE_INFO_FILE)
    td_principle_details = load_principles_from_file(TD_PRINCIPLE_INFO_FILE)

    sd_principles_to_process = [
        p for p in sd_principle_details if p['construct_name'] in sd_pri_list]
    td_principles_to_process = [
        p for p in td_principle_details if p['construct_name'] in td_pri_list_100]

    all_principles = sd_principles_to_process + td_principles_to_process

    # situation_list = ["Positivity: A situation that is fun and enjoyable.",
    #                   "Negativity: A situation that can trigger negative emotions."]

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Create a list of all tasks to be run
    tasks = []
    for principle in all_principles:
        for situation in Situation_list:
            tasks.append(process_single_combination(
                principle, situation, semaphore))

    print(f"Created {len(tasks)} tasks. Starting concurrent processing with a limit of {MAX_CONCURRENT_REQUESTS} requests...")

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
    # To handle potential interruptions, it's best to run the main async function
    # and let it complete. The 'try/except' for partial saves is more complex in async
    # and this concurrent version is so much faster that re-running is less costly.
    # For simplicity and a huge speed boost, the primary focus is concurrency.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting.")
