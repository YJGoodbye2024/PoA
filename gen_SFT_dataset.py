"""
把genered_data.json转换为SFT数据集格式
python gen_SFT_dataset.py --mode full 完整转换
python gen_SFT_dataset.py --mode split_only 仅仅把SFT_data_without_thoughts.json拆分为训练集和测试集
"""


import argparse
import asyncio
import json
import os
import random
import re
import time
from collections import defaultdict

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from prompt_all import gen_dataset_sys_prompt, gen_dataset_prompt, format_prompt_system, format_prompt_user

# --- 配置 ---
# 在这里定义您想使用的所有模型。每个键是一个自定义的别名。
MODELS = {
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat"
    },
    "pumpkin_gemini_pro": {
        "api_key": os.getenv("API_KEY_FULL"),
        "base_url": os.getenv("BASE_URL_FULL"),
        "model_name": "gemini-2.5-pro-preview-05-06"
    },
    "pumpkin_claude_sonnet": {
        "api_key": os.getenv("API_KEY_FULL"),
        "base_url": os.getenv("BASE_URL_FULL"),
        "model_name": "claude-4-5-sonnet-20250929"
    },
    "pumpkin_gpt": {
        "api_key": os.getenv("API_KEY_FULL"),
        "base_url": os.getenv("BASE_URL_FULL"),
        "model_name": "gpt-5"
    }
}

# --- ✅ 模型选择开关 ---
SFT_TRANSFORMATION_MODEL_PROFILE = "pumpkin_gpt"
JSON_FIXER_MODEL_PROFILE = "deepseek"

# --- 文件路径 ---
INPUT_FILE = 'Dataset/generated_data.json'
OUTPUT_FILE_CLEANED = 'Dataset/sft_data/SFT_data_without_thoughts.json'
OUTPUT_TRAIN = 'Dataset/sft_data/SFT_sharegpt_train.json'
OUTPUT_TEST = 'Dataset/sft_data/SFT_sharegpt_test.json'
SPLIT_RANDOM_SEED = 42

# --- 并发控制 ---
MAX_CONCURRENT_REQUESTS = 30


def load_source_data(file_path):
    """从JSON文件加载原始数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：输入文件 {file_path} 未找到。")
        return []
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 包含无效的JSON。")
        return []


async def fix_json_async(malformed_string, client, model):
    """使用LLM修复损坏的JSON字符串（对象或数组）"""
    try:
        user_prompt = format_prompt_user.format(
            malformed_string=malformed_string)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": format_prompt_system},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=8192
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n  -> ❌ 在尝试自动修复API调用时发生错误: {e}")
        return None


# --- MODIFICATION: Reworked the function to only get the conversation turns ---
async def transform_conversation_to_turns_async(data_item, semaphore, sft_client, sft_model, fixer_client, fixer_model):
    """
    使用LLM将`conversation`文本转换为SFT的`human`/`assistant`轮次数组。
    """
    async with semaphore:
        try:
            prompt = gen_dataset_prompt.format(
                scenario=data_item.get("scenario", ""),
                conversation=data_item.get("conversation", "")
            )
            response = await sft_client.chat.completions.create(
                model=sft_model,
                messages=[
                    {"role": "system", "content": gen_dataset_sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8192
            )
            raw_json_string = response.choices[0].message.content
            cleaned_string = raw_json_string.strip()
            match = re.search(
                r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned_string, re.DOTALL | re.IGNORECASE)
            json_to_parse = match.group(1).strip() if match else cleaned_string

            try:
                # 期望得到的是一个列表 (JSON array)
                parsed_array = json.loads(json_to_parse)
                if isinstance(parsed_array, list):
                    # 成功，返回原始数据项和转换后的轮次列表
                    return data_item, parsed_array
                else:
                    print(
                        f"\n警告：模型未返回预期的列表格式 (原则: '{data_item.get('principle')}')... 正在尝试修复...")
                    # 即使格式不是列表，也尝试修复
                    raise json.JSONDecodeError(
                        "Model did not return a list.", json_to_parse, 0)
            except json.JSONDecodeError:
                # 修复逻辑现在也在这里处理
                corrected_string = await fix_json_async(json_to_parse, fixer_client, fixer_model)
                if corrected_string:
                    try:
                        parsed_array = json.loads(corrected_string)
                        if isinstance(parsed_array, list):
                            print(
                                f"  -> ✓ 自动修复成功 (原则: '{data_item.get('principle')}')")
                            return data_item, parsed_array
                    except json.JSONDecodeError:
                        pass  # 最终失败
                print(f"  -> ❌ 修复后依然无法解析为列表。已跳过该条目。")
                return data_item, None
        except Exception as e:
            print(f"\nAPI调用时发生错误 (原则: '{data_item.get('principle')}'): {e}")
            return data_item, None


def save_data_to_json(data, file_path):
    """将转换后的SFT数据保存到JSON文件"""

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n数据已成功保存至 {file_path}")
    except IOError as e:
        print(f"写入文件时发生错误: {e}")


def strip_metadata(dataset):
    """移除训练/测试输出中不需要的元数据字段。"""
    stripped = []
    for item in dataset:
        stripped.append({
            "system": item.get("system", ""),
            "conversations": item.get("conversations", []),
        })
    return stripped


def clean_human_thoughts(sft_data):
    """清洗'human'轮次中的内心独白"""
    data_to_clean = json.loads(json.dumps(sft_data))
    thought_pattern = re.compile(r'\[.*?\]', re.DOTALL)
    for item in data_to_clean:
        for turn in item.get("conversations", []):
            if turn.get("from") == "human":
                original_value = turn.get("value", "")
                cleaned_value = thought_pattern.sub('', original_value)
                lines = [line.strip()
                         for line in cleaned_value.split('\n') if line.strip()]
                cleaned_value = '\n'.join(lines)
                if cleaned_value:
                    cleaned_value += '\n\n'
                turn['value'] = cleaned_value
    return strip_metadata(data_to_clean)


def split_dataset_by_principle_situation(dataset, seed=SPLIT_RANDOM_SEED):
    """根据principle随机抽取一个situation作为测试集，其余为训练集。"""
    rng = random.Random(seed)
    principle_to_situations = defaultdict(lambda: defaultdict(list))

    for item in dataset:
        principle = item.get("principle", "")
        situation = item.get("situation", "")
        principle_to_situations[principle][situation].append(item)

    train, test = [], []
    for principle, situations in principle_to_situations.items():
        situation_keys = list(situations.keys())
        if not situation_keys:
            continue
        chosen_situation = rng.choice(situation_keys)
        for situation, grouped_items in situations.items():
            if situation == chosen_situation:
                test.extend(grouped_items)
            else:
                train.extend(grouped_items)

    return train, test


async def run_full_pipeline():
    start_time = time.time()

    # 初始化客户端和模型
    try:
        sft_config = MODELS[SFT_TRANSFORMATION_MODEL_PROFILE]
        fixer_config = MODELS[JSON_FIXER_MODEL_PROFILE]
        sft_client = AsyncOpenAI(
            api_key=sft_config["api_key"], base_url=sft_config["base_url"])
        sft_model_name = sft_config["model_name"]
        fixer_client = AsyncOpenAI(
            api_key=fixer_config["api_key"], base_url=fixer_config["base_url"])
        fixer_model_name = fixer_config["model_name"]
        print(
            f"--- SFT Transformation Model: {SFT_TRANSFORMATION_MODEL_PROFILE} ({sft_model_name}) ---")
        print(
            f"--- JSON Fixer Model: {JSON_FIXER_MODEL_PROFILE} ({fixer_model_name}) ---")
    except KeyError as e:
        print(f"错误：未找到名为 '{e.args[0]}' 的模型配置。请检查 MODELS 字典。")
        return

    source_data = load_source_data(INPUT_FILE)
    if not source_data:
        print("无数据可处理，脚本退出。")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [transform_conversation_to_turns_async(
        item, semaphore, sft_client, sft_model_name, fixer_client, fixer_model_name
    ) for item in source_data]
    print(f"\n已创建 {len(tasks)} 个转换任务。开始并发处理...")

    sft_dataset = []
    try:
        future_tasks = asyncio.as_completed(tasks)
        for future in tqdm(future_tasks, total=len(tasks), desc="Converting conversations to turns"):
            # --- MODIFICATION: Assemble the final JSON object here in the main loop ---
            original_item, turns_array = await future

            if turns_array is not None:
                # Python script handles the static parts
                conversations = [dict(turn) for turn in turns_array]

                final_sft_object = {
                    "system": original_item.get("scenario", ""),
                    "conversations": conversations
                }
                sft_dataset.append(final_sft_object)

    except (KeyboardInterrupt, asyncio.CancelledError) as e:
        print(f"\n\n! 脚本被中断: {e}")
        print("! 正在尝试保存已处理的部分数据...")
    finally:
        if sft_dataset:
            is_partial = len(sft_dataset) < len(
                source_data)  # Compare with source data
            print("\n正在清洗配角的内心独白...")
            cleaned_dataset = clean_human_thoughts(sft_dataset)
            print("清洗完成。")

            output_cleaned = OUTPUT_FILE_CLEANED.replace(
                '.json', '_partial.json') if is_partial else OUTPUT_FILE_CLEANED
            save_data_to_json(cleaned_dataset, output_cleaned)

            train_dataset, test_dataset = split_dataset_by_principle_situation(
                cleaned_dataset, seed=SPLIT_RANDOM_SEED
            )

            train_output = OUTPUT_TRAIN.replace(
                '.json', '_partial.json') if is_partial else OUTPUT_TRAIN
            test_output = OUTPUT_TEST.replace(
                '.json', '_partial.json') if is_partial else OUTPUT_TEST

            save_data_to_json(train_dataset, train_output)
            save_data_to_json(test_dataset, test_output)

        else:
            print("没有成功转换的数据可供保存。")

        end_time = time.time()
        print(f"\n脚本在 {end_time - start_time:.2f} 秒内完成。")
        print(
            f"成功转换并保存了 {len(sft_dataset)} / {len(source_data)} 条数据。")


def split_existing_dataset():
    dataset = load_source_data(OUTPUT_FILE_CLEANED)
    if not dataset:
        print("无法读取已清洗的 SFT 数据，无法拆分训练/测试集。")
        return

    train_dataset, test_dataset = split_dataset_by_principle_situation(
        dataset, seed=SPLIT_RANDOM_SEED
    )

    save_data_to_json(strip_metadata(train_dataset), OUTPUT_TRAIN)
    save_data_to_json(strip_metadata(test_dataset), OUTPUT_TEST)

    print(
        f"\n已根据 {OUTPUT_FILE_CLEANED} 生成训练集 ({len(train_dataset)} 条) "
        f"和测试集 ({len(test_dataset)} 条)。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成或拆分 SFT 数据集。")
    parser.add_argument(
        "--mode",
        choices=["full", "split_only"],
        default="full",
        help="full: 执行完整的生成、清洗与拆分流程；split_only: 使用现有的 SFT_data_without_thoughts.json 仅生成训练/测试集。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "split_only":
        split_existing_dataset()
    else:
        asyncio.run(run_full_pipeline())


if __name__ == "__main__":
    main()
