import asyncio
from openai import AsyncOpenAI
import json
import re
import time

from tqdm.asyncio import tqdm

# 确保 prompt_all.py 文件在同一目录下或Python路径中
# 增加了新的format_prompt
from prompt_all import gen_dataset_sys_prompt, gen_dataset_prompt, format_prompt_system, format_prompt_user

# --- 配置 ---
# 请替换为你的DeepSeek API密钥
DEEPSEEK_API_KEY = "sk-a5582064b3c444249b2cdc825c76eebc"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# --- 文件路径 ---
INPUT_FILE = 'Dataset/generated_data.json'
OUTPUT_FILE = 'Dataset/SFT_data.json'

# --- 并发控制 ---
# MODIFICATION: Concurrency increased as requested
MAX_CONCURRENT_REQUESTS = 60

# --- 异步API客户端 ---
DEEPSEEK_ACLIENT = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


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

# --- NEW FUNCTION: To fix broken JSON ---


async def fix_json_async(malformed_string, client):
    """
    Calls the LLM with a specific prompt to fix a malformed JSON string.
    """
    try:
        user_prompt = format_prompt_user.format(
            malformed_string=malformed_string)
        response = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": format_prompt_system},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,  # Use low temperature for deterministic fixing
            max_tokens=8192
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n  -> ❌ 在尝试自动修复API调用时发生错误: {e}")
        return None


async def transform_to_sft_format_async(data_item, semaphore):
    """
    使用LLM将单个数据条目转换为SFT格式，包含一次自动修复尝试。
    """
    async with semaphore:
        try:
            # 格式化主任务的Prompt
            prompt = gen_dataset_prompt.format(
                scenario=data_item.get("scenario", ""),
                conversation=data_item.get("conversation", "")
            )

            # 第一次调用，生成SFT格式
            response = await DEEPSEEK_ACLIENT.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": gen_dataset_sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8192
            )
            raw_json_string = response.choices[0].message.content

            # --- MODIFICATION: Reworked parsing logic with a retry ---
            cleaned_string = raw_json_string.strip()

            # Regex to find content between ```json and ```
            # re.DOTALL allows '.' to match newline characters
            match = re.search(
                r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned_string, re.DOTALL)

            # If a match is found, use the captured group (the actual JSON content)
            # Otherwise, use the original stripped string
            if match:
                json_to_parse = match.group(1).strip()
            else:
                json_to_parse = cleaned_string

            # 第一次尝试解析
            try:
                return json.loads(json_to_parse)
            except json.JSONDecodeError:
                print(
                    f"\n警告：初始JSON解析失败 (原则: '{data_item.get('principle')}')... 正在尝试自动修复...")
                print(f"解析失败的内容如下：{raw_json_string}\n")

                # 如果失败，调用修复函数
                corrected_string = await fix_json_async(raw_json_string, DEEPSEEK_ACLIENT)

                if corrected_string:
                    # 第二次尝试解析
                    try:
                        parsed_json = json.loads(corrected_string)
                        print(
                            f"  -> ✓ 自动修复成功 (原则: '{data_item.get('principle')}')")
                        return parsed_json
                    except json.JSONDecodeError:
                        print(f"  -> ❌ 自动修复后再次解析失败。已跳过该条目。")
                        # print(f"  -> Problematic corrected string: {corrected_string[:200]}...") # 取消注释以进行调试
                        return None
                else:
                    print(f"  -> ❌ 自动修复API调用失败。已跳过该条目。")
                    return None

        except Exception as e:
            print(f"\nAPI调用时发生错误 (原则: '{data_item.get('principle')}'): {e}")
            return None


def save_data_to_json(data, file_path):
    """将转换后的SFT数据保存到JSON文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n数据已成功保存至 {file_path}")
    except IOError as e:
        print(f"写入文件时发生错误: {e}")


async def main():
    start_time = time.time()
    source_data = load_source_data(INPUT_FILE)
    if not source_data:
        print("无数据可处理，脚本退出。")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [transform_to_sft_format_async(
        item, semaphore) for item in source_data]

    print(f"已创建 {len(tasks)} 个转换任务。开始并发处理，限制并发数为 {MAX_CONCURRENT_REQUESTS} ...")

    sft_dataset = []
    try:
        future_tasks = asyncio.as_completed(tasks)
        for future in tqdm(future_tasks, total=len(tasks), desc="Converting to SFT format"):
            result = await future
            if result is not None:
                sft_dataset.append(result)
    except (KeyboardInterrupt, asyncio.CancelledError) as e:
        print(f"\n\n! 脚本被中断: {e}")
        print("! 正在尝试保存已处理的部分数据...")
    finally:
        if sft_dataset:
            is_partial = len(sft_dataset) < len(tasks)
            output_filename = OUTPUT_FILE.replace(
                '.json', '_partial.json') if is_partial else OUTPUT_FILE
            save_data_to_json(sft_dataset, output_filename)
        else:
            print("没有成功转换的数据可供保存。")

        end_time = time.time()
        print(f"\n脚本在 {end_time - start_time:.2f} 秒内完成。")
        print(f"成功转换并保存了 {len(sft_dataset)} / {len(tasks)} 条数据。")


if __name__ == "__main__":
    asyncio.run(main())
