from openai import OpenAI
import json
import re
import os
import time

from prompt_all import principle_info_prompt, format_prompt, gen_relationship_prompt_system, gen_relationship_prompt
from principle_situaton import pri_list


BASE_URL = "https://api.pumpkinaigc.online/v1"
API_KEY = "sk-WKak50Ii5K68isoe7bF316D6E7Eb44A3Aa32843eBaE4866f"

CHOSED_MODEL = "gemini-2.5-pro-preview-05-06"
CHOSED_MODEL_2 = "claude-sonnet-4-20250514"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def get_model_answer(client=client, model=CHOSED_MODEL, sys_prompt="", user_prompt=""):
    response = client.chat.completions.create(
        model=CHOSED_MODEL,
        stream=False,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.choices[0].message.content


def gen_prin_rela(principle_list=pri_list, output_filename='Dataset/relationships.json'):
    full_principle_list_str = ", ".join(principle_list)
    principle_set = set(principle_list)
    all_relationships = {}
    failed_principles = []

    # 正则表达式捕获realted principle的名称
    regex_pattern = re.compile(r"^Related Principle:\s*(.*)", re.MULTILINE)

    for prin in principle_list:
        try:
            # ---- Step 1: Call Model A for Analysis ----
            res_content = get_model_answer(sys_prompt=gen_relationship_prompt_system, user_prompt=gen_relationship_prompt.format(
                target_principle=prin,
                principle_list=full_principle_list_str
            ))
            print(f"\n模型分析 '{prin}' 的结果:\n{res_content}\n")
            # ---- Step 2: Use Regex to Extract Principles ----
            matches = regex_pattern.findall(res_content)

            # 清理每个匹配项，去除可能存在的前后空格
            # 只有当清理后的原则名称确实存在于`principle_set`中时，才将其加入列表。
            related_principles = [
                name.strip().lower() for name in matches if name.strip().lower() in principle_set
            ]

            # ---- Step 3: Store the Result ----
            all_relationships[prin] = related_principles

            # ---- Step 4: Rate Limiting ----
            time.sleep(1)  # 每次API调用后暂停1秒

        except Exception as e:
            print(
                f"An unexpected error occurred while processing '{prin}': {e}")
            failed_principles.append(prin)
            continue

    # 在函数处理完成后，将 all_relationships 字典保存到文件中

    print(f"\n所有原则处理完毕。正在将结果保存到文件 '{output_filename}'...")

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            # indent=2 使JSON文件格式化，易于阅读
            # ensure_ascii=False 确保非ASCII字符（如中文）能正确写入
            json.dump(all_relationships, f, indent=2, ensure_ascii=False)
        print(f"成功将结果保存到 '{output_filename}'。")
    except IOError as e:
        print(f"错误：无法将结果写入文件。原因: {e}")

    return all_relationships, failed_principles


def extract_dict(res_content, retries=2):
    """
    Extracts a dictionary from a string, with a limited number of retries for fixing format.
    """
    if retries <= 0:
        raise ValueError("Failed to parse JSON after multiple retries.")

    match = re.search(r'```json(.*)```', res_content, re.DOTALL)
    json_str = match.group(1).strip() if match else res_content

    try:
        result_dict = json.loads(json_str)
        return result_dict
    except json.JSONDecodeError as e:
        print("Initial JSON parsing failed. Asking model to fix format...")
        response = client.chat.completions.create(
            model="claude-sonnet-4-20250514",
            stream=False,
            messages=[
                {"role": "system", "content": format_prompt},
                {"role": "user", "content": "JSON content to fix: " + json_str},
            ],
            temperature=0.1,
        )
        # Recursive call with decremented retry counter
        return extract_dict(response.choices[0].message.content, retries - 1)


def gen_principle_info(principle_list=pri_list, output_file="Dataset/principle_info.json"):
    all_principles_info = []
    for i, prin_name in enumerate(principle_list):
        principle_info_prompt_format = principle_info_prompt.format(
            PRINCIPLE_NAME=prin_name,
        )
        response_content = get_model_answer(
            sys_prompt="You are a psychologist.", user_prompt=principle_info_prompt_format)

        print(
            f"Processing {i+1}/{len(principle_list)}...\n principle信息：{response_content}\n")
        # 格式化输出为json
        result = extract_dict(response_content)
        all_principles_info.append(result)
        time.sleep(1)  # API速率控制

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_principles_info, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":

    # 生成relationship.json文件
    # gen_prin_rela()

    # 生成principle_info.json文件
    gen_principle_info(principle_list=[
                       'loss aversion'], output_file="Dataset/principle_info_test.json")
