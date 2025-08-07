from openai import OpenAI
import json
import re
from rag_for_scene import add_scenario_to_memory, find_similar_scenarios
from prompt_scene import scene_sys_prompt, single_principle_role_prompt, multi_principles_role_prompt, scene_gen_prompt
from principle_list_dict import pri_list, pri_rel


BASE_URL = "https://api.pumpkinaigc.online/v1"
API_KEY = "sk-WKak50Ii5K68isoe7bF316D6E7Eb44A3Aa32843eBaE4866f"

PRIN_FILE = 'Dataset/principles_format.json'
SCENE_FILE = 'Dataset/scenes.json'
# CHOSED_MODEL="claude-opus-4-20250514"
CHOSED_MODEL = "claude-sonnet-4-20250514"
DOMAINS = ["职场与事业", "恋爱与婚姻", "家庭与亲情", "财务与投资", "健康与身心", "社交与友谊", "学习与成长"]


format_prompt = """你是一个专业的JSON格式化和修正工具。
你的任务是接收用户提供的可能格式错误的字符串，并将其修正为一个语法完全正确的、有效的JSON对象。

**核心修正规则：**
- 字符串值必须使用双引号 `"` 包围。
- 如果字符串内部包含双引号，直接删除内部的双引号。

**输出格式要求：**
- 你的回答**必须**只包含修正后的JSON对象本身。
- **禁止**输出任何解释性文字、开场白、结束语或任何非JSON内容。
- **禁止**使用Markdown代码块（例如 ```json）。
- 最终输出的必须是一个可以直接被任何标准JSON解析器成功解析的纯文本。"""


client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 读取人类行为心理原则数据


def load_principles(prin_file=PRIN_FILE):
    with open(prin_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_dict(res_content):
    # print(f"在提取json内容之前，模型的第一次回复是{res_content}\n")
    match = re.search(r'```json(.*)```', res_content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = res_content
    try:
        result_dict = json.loads(json_str)
        return result_dict
    except json.JSONDecodeError as e:
        # 如果解析失败，让大模型辅助修改
        response = client.chat.completions.create(
            model="claude-sonnet-4-20250514",
            stream=False,
            messages=[
                {"role": "system", "content": format_prompt},
                {"role": "user", "content": "JSON内容如下："+json_str},
            ],
            temperature=0.1,
        )
        return extract_dict(response.choices[0].message.content)


def generate_scenes_domain_1(domain):
    principles = load_principles()
    scenes = []

    for i, item in enumerate(principles):
        if i > 0:
            break
        exclusion_list = find_similar_scenarios(item, top_k=3, domain=domain)
        formatted_role_prompt = single_principle_role_prompt.format(
            primary_principle=item['principle'],
            p_principle_json=item,
            domain=domain,
            retrieved_summary_1=exclusion_list[0] if len(
                exclusion_list) > 0 else "无",
            retrieved_summary_2=exclusion_list[1] if len(
                exclusion_list) > 1 else "无",
            retrieved_summary_3=exclusion_list[2] if len(
                exclusion_list) > 2 else "无",
        )

        print(f"输入为："+scene_sys_prompt+formatted_role_prompt+"\n")

        response1 = client.chat.completions.create(
            model=CHOSED_MODEL,
            stream=False,
            messages=[
                {"role": "system", "content": scene_sys_prompt},
                {"role": "user", "content": formatted_role_prompt},
            ]
        )
        print(f"第一步生成的流畅故事：{response1.choices[0].message.content}\n")

        formatted_scene_gen_prompt = scene_gen_prompt.format(
            story=response1.choices[0].message.content,
            principleName=json.dumps([item['principle']], ensure_ascii=False)
        )

        response2 = client.chat.completions.create(
            model=CHOSED_MODEL,
            stream=False,
            messages=[
                {"role": "system", "content": "你是一位精通心理学、人类行为学和json格式化的专家"},
                {"role": "user", "content": formatted_scene_gen_prompt},
            ]
        )
        print(f"第二步生成的格式化json（抽取之前）：{response2.choices[0].message.content}\n")

        result1 = extract_dict(response2.choices[0].message.content)
        # print(result1)
        if result1 is None:
            print(f"{domain}领域单原则第{i+1}个原则生成scene失败: {item['principle']}\n")
            continue
        else:
            add_scenario_to_memory(result1, domain=domain)
            scenes.append(result1)
            print(f"{domain}领域单原则第{i+1}个原则生成scene成功（抽取之后）: {result1}\n")

        # response2 = client.chat.completions.create(
        #     model=CHOSED_MODEL,
        #     stream=False,
        #     messages=[
        #         {"role": "system", "content":single_principle_sys_prompt},
        #         {"role": "user", "content": role_prompt+f"{item}"},
        #     ]
        # )
        # result2= extract_dict(response2.choices[0].message.content)
        # if result2 is None:
        #     print(f"{domain}领域单原则第{i+1}个原则生成scene失败: {item['principle']}\n")
        #     continue
        # else:
        #     add_scenario_to_memory(result2)
        #     scenes.append(result2)
        #     print(f"{domain}领域单原则第{i+1}个原则生成scene成功: {result2}\n")

    return scenes


def generate_scenes_domain_2(domain):
    principles = load_principles()
    scenes = []

    with open('Dataset/principles_for_find.json', 'r', encoding='utf-8') as f:
        prin_content = json.load(f)

    for i, item in enumerate(principles):
        if i > 0:
            break
        print(f"当前原则的关联原则列表：{pri_rel[item['principle']]}\n")
        for vlaue in pri_rel[item['principle']]:
            exclusion_list = find_similar_scenarios(
                item, top_k=3, domain=domain)

            formatted_role_prompt = multi_principles_role_prompt.format(
                primary_principle=item['principle'],
                secondary_principle=vlaue,
                domain=domain,
                p_principle_json=item,
                s_principle_json=prin_content[vlaue],
                retrieved_summary_1=exclusion_list[0] if len(
                    exclusion_list) > 0 else "无",
                retrieved_summary_2=exclusion_list[1] if len(
                    exclusion_list) > 1 else "无",
                retrieved_summary_3=exclusion_list[2] if len(
                    exclusion_list) > 2 else "无",
            )

            print(f"输入为："+scene_sys_prompt+formatted_role_prompt+"\n")

            response1 = client.chat.completions.create(
                model=CHOSED_MODEL,
                stream=False,
                messages=[
                    {"role": "system", "content": scene_sys_prompt},
                    {"role": "user", "content": formatted_role_prompt},
                ]
            )
            print(f"第一步生成的流畅故事：{response1.choices[0].message.content}\n")

            print(f"第二步塞入json的原则为：{[item['principle'], vlaue]}")
            formatted_scene_gen_prompt = scene_gen_prompt.format(
                story=response1.choices[0].message.content,
                principleName=json.dumps(
                    [item['principle'], vlaue], ensure_ascii=False)
            )

            response2 = client.chat.completions.create(
                model=CHOSED_MODEL,
                stream=False,
                messages=[
                    {"role": "system", "content": "你是一位精通心理学、人类行为学和json格式化的专家"},
                    {"role": "user", "content": formatted_scene_gen_prompt},
                ]
            )

            print(
                f"第二步生成的格式化json（抽取之前）：{response2.choices[0].message.content}\n")

            result1 = extract_dict(response2.choices[0].message.content)

            if result1 is None:
                print(
                    f"{domain}领域多原则第{i+1}个原则生成scene失败: {item['principle']}\n")
                continue
            else:
                add_scenario_to_memory(result1, domain=domain)
                scenes.append(result1)
                print(f"{domain}领域多原则第{i+1}个原则生成scene成功(抽取之后): {result1}\n")

            # response2 = client.chat.completions.create(
            #     model=CHOSED_MODEL,
            #     stream=False,
            #     messages=[
            #         {"role": "system", "content":sys_prompt},
            #         {"role": "user", "content": role_prompt+f"{item}"},
            #     ]
            # )
            # result2= extract_dict(response2.choices[0].message.content)
            # if result2 is None:
            #     print(f"第{i+1}个原则生成scene失败: {item['principle']}")
            #     continue
            # else:
            #     scenes.append(result2)
            #     print(f"第{i+1}个原则生成scene成功: {result2}")

            # response3 = client.chat.completions.create(
            #     model=CHOSED_MODEL,
            #     stream=False,
            #     messages=[
            #         {"role": "system", "content":sys_prompt},
            #         {"role": "user", "content": role_prompt+f"{item}"},
            #     ]
            # )
            # result3= extract_dict(response3.choices[0].message.content)
            # if result3 is None:
            #     print(f"第{i+1}个原则生成scene失败: {item['principle']}")
            #     continue
            # else:
            #     scenes.append(result3)
            #     print(f"第{i+1}个原则生成scene成功: {result3}")
    return scenes


def generate_scenes():
    scenes = []
    for domain in DOMAINS:
        scenes.extend(generate_scenes_domain_1(domain))
        scenes.extend(generate_scenes_domain_2(domain))

    # 保存生成的场景到文件
    with open(SCENE_FILE, 'w', encoding='utf-8') as f:
        json.dump(scenes, f, ensure_ascii=False, indent=4)

    print(f"生成的场景已保存到{SCENE_FILE}。")
    print(f"总共生成了 {len(scenes)} 个场景。")


if __name__ == "__main__":
    generate_scenes()
