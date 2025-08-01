

from openai import OpenAI
import json

CLAUDE_API="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"
BASE_URL="https://api.pumpkinaigc.online/v1"
API_KEY="sk-WKak50Ii5K68isoe7bF316D6E7Eb44A3Aa32843eBaE4866f"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 读取人类行为心理原则数据
def load_principles():
    with open('Dataset/principles.json', 'r', encoding='utf-8') as f:
        return json.load(f)

principles = load_principles()

response = client.chat.completions.create(
    model="claude-opus-4-20250514",
    # model="claude-sonnet-4-20250514",
    stream=False,
    messages=[
        {"role": "system", "content": "你是一位跨学科的智慧学者，完美融合了经验丰富的心理学家的洞察力、敏锐的社会学家的结构化视角、深邃的思想家的哲学思辨能力，以及进化生物学家对生命本源的理解。你的专长是穿透表象，揭示并阐释那些驱动人类行为、情感和决策的，如物理定律般普适而根本的运作法则（Human Principles）。你最擅长根据人类行为心理学原则创造真实、生动的训练场景。\n\n你的任务是：根据给定的人类行为心理原则，生成相应的具体场景，用于训练模型理解和应用这些原则。\n\n场景设计要求：\n1. 场景要真实可信，贴近日常生活\n2. 能够清晰体现所给原则的核心机制\n3. 包含具体的情境描述和可能的行为选择\n4. 格式应为一个完整的情境描述，以问题形式结尾\n\n示例格式：'你花了大价钱，费劲周章从国外买了一件后来发现并不完美（不是非常差，但是没有达到你的预期）的产品。朋友知道你为此付出代价的整个过程，当朋友询问你这个东西怎么样时，你会给出怎样的回答？'\n\n请为每个给定的原则生成2-3个不同角度的场景。"},
        {"role": "user", "content": f"请根据以下人类行为心理原则生成训练场景：\n\n原则名称：{principles[0]['principle']}\n\n核心描述：{principles[0]['description']}\n\n案例：{principles[0]['case']}\n\n请为这个原则生成2-3个不同的生活场景，每个场景都要能体现这个原则的核心机制。"}
    ]
)


print(response.choices[0].message.content)
