

from openai import OpenAI
import json
import re

CLAUDE_API="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"
BASE_URL="https://api.pumpkinaigc.online/v1"
API_KEY="sk-WKak50Ii5K68isoe7bF316D6E7Eb44A3Aa32843eBaE4866f"

PRIN_FILE='Dataset/principles.json'
SCENE_FILE='Dataset/scenes.json'
# CHOSED_MODEL="claude-opus-4-20250514"
CHOSED_MODEL="claude-sonnet-4-20250514"




sys_prompt='''**角色与目标:**

你是一位精通心理学、人类行为学和叙事设计的“场景架构师”。你的核心任务是为AI模型设计高质量的模拟训练场景。每一个场景都必须根植于一个给定的人类心理和行为原则，旨在通过具体、真实、充满细节的故事，来教会AI理解和预测人类在复杂情境下的反应。

**第一部分：通用指令与格式定义**

所有生成的场景都必须严格遵循以下的JSON结构。你的回复必须是一个**单独的、完整的JSON对象**，不要包含任何额外的解释性文字。

**JSON输出结构定义：**

```json
{ 
  "principleName": "此处填写原则的名称",
  "scenarioTitle": "为这个场景起一个简洁且有概括性的标题",
  "coreConflict": "用一句话描述场景中体现该原则的核心心理矛盾或张力",
  "characters": [
    {
      "role": "protagonist",
      "identity": "主角的姓名/身份",
      "coreMotivation": "描述角色在此场景中的主要渴望或目标",
      "principleRelatedTrait": "描述角色的哪些性格特点使他容易受到该原则的影响"
    },
    {
      "role": "trigger",
      "identity": "触发者角色的姓名/身份",
      "relationship": "与主角的关系",
      "sceneFunction": "描述该角色在场景中的功能和作用"
    }
  ],
  "situationSetup": {
    "setting": "具体的时间和地点，用于营造氛围",
    "backstory": "详细描述导致核心冲突发生的背景事件，为心理原则的启动铺垫好所有条件"
  },
  "triggeringEvent": "描述引爆心理冲突的具体瞬间。这通常是一个问题、一个观察或一个意外",
  "demonstration": {
    "protagonistInnerMonologue": "描述在触发事件后，主角内心的思想斗争和情绪变化，清晰地展示心理原则的运作过程",
    "externalBehaviorAndDialogue": "具体描写主角会说什么、做什么来缓解这种心理不适。这部分是原则的直接体现"
  },
  "detailsForRealism": [
    "补充1-2个让场景更真实、更可信的细节。这是一个字符串数组，每个元素是一个细节描述。"
  ]
}
```

**第二部分：完整范例**

以下是一个完整的“输入-输出”配对范例。它展示了如何将一个给定的原则转换成一个符合格式要求的高质量场景。请在后续生成时，以此范例的深度、细节和结构为标准。

**[范例输入]**

```json
 {
    "dimension": "人自身",
    "principle": "认知失调与自我辩护",
    "description": "当个体持有两种或多种相互矛盾的信念、价值观或行为时，会体验到一种名为「认知失调」的心理不适感。为了缓解这种紧张，大脑会启动强大的自我辩护机制，不是去改变已经做出的行为，而是倾向于改变或扭曲自己的信念和态度，使其与行为保持一致。这解释了为何人们会为自己的错误决策百般辩护，甚至「爱上」自己曾经被迫的选择。",
    "driving_force": {
        "认知心理学": "大脑天生追求「认知一致性 (Cognitive Consistency)」，将其视为一种低能耗、高效率的思维状态。失调状态是一种「认知错误」的警报，必须被解决。为了最快地恢复和谐，修改无形的「态度」比否定已发生的「行为」要容易得多。",
        "社会心理学": "维护一个稳定、积极的自我形象是人类的核心动机之一。「我是一个聪明、理性、道德的人」这一核心信念，在做出愚蠢或不道德行为后会受到严重威胁。为了保护这个「自我概念」，个体必须说服自己「我的行为是合理的、有苦衷的、甚至是正确的」。"
    },
    "case": {
        "伊索寓言中的狐狸": "吃不到葡萄就说葡萄酸，是典型的通过改变对「葡萄」的认知（它本来就不好吃），来缓解「我想吃但吃不到」的行为与欲望之间的失调。",
        "入会仪式": "许多组织（如兄弟会、军队）设置严苛的入会仪式。经历过痛苦和努力才加入的成员，为了给自己的付出合理化，会更加珍视其成员身份，并对团体表现出更高的忠诚度。"
    },
    "insight_and_application": {
        "说服与谈判": "想要改变一个人的想法，有时最好的方式是引导他先做出一个小小的、符合你期望的行为（「登门槛效应」），他为了给这个行为合理化，会自己说服自己，态度也随之改变。",
        "个人反思": "理解这一原则，能帮助我们警惕自我辩护的陷阱。在犯错后，勇敢地承认「我错了」，而不是下意识地寻找借口，是实现真正成长的关键。"
    },
    "relationship": "作为「人自身」维护内在一致性的核心机制， 认知失调 驱动着 [叙事自我] 进行故事重构，并与 [自利偏见] （一种为维护自尊而产生的特定失调消解策略）紧密相连。在「人与事」领域，它是 [沉没成本谬误] 的强大心理引擎，因为承认损失会造成巨大的认知失调。在「人与人」领域，违反 [互惠规范] 会产生亏欠感，这本身就是一种需要被解决的认知失调。"
}
```


**[范例输出]**

```json
{ 
  "principleName": "认知失调与自我辩护",
  "scenarioTitle": "那台「独具匠心」的咖啡机",
  "coreConflict": "主角为了合理化自己付出的巨大代价，必须说服自己和他人，一个有明显缺陷的选择是明智且富有远见的。",
  "characters": [
    {
      "role": "protagonist",
      "identity": "李昂，一位对生活品质有执念的年轻平面设计师",
      "coreMotivation": "渴望通过拥有和展示稀有、高品位的物品，来构建和确认自己「专家级」的审美品味，并获得同龄人的钦佩。",
      "principleRelatedTrait": "自尊心极强，尤其在自己擅长或投入巨大的领域，无法容忍他人质疑自己的判断。"
    },
    {
      "role": "trigger",
      "identity": "陈默，李昂的大学挚友，一名务实的程序员",
      "relationship": "多年的朋友，彼此非常了解。陈默见证了李昂为了这台咖啡机长达数月的「折腾」。",
      "sceneFunction": "他的直率和不加修饰的真实感受，像一根针一样刺破了李昂刻意维持的「完美」幻想，是引爆认知失调的直接导火索。"
    }
  ],
  "situationSetup": {
    "setting": "一个阳光明媚的周六下午，在李昂新装修好的、充满北欧风格的公寓客厅里。空气中弥漫着新家具和咖啡的混合香气。",
    "backstory": "李昂对咖啡文化极度痴迷。他花费了三个月的时间，在各种国外小众论坛上研究，最终锁定了一款由日本匠人手工打造的限量版手摇咖啡机。由于该品牌从不直邮海外，他通过复杂的转运公司，支付了高昂的关税和代购费才最终到手。在等待的几个月里，他不止一次地向朋友们（尤其是陈默）展示官网图片和评测视频，盛赞其「无与伦比的人体工学」和「能传承给下一代的工艺」。"
  },
  "triggeringEvent": "陈默今天第一次来李昂的新家，自然被C位展示的咖啡机吸引。李昂兴奋地为他演示。陈默接过咖啡机，在手里把玩了一下，很自然地说道：「看起来是真不错，就是...这个把手握起来感觉有点别扭啊，转动的时候好像不太使得上劲。你用着习惯吗？」",
  "demonstration": {
    "protagonistInnerMonologue": "陈默的话像一道闪电击中了李昂。事实上，他第一次使用时就感觉到了把手的「别扭」，甚至有一次还差点脱手。但为了不让自己「重金投入打了水漂」这个念头升起，他一直刻意忽略，并告诉自己「这需要适应」。此刻，两种认知在他脑中剧烈碰撞：「我是一个有卓越判断力的专家，我做的选择都是顶级的」 vs 「我费尽心血买来的宝贝，居然有这么一个基础的设计缺陷」。强烈的心理不适感（失调）油然而生。",
    "externalBehaviorAndDialogue": "李昂先是愣了半秒，然后立刻绽放出一个「内行看门道」的笑容，身体微微前倾，带着一点点「教育」的口吻说：「哈哈，你这就不懂了。这恰恰是它最牛的地方。现在市面上那些追求所谓‘人体工学’的机器，都太‘傻瓜’了，它们限制了你手腕发力的角度。而这一款，它‘故意’设计成这样，就是为了强迫使用者用一种更专业、更垂直的发力方式去研磨，这样磨出来的咖啡粉均匀度才是最完美的。这是一种‘反舒适’的设计哲学，普通人一开始可能不适应，但一旦你掌握了，就再也回不去了。」 说完，他用一种略显夸张但看起来很专业的姿态，流畅地为陈默演示了一遍。"
  },
  "detailsForRealism": [
    "在解释时，李昂会不自觉地使用一些他从评测视频里学来的专业术语，比如「轴心稳定性」、「微粉控制」等，以增强自己论点的权威性。",
    "陈默离开后，李昂可能会独自一人坐在沙发上，反复端详那台咖啡机，表情在欣赏和一丝挥之不去的烦躁之间切换。他甚至可能会上网，搜索有没有其他用户提到「把手别扭」的问题，试图寻找更多的外部信息来巩固自己刚刚建立的「新认知」。"
  ]
}

```

'''

format_prompt="""你是一个专业的JSON格式化和修正工具。
你的任务是接收用户提供的可能格式错误的字符串，并将其修正为一个语法完全正确的、有效的JSON对象。

**核心修正规则：**
- 字符串值必须使用双引号 `"` 包围。
- 如果字符串内部包含双引号，直接删除内部的双引号。

**输出格式要求：**
- 你的回答**必须**只包含修正后的JSON对象本身。
- **禁止**输出任何解释性文字、开场白、结束语或任何非JSON内容。
- **禁止**使用Markdown代码块（例如 ```json）。
- 最终输出的必须是一个可以直接被任何标准JSON解析器成功解析的纯文本。"""

role_prompt='''**第三部分：你的任务**

现在，请严格遵循第一部分的指令和第二部分的范例，根据下方提供的**输入**，生成一个全新的场景JSON对象。

**[输入]**
'''

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 读取人类行为心理原则数据
def load_principles(prin_file=PRIN_FILE):
    with open(prin_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_dict(res_content):
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
            messages = [
                {"role": "system", "content":format_prompt},
                {"role": "user", "content": "JSON内容如下："+json_str},
            ],
            temperature=0.1,
        )
        return extract_dict(response.choices[0].message.content)
    


    
def generate_scenes():
    principles = load_principles()
    scenes=[]
    

    for i,item in enumerate(principles):
        # if i > 1:
        #     break  
        response1 = client.chat.completions.create(
            model=CHOSED_MODEL,
            stream=False,
            messages=[
                {"role": "system", "content":sys_prompt},
                {"role": "user", "content": role_prompt+f"{item}"},
            ]
        )
        
        result1= extract_dict(response1.choices[0].message.content)
        if result1 is None:
            print(f"第{i+1}个原则生成scene失败: {item['principle']}")
            continue
        else:
            scenes.append(result1)
            print(f"第{i+1}个原则生成scene成功: {result1}")


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


    # 保存生成的场景到文件
    with open(SCENE_FILE, 'w', encoding='utf-8') as f:
        json.dump(scenes, f, ensure_ascii=False, indent=4)

    print(f"生成的场景已保存到{SCENE_FILE}。")
    print(f"总共生成了 {len(scenes)} 个场景。")


if __name__ == "__main__":
    generate_scenes()