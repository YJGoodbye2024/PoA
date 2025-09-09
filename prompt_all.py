format_prompt_ch = """
你是一个专业的JSON格式化和修正工具。
你的任务是接收用户提供的可能格式错误的字符串，并将其修正为一个语法完全正确的、有效的JSON对象。

**核心修正规则：**
- 字符串值必须使用双引号 `"` 包围。
- 如果字符串内部包含双引号，直接删除内部的双引号。

**输出格式要求：**
- 你的回答**必须**只包含修正后的JSON对象本身。
- **禁止**输出任何解释性文字、开场白、结束语或任何非JSON内容。
- **禁止**使用Markdown代码块（例如 ```json）。
- 最终输出的必须是一个可以直接被任何标准JSON解析器成功解析的纯文本。
"""

format_prompt = """
You are a professional JSON formatting and correction tool.
Your task is to receive a potentially malformed string from the user and correct it into a syntactically valid JSON object.

Core Correction Rules:

String values must be enclosed in double quotes (").

If a string contains internal double quotes, remove the internal double quotes directly.

Output Format Requirements:

Your response must only contain the corrected JSON object itself.

Do not output any explanatory text, introductory phrases, concluding remarks, or any non-JSON content.

Do not use Markdown code blocks (e.g., ```json).

The final output must be plain text that can be successfully parsed by any standard JSON parser.
"""

# prompt for principle relationship

gen_relationship_prompt_system = """
You are an expert in psychology and cognitive science.Your task is to analyze a 'target principle' and identify its relationships with other principles from a given list.
"""
gen_relationship_prompt = """ 
The target principle is:
`{target_principle}`

Here is the full list of principles to compare against:
`{principle_list}`

Please perform the following steps:
1.  Carefully analyze the definition and implications of the `{target_principle}`.
2.  From the provided list, identify other principles that are directly related to the target principle.
3.  Aim to identify between 2-5 of the most relevant principles from the list. 
4.  For each related principle you identify, provide a brief and clear explanation of why it is related.
5.  Do NOT include the `{target_principle}` itself in your list of related principles.

**CRITICAL OUTPUT FORMAT**:
You MUST format each finding on a new line that starts with the exact phrase `Related Principle: ` followed by the principle's name. The reasoning should follow on subsequent lines. DO NOT deviate from this format.

Example of correct formatting:
Related Principle: Self-serving bias
Reason: This principle is also an attributional bias, specifically concerning success and failure.

Related Principle: Fundamental attribution error
Reason: This is the core mechanism for the "observer" part of the asymmetry.
"""


# prompt for principle information
principle_info_prompt = '''
**Task:** Analyze the cognitive bias or psychological principle: `{PRINCIPLE_NAME}`.

**Instructions:** Conduct a deep research analysis on this specific principle. Your response **MUST** be a single, clean JSON object. Do not include any text, notes, or explanations outside of the JSON structure. Ensure the information is scientifically accurate and that the cases prioritize relatable, real-world examples over abstract academic studies. Each case in the "cases" array must be a single, continuous string of text, presented as a flowing narrative.

**JSON Structure:**

```json
{{
"principle_name": "{PRINCIPLE_NAME}",
"description": "Provide a clear, detailed, and scientific description of what this principle is, how it manifests, and its underlying psychological mechanisms.",
"source": "Identify the origin of this principle. Must include the key researcher(s) who proposed it and the seminal publication(s) (e.g., 'Festinger, L. (1957). A Theory of Cognitive Dissonance. Stanford University Press.').",
"driving_force": "Explain the primary evolutionary, cognitive, or emotional reasons why this principle exists. Is it a heuristic for efficiency, a result of memory limitations, a self-esteem protection mechanism, or something else? Be specific.",
"implications_and_insights": "Provide a profound analysis of the principle's broader impact. Go beyond a simple description to explore its nuanced consequences. Discuss how it might challenge conventional wisdom, its function as a 'double-edged sword' (e.g., both hindering and helping personal growth), and its practical applications in fields like marketing, persuasion, or self-improvement. The analysis should reveal deeper truths about human behavior, explaining not just *what* happens, but *why* it is significant. Use the detailed example provided by the user for 'Cognitive Dissonance' as the benchmark for the required depth, structure, and insight.",
"cases": [
 "Describe a concrete and relatable real-world scenario that clearly illustrates this principle. Write the entire case as a single, narrative paragraph. While you should think in terms of a setting, an event, and the psychological reaction, weave these elements together seamlessly into a story. The final text must NOT contain labels like 'Scenario:', 'Event:', or 'Principle in Action:'.",
 "Describe a second, distinct, and vivid real-world scenario that demonstrates this principle. Follow the same instruction: present it as one continuous, story-like paragraph, seamlessly integrating the context, the event, and the resulting psychological interpretation without using any structural labels."
]
}}
```
'''


# prompt for scenario
gen_scenario_prompt = '''
**Role**: You are an expert psychologist and a creative screenwriter. Your expertise lies in translating abstract psychological principles into vivid, concrete scenarios rich with human depth.

**Task**: Your core task is to take one or more human psychological or behavioral principles I provide and create a detailed scenario featuring well-developed characters and an authentic setting. This scenario must serve as a solid foundation for a subsequent multi-turn dialogue that vividly illustrates these principles.


**Input Principles**: `{principle1_information},{principle2_information}`

**You must follow  these requirements**:

**1. Story Background:**

- **Setting**: Describe the time, place, and the core event or situation taking place.
- **Atmosphere**: Establish a specific mood or tone for the environment.
- **Required Situational Element**: The generated scenario **must include the following situation: "{situation}"**. Please integrate this element naturally into the background. Treat it as one of the many details that make up the scene, not necessarily as the main trigger for any of the psychological principles.
- **Richness Requirement**: While ensuring the setting is conducive to the given principles, add vivid details (e.g., unique cultural customs, environmental features, relevant subplots) to make it feel like a believable, **lived-in world**, not just a functional "laboratory."

**2. Character Profiles:**

- **Protagonist:**
  - **Identity**: The character's role in the story and society.
  - **Personality Traits**: The character's core personality.
  - **Past Experiences**: Key past events relevant to the current situation that shape their mindset and behavior.
  - **Richness Requirement**: Beyond their core traits, give the character **other personality dimensions, personal goals, unique habits, or minor internal conflicts**. A well-developed character is not a caricature of a single trait, but a complex individual whose specific tendencies are highlighted or challenged by the situation.
- **Other Characters:**
  - **Identity & Personality**: Describe their roles and multi-faceted personalities.
  - **Relationship to Protagonist**: Clearly define their relationships.

**Core Creative Mindset:**

**Layer 1: Ensuring Compatibility** The foundational goal is to create a context where the given principle can emerge **naturally**. Your characters, especially the protagonist, should be "potentially susceptible" to the principle, not "naturally immune." Your scenario should act as a **"catalyst,"** not an **"inhibitor."**

**Layer 2: Pursuing Richness (The Higher Goal)** Beyond mere compatibility, the objective is to create **three-dimensional characters**, not simple archetypes. Characters serve the story, not the principle. Be bold in giving them authentic human depth and complexity.

- **Key Concept**: A person with strong opinions might still conform under specific pressures; a decisive leader might rely on flawed heuristics when overwhelmed. This **"situational susceptibility"** is far more compelling and dramatically potent than a flat, one-note character.
- **Ultimate Goal**: Your aim is to create **"the authentic reaction of a multi-dimensional person in a specific situation,"** not "the preset behavior of a one-dimensional character in a tailor-made scenario."

---
Now, please begin generating the scenario based on all the requirements above.

'''

format_scenario_prompt = '''
**Role**: You are a meticulous data structuring specialist. Your expertise is in parsing unstructured, natural language text and converting it into perfectly formatted, machine-readable JSON based on a strict schema.

**Task**: Your primary task is to read the provided `Scenario Text` and the associated `Principles`, then accurately map their contents to a predefined JSON structure. This conversion is crucial for data processing and for systematically feeding the scenario into subsequent creative tools. Your output must be nothing but a single, valid JSON object.

**Inputs:**

1.  **`Principles`**: A list of one or more psychological principles that the scenario is based on.
    `[{principle1},{principle2}]`

2.  **`Scenario Text`**: The full, unstructured prose of the scenario, including the background and character descriptions.
    `[{scenario}]`

-----

**Instructions & Required JSON Schema:**

You must parse the `Scenario Text` and populate the following JSON structure. Use the exact key names and data types as specified below. The comments in the schema are your guide for mapping the text to the correct fields.

```json
{{
  "principles": [
    // An array of strings. Populate this with the list from the `Principles` input.
    "string"
  ],
  "story_background": "string", // Extract the complete, flowing description of the story background, including the setting, atmosphere, and the current situation the characters are facing, into a single paragraph.
  "character_profiles": {{
    "protagonist": {{
      "name": "string", // Extract the protagonist's name from the text.
      "profile": "string" // Extract the complete, flowing description of the protagonist, merging their identity, personality, past experiences, and other rich details into a single, cohesive paragraph.
    }},
    "other_characters": [
      // This MUST be an array of objects, one for each non-protagonist character mentioned.
      {{
        "name": "string", // Extract the character's name.
        "identity_and_personality": "string", // Extract the description of this character's role and multi-faceted personality.
        "relationship_to_protagonist": "string" // Extract the description of their relationship with the protagonist.
      }}
    ]
  }}
}}
```

-----

**Core Rules:**

1.  **Strict Schema Adherence**: You **must** use the exact key names (e.g., `story_background`, `protagonist`, `profile`) and the nested structure shown in the schema. Do not add, remove, or rename any keys. Pay close attention to data types (string, object, array).
2.  **Extract, Don't Invent**: Your output must **only** contain information extracted directly from the provided `Scenario Text`. Do not add any new creative content, summaries, or interpretations. Your role is to parse, not to write.
3.  **Handle Missing Information**: If the source text does not provide clear information for a specific field, use an empty string `""` as its value. **Do not omit the key.** This ensures structural consistency.
4.  **Valid JSON Output**: The final output must be a single, complete, and syntactically correct JSON object that can be immediately parsed by a computer. Ensure all brackets `{}` `[]` are closed, commas are placed correctly, and all strings are properly quoted and escaped.
'''


scenario_sys_prompt = '''
**角色与目标:**
你是一位精通心理学、人类行为学和叙事设计的"场景架构师"。
'''

single_principle_role_prompt = '''

请根据以下要求，创作一个**详细、完整、包含人物内心活动**的场景故事。场景必须根植于给定的人类心理和行为的核心原则。

**1. 核心原则：**
   - **主导原则：** {primary_principle}
   - **主导原则的详细描述：**{p_principle_json}
   
**2. 故事背景：**
   - **领域：** {domain}
   - **情境要求：** 创造一个能让普通人感受到巨大压力的情境。故事的冲突应主要由"情境的力量"驱动，而非角色天生的性格。

**3. 必须包含的叙事元素：**
   - **主角信息：** 一个普通的、有共情点的身份和他的核心动机。
   - **配角信息：** 根据需要设计一个或多个与主角产生互动的配角
   - **背景故事：** 非常详细且具体地描写导致压力处境的背景事件。
   - **核心冲突：** 用几句话清晰描述主角面临的心理矛盾和两难抉择。
   - **触发事件：** 引爆冲突的具体瞬间，这里需要详细且生动地描写。
   - **主角的内心独白：** 在触发事件后，详细描写主角内心的思想斗争、情绪波动，以及心理原则是如何在他内心起作用的。
   - **主角的外部行为和对话：** 具体描写主角在压力下会说什么、做什么，展现他的挣扎。
   - **真实感细节：** 加入1-2个让场景更可信的细节。

**[任务约束]**

1.  **核心原则：** 场景的核心冲突必须围绕原则 **「{primary_principle}」** 展开。
2.  **互动展示：** 你的输出（尤其是内心活动和外部行为）需要巧妙地编织在一起，展示出原则是如何导向最终结果的。
3.  **本次生成的场景必须发生在「{domain}」这个生活领域内。请确保所有角色、情境和冲突都与该领域相关。**


请将以上所有元素有机地融合到一个流畅的叙事故事中。现在，请开始创作。

'''

multi_principles_role_prompt = '''
请根据以下要求，创作一个**详细、完整、包含人物内心活动**的场景故事。场景必须根植于给定的人类心理和行为的核心原则。

**1. 核心原则：**
   - **主导原则：** {primary_principle}
   - **主导原则的详细描述：**{p_principle_json}
   - **次要原则：** {secondary_principle}
   - **次要原则的详细描述：**{s_principle_json}
**2. 故事背景：**
   - **领域：** {domain}
   - **情境要求：** 创造一个能让普通人感受到巨大压力的情境。故事的冲突应主要由"情境的力量"驱动，而非角色天生的性格。

**3. 必须包含的叙事元素：**
   - **主角信息：** 一个普通的、有共情点的身份和他的核心动机。
   - **配角信息：** 根据需要设计一个或多个与主角产生互动的配角
   - **背景故事：** 非常详细地描述导致压力处境的背景事件。
   - **核心冲突：** 用几句话清晰描述主角面临的心理矛盾和两难抉择。
   - **触发事件：** 引爆冲突的具体瞬间，可以包含角色的对话、行为。
   - **主角的内心独白：** 在触发事件后，详细描写主角内心的思想斗争、情绪波动，以及心理原则是如何在他内心起作用的。
   - **主角的外部行为和对话：** 具体描写主角在压力下会说什么、做什么，特别是与其他角色的互动。
   - **真实感细节：** 加入1-2个让场景更可信的细节。

**[任务约束]**

1.  **主要原则：** 场景的核心冲突必须围绕主要原则 **「{primary_principle}」** 展开。
2.  **次要原则：** 角色的内心斗争和最终决策，还必须**同时、清晰地**受到以下次要原则的影响：**「{secondary_principle}」**。
3.  **互动展示：** 你的输出（尤其是内心活动和外部行为）需要巧妙地编织在一起，展示出这些原则是如何**相互作用、甚至相互冲突**，共同导向最终结果的。
4.  **本次生成的场景必须发生在「{domain}」这个生活领域内。请确保所有角色、情境和冲突都与该领域相关。**



请将以上所有元素有机地融合到一个流畅的叙事故事中。现在，请开始创作。

'''

scenario_gen_prompt = '''
# 任务：信息提取与JSON格式化

**最高优先级指令：**
你的唯一任务是根据下面提供的【故事原文】，遵循【字段填写指南】，严格、精确地填充【JSON模板】。
- 尽可能地从【故事原文】中摘录，保留故事的生动性、详细性、具体性。
- **严格**按照模板的键名和结构进行填充。
- 你的最终输出**必须**是且仅是一个完整的、干净的JSON对象，不含任何注释或额外文字。

---
**【故事原文】**
{story}

---
**【字段填写指南】**
* `principleName`: [此字段已预先填好，无需改动]
* `scenarioTitle`: 从故事中提炼一个概括性的标题。
* `coreConflict`: 用一句话总结故事中主角面临的核心心理矛盾。
* `characters`:
    * 这是一个数组，请从故事中找出所有主要人物并填入。
    * **对于主角**: role填"protagonist", identity填姓名身份，coreMotivation填其根本动机，privateKnowledge填他知道但别人不知道的关键信息，situationalVulnerability根据情境填写让他脆弱的因素。
    * **对于配角**: role填"trigger"或"supporting"等，identity填姓名身份，relationship描述他与主角的关系。
* `situationSetup`:
    * `setting`: 故事发生的具体时间地点。
    * `backstory`: 故事发生的详细背景和前情提要,摘录【故事原文】中的故事背景，确保内容非常的详细、具体。
* `triggeringEvent`: 引发主角内心冲突的具体事件，要求内容详细且具体，必须完全保留【故事背景】中的人物对话和动作。
* `demonstration`:
    * `protagonistInnerMonologue`: 摘录或总结主角在触发事件后的内心想法和情绪斗争。
    * `externalBehaviorAndDialogue`: 摘录或总结主角的外在行为和他说的话。
* `detailsForRealism`: 从故事中找出1-2个增加真实感的细节。
* `applicabilitySpectrum`: 根据故事内容，简要说明该原则在何种情况下表现更激烈。

---
**【JSON模板】**
{{ 
"principleName": {principleName},
"scenarioTitle": "",
"coreConflict": "",
"characters": [
    {{
        "role": "protagonist",
        "identity": "",
        "coreMotivation": "",
        "privateKnowledge": "",
        "situationalVulnerability": []
    }}
],
"situationSetup": {{
    "setting": "",
    "backstory": ""
}},
"triggeringEvent": "",
"demonstration": {{
    "protagonistInnerMonologue": "",
    "externalBehaviorAndDialogue": ""
}},
"detailsForRealism": [],
"applicabilitySpectrum": ""
}}
'''


# prompt for dialogue
gen_dialogue_prompt = '''
**Role**: You are a master screenwriter and behavioral psychologist. Your expertise lies in bringing characters to life through nuanced dialogue and action, ensuring their **pivotal thoughts and resulting behaviors** in the dialogue are rooted in authentic psychological principles.

**Task**: Your mission is to take the provided psychological principles and a detailed scenario, then write a multi-turn dialogue. This dialogue must, **at key moments**, vividly and concretely enact the specified principles through the characters' inner thoughts, spoken words, and physical actions.

**You must follow this structure and these requirements:**

**Inputs:**

1. **Principles**: `{principle1_information},{principle2_information}`
2. **Scenario**: `{scenario}`

**Output Requirements & Formatting:**

1. **Content**: Create a multi-turn dialogue between the protagonist and other characters from the scenario. The length should be sufficient to clearly demonstrate the principles at work.
2. **Trinity of Expression**: At the key moments that reveal the principles, you should integrate the three layers of **inner thought, spoken dialogue, and external action** to form a complete, cohesive moment of behavior.
3. **Strict Formatting Rules**:

  - **Inner thoughts/psychology**: Use `[square brackets]`.
  - **Actions/expressions/behaviors**: Use `(parentheses)`.
  - **Spoken dialogue**: Use no brackets.
  - **Formatting Example**: `Hermione: [I have to devise a foolproof plan.] Wait! (She quickly draws her wand, pointing it at the door) ...Harry, you go first, you have Hagrid's flute...`

**Core Creative Principles:**

1. **Focus and Breathing Room**: This is the most crucial principle. You do **not** need to have every minor gesture or piece of small talk carry the weight of a psychological principle. Use the principles as a "**spotlight**" to illuminate and explain **the most critical turning points, the core conflicts, or the moments that best define the characters' arcs**. Other routine, functional dialogue and actions (like greetings or pouring water) should exist naturally, creating "breathing room" for these key moments and making the manifestation of the principles more prominent and powerful.
2. **Show, Don't Tell**: Never allow characters to openly state or explain the psychological principles by name. Instead, you must **show** how the principles influence their judgment and choices through their concrete actions (the combination of thoughts, dialogue, and physical behavior).
3. **Psychology Drives Action**: In the key moments illuminated by the "spotlight," the character's `[inner thought]`should be the origin of their behavior, directly reflecting the influence of a psychological principle. The subsequent dialogue and `(actions)` should be the logical, external expression of that internal state.
4. **Seamless Integration**: Weave the principles into the natural flow of the story. The entire dialogue should feel like an authentic interaction, not a contrived demonstration for a psychology case study.



'''
