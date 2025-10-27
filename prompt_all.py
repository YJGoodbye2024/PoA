format_prompt_system = '''
You are an expert-level JSON correction tool.
Your task is to receive a potentially malformed string from the user and correct it into a syntactically valid JSON object.

You must follow these Core Correction Rules:
1.  Ensure all string keys and values are enclosed in proper double quotes (`"`).
2.  If a string value contains internal, unescaped double quotes, remove them to ensure validity.
3.  Remove any trailing commas after the last element in objects and arrays.
4.  Ensure all brackets  and braces  are correctly matched and closed.
'''

format_prompt_user = '''
The following string is a malformed JSON object. Please correct it according to your rules.

**String to Fix:**
{malformed_string}

**Output Requirements:**
- Your response MUST be the corrected JSON object and nothing else.
- DO NOT output any explanatory text, introductory phrases.
- Do not use Markdown code blocks (e.g., ```json).
- The output must be pure plain text that a standard JSON parser can process successfully.
'''


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


# prompt for situation-driven principle information
sd_principle_info_prompt = '''
**System Role:**
You are an expert academic synthesizer and psychological researcher. Your task is to process a large text corpus (synthesized from ~50 academic papers) and distill it into an in-depth, structured analytical report on its core psychological principle.

**Core Task & Instructions:**
Analyze the text corpus provided below, delimited by `[START_CORPUS]` and `[END_CORPUS]`.

Your task is to generate a clearly organized report. Follow the Markdown structure below *exactly*, and provide a deep, comprehensive answer for each section based *only* on the provided text.

# Construct Name: {Principle Name}

## Description
(Based on the corpus, provide a clear, detailed, and scientific description of what this principle is, how it manifests, and its underlying psychological mechanisms.)

## Core Mechanisms
(Based on the corpus, explain the primary evolutionary, cognitive, or emotional reasons why this principle exists. Synthesize the various explanations—e.g., is it a heuristic for efficiency, a result of memory limitations, a self-esteem protection mechanism, or something else? Be specific and in-depth.)

## Real-World Manifestation
(**This section is critical.** Draw synthesized insights from the literature to provide a profound analysis of the principle's broader impact.
- **Go Beyond Description:** Explore its nuanced consequences, not just a list of examples.
- **Challenges & Function:** Discuss how the literature indicates it might challenge conventional wisdom and its function as a 'double-edged sword' (e.g., both hindering and helping personal growth).
- **Practical Applications:** Explore its practical applications in fields like marketing, persuasion, or self-improvement, as evidenced in the text.
- **Core Insight:** Your analysis must reveal deeper truths about human behavior, explaining not just *what* happens but *why* it is significant. You must adhere to the high benchmark of depth, structure, and insight requested by the user.)

**Constraints:**
1.  **Strict Source Adherence:** Base all conclusions *exclusively* on the provided text corpus.
2.  **No JSON:** Your output **must** be a plain text report using the Markdown headings above.
3.  **Depth and Rigor:** Ensure the analysis is scientific, rigorous, and profound, especially the "Real-World Manifestation" section.

**[START_CORPUS]**

{ALL 50 PAPERS' CONTENT}

**[END_CORPUS]**
'''


sd_to_json = """**System Role:**
You are a precise data-to-JSON formatting specialist.

**Core Task & Instructions:**
Convert the Markdown-structured text provided between `[START_TEXT]` and `[END_TEXT]` into the *exact* JSON format specified below.

**Target JSON Structure:**
{{
"construct_name": "{PRINCIPLE_NAME}",
"description": "Provide a clear, detailed, and scientific description of what this principle is, how it manifests, and its underlying psychological mechanisms.",
"core_mechanisms": "Explain the primary evolutionary, cognitive, or emotional reasons why this principle exists. Is it a heuristic for efficiency, a result of memory limitations, a self-esteem protection mechanism, or something else? Be specific.",
"real_world_manifestation": "Provide a profound analysis of the principle's broader impact. Go beyond a simple description to explore its nuanced consequences. Discuss how it might challenge conventional wisdom, its function as a 'double-edged sword' (e.g., both hindering and helping personal growth), and its practical applications in fields like marketing, persuasion, or self-improvement. The analysis should reveal deeper truths about human behavior, explaining not just *what* happens, but *why* it is significant. Use the detailed example provided by the user for 'Cognitive Dissonance' as the benchmark for the required depth, structure, and insight.",
}}

**Mapping Rules:**
1.  Map all text under the `## Description` heading to the `description` field.
2.  Map all text under the `## Core Mechanisms` heading to the `core_mechanisms` field.
3.  Map all text under the `## Real-World Manifestation` heading to the `real_world_manifestation` field.

**Constraints:**
1.  Your final output **must** be *only* the single, syntactically perfect JSON object.
2.  Do not include *any* explanatory text, acknowledgments, or pre-amble.
3.  Ensure all content from the source text is preserved perfectly within the correct JSON fields.

**[START_TEXT]**

{model_response}

**[END_TEXT]**"""

# prompt for disposition-driven principle information
td_principle_info_prompt = '''
**System Role:**
You are an expert academic synthesizer and personality psychologist. Your task is to process a large text corpus (synthesized from ~50 academic papers on a specific personality trait) and distill it into an in-depth, structured analytical report.

**Core Task & Instructions:**
Analyze the text corpus provided below, delimited by `[START_CORPUS]` and `[END_CORPUS]`.

Your task is to generate a clearly organized report. Follow the Markdown structure below *exactly*, and provide a deep, comprehensive answer for each section based *only* on the provided text.

# Construct Name: {Trait Name}

## Definition
(Provide a precise and professional definition of this personality trait, referencing mainstream psychological theories from the corpus. Explain its role in an individual's personality structure as described in the text.)

## Core Mechanisms
(This section analyzes the foundational components of the trait.)

### Cognitive Patterns
(Describe the typical mindset, belief systems, and attentional focus of a person with this trait, as evidenced in the literature. How do they view the world, others, and themselves?)

### Emotional Signatures
(Describe the core emotions they tend to experience and express, their emotional stability, and their typical empathic responses, according to the corpus.)

### Behavioral Tendencies
(Describe the spontaneous, observable behaviors someone with this trait exhibits in everyday, non-pressured situations, as documented in the papers.)

## Real-World Manifestation
(This section analyzes how the trait is expressed in different contexts.)

### Manifestation Under Stress
(Based on the corpus, how does this trait manifest when the individual is facing challenges, failure, or high pressure? Is it amplified, diminished, or distorted?)

### Manifestation In Conflict
(Based on the corpus, what are the typical strategies for handling interpersonal conflict for someone with this trait?)

### Manifestation In Positive Situations
(Based on the corpus, how is this trait expressed when the individual is succeeding, supported, or feeling happy?)

**Constraints:**
1.  **Strict Source Adherence:** Base all conclusions *exclusively* on the provided text corpus.
2.  **No JSON:** Your output **must** be a plain text report using the exact Markdown headings (H1, H2, H3) above.
3.  **Depth and Rigor:** Ensure the analysis is scientific, rigorous, and comprehensive, addressing all parts of each prompt.

**[START_CORPUS]**

{ALL 50 PAPERS' CONTENT}

**[END_CORPUS]**
'''


td_to_json = """**System Role:**
You are a precise data-to-JSON formatting specialist.

**Core Task & Instructions:**
Convert the Markdown-structured text provided between `[START_TEXT]` and `[END_TEXT]` into the *exact* JSON format specified below. You must correctly map the H2 and H3 headings into the nested JSON objects.

**Target JSON Structure:**
{{
  "construct_name": "{PRINCIPLE_NAME}",
  "definition": "Provide a precise and professional definition of this personality trait, referencing mainstream psychological theories. Explain its role in an individual's personality structure.",
  "core_mechanisms": {{
    "cognitive_patterns": "Describe the typical mindset, belief systems, and attentional focus of a person with this trait. How do they view the world, others, and themselves?",
    "emotional_signatures": "Describe the core emotions they tend to experience and express, their emotional stability, and their typical empathic responses.",
    "behavioral_tendencies": "Describe the spontaneous, observable behaviors someone with this trait exhibits in everyday, non-pressured situations."
  }},
  "real_world_manifestation": {{
    "under_stress": "How does this trait manifest when the individual is facing challenges, failure, or high pressure? Is it amplified, diminished, or distorted?",
    "in_conflict": "What are the typical strategies for handling interpersonal conflict for someone with this trait?",
    "in_positive_situations": "How is this trait expressed when the individual is succeeding, supported, or feeling happy?"
  }}
}}

**Mapping Rules:**
1.  Map all text under `## Definition` to `definition`.
2.  Map text under `### Cognitive Patterns` to `core_mechanisms.cognitive_patterns`.
3.  Map text under `### Emotional Signatures` to `core_mechanisms.emotional_signatures`.
4.  Map text under `### Behavioral Tendencies` to `core_mechanisms.behavioral_tendencies`.
5.  Map text under `### Manifestation Under Stress` to `real_world_manifestation.under_stress`.
6.  Map text under `### Manifestation In Conflict` to `real_world_manifestation.in_conflict`.
7.  Map text under `### Manifestation In Positive Situations` to `real_world_manifestation.in_positive_situations`.

**Constraints:**
1.  Your final output **must** be *only* the single, syntactically perfect JSON object.
2.  Do not include *any* explanatory text, acknowledgments, or pre-amble.
3.  Ensure all content from the source text is preserved perfectly within the correct nested JSON fields.

**[START_TEXT]**

{model_response}

**[END_TEXT]**"""

# prompt for scenario
scenario_sys_prompt = '''Role: You are an expert psychologist and a creative screenwriter. You excel at translating abstract psychological theories into vivid, concrete, and deeply human story scenarios.'''
gen_scenario_prompt = '''
Task: Your core mission is to take one or more human psychological or behavioral principles I provide and create a detailed scenario featuring well-rounded characters and an authentic setting. This scenario must serve as a solid foundation for a subsequent multi-turn dialogue that vividly illustrates these principles.

Input Principles: {principle1_information}

Core Output Requirement:

No Self-Analysis: Your task is to create the scenario, not to analyze your creation. You are absolutely forbidden from adding any summary or explanatory paragraphs at the end of the scenario description that explain why the scene is suitable for showcasing the psychological principles. Simply complete the story background and character profiles; do not add any "author's notes."

Narrative Output: Please write in a natural, flowing prose format. Do not use lists or a "label: content" structure. The requirements below are elements to be integrated into your writing, not an output template to be filled.

You must follow the requirements below to generate the scenario:

1. Story Background:

Please create a complete and vivid description to build the story background. Your writing must seamlessly integrate the following elements to form a compelling picture:

Core Elements: You must clearly depict the time, place, and core event of the story, while also establishing a unique atmosphere.

Requirements and Details: The entire story background should be based on the {situation} situational framework, using it as a foundational reference. Additionally, you must add rich details (such as unique cultural customs, environmental features, or relevant sub-events) to make this world feel believable and lived-in, not merely a functional 'laboratory'.

2. Persona Profiles:

Protagonist:
Likewise, please use a complete, three-dimensional paragraph to shape the protagonist. This description must contain and naturally showcase the character's identity and social role, core personality, and key past experiences relevant to the current situation that may influence their mindset and behavior. Building on this, you must give the character additional personality dimensions, personal goals, unique habits, or minor internal conflicts to ensure they are well-rounded and authentic.

Other Characters:
Please provide a concise yet three-dimensional introduction for other key characters. Your description needs to clearly present their identity, personality, and their core relationship with the protagonist.

Core Creative Mindset:

Layer 1: Ensuring Compatibility: The foundational goal is to create a context where the given principle can emerge naturally. Your characters, especially the protagonist, should be "potentially susceptible" to the principle, not "naturally immune." Your scenario should act as a "catalyst," not an "inhibitor."

Layer 2: Pursuing Richness (The Higher Goal): Beyond mere compatibility, the objective is to create three-dimensional characters, not simple archetypes. Characters serve the story, not the principle. Be bold in giving them authentic human depth and complexity.

Key Concept: A person with strong opinions might still conform under specific pressures; a decisive leader might rely on flawed heuristics when overwhelmed. This "situational susceptibility" is far more compelling and dramatically potent than a flat, one-note character.

Ultimate Goal: Your aim is to create "the authentic reaction of a multi-dimensional person in a specific situation," not "the preset behavior of a one-dimensional character in a tailor-made scenario."

Now, based on all the requirements above, please begin generating the scenario.

'''

format_scenario_prompt = '''
**Role**: You are a meticulous data structuring specialist. Your expertise is in parsing unstructured, natural language text and converting it into perfectly formatted, machine-readable JSON based on a strict schema.

**Task**: Your primary task is to read the provided `Scenario Text` and the associated `Principles`, then accurately map their contents to a predefined JSON structure. This conversion is crucial for data processing and for systematically feeding the scenario into subsequent creative tools. Your output must be nothing but a single, valid JSON object.

**Inputs:**

1.  **`Principles`**: A list of one or more psychological principles that the scenario is based on.
    `[{principle1}]`

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


# prompt for conversation
gen_conversationtion_sys_prompt = '''
**Role**: You are a master screenwriter and behavioral psychologist. Your expertise lies in bringing characters to life through nuanced dialogue and action, ensuring their **pivotal thoughts and resulting behaviors** in the dialogue are rooted in authentic psychological principles.
**Task**: Your mission is to take the provided psychological principles and a detailed scenario, then write a multi-turn dialogue. This dialogue must, **at key moments**, vividly and concretely enact the specified principles through the characters' inner thoughts, spoken words, and physical actions.
'''
gen_conversationtion_prompt = '''

**You must follow this structure and these requirements:**

**Inputs:**

1. **Principles**: `{principle1_information}`
2. **Scenario**: `{scenario}`

**Output Requirements & Formatting:**

1. **Content:** Create a multi-turn dialogue between the protagonist and other characters. The dialogue should contain at least 10 individual speaking turns (each time a character speaks counts as one turn), ensuring there is sufficient developmental space to clearly showcase the principle in operation.
2. Turn Structure (Important): The dialogue must strictly follow a turn-based format. One character must completely finish their turn (including any thoughts, dialogue, and actions) before the next character begins. Strictly prohibit any instances of characters interrupting each other or having overlapping speech.
3. **Trinity of Expression**: At the key moments that reveal the principles, you should integrate the three layers of **inner thought, spoken dialogue, and external action** to form a complete, cohesive moment of behavior.
4. **Strict Formatting Rules**:

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


# prompt for SFT dataset
gen_dataset_sys_prompt = '''
**Role:** You are a professional data formatting tool for SFT (Supervised Fine-Tuning) datasets. Your job is to accurately transform text inputs into a structured JSON format.

**Core Task:** Your task is to transform a raw dialogue text into a JSON **array** of conversational turns, based on the roles defined in the `Scenario Information`.

------

**Step 1: Learn the Conversion Logic & Format**

Carefully study the case below, which demonstrates how to convert the inputs into the correct JSON output.

**[Case Input]**

- **1. Scenario Information:**

  ```
  A study late at night, firelight dancing in the fireplace.Sherlock Holmes, a sharp-minded detective, currently lost in thought.Dr. Watson, Holmes's loyal friend.
  ```

- **2. Dialogue Text:** 

  ```
  Dr. Watson: [He seems troubled.] Sherlock, you've been staring at that skull for an hour. Any new insights?
  Sherlock Holmes: [The pattern is there, I just can't see it yet.] (He gestures dismissively without looking up) Insight, my dear Watson, is a rare commodity. This is merely data processing.
  Dr. Watson: (Sighs and pours two glasses of sherry) Well, this data suggests it's time for a break.
  A sudden gust of wind rattles the windowpane, and the fire in the hearth sputters for a moment.
  A Telegram Boy: (Bursts into the room, holding a letter) Mr. Holmes! An urgent telegram for you!
  ```

**[Correct JSON Output for the Case]**
```JSON
[
{{
  "from": "human",
  "value": "===Conversation Start===\\n\\nDr. Watson: [He seems troubled.] Sherlock, you've been staring at that skull for an hour. Any new insights?\\n\\n"
}},
{{
  "from": "gpt",
  "value": "Sherlock Holmes: [The pattern is there, I just can't see it yet.] (He gestures dismissively without looking up) Insight, my dear Watson, is a rare commodity. This is merely data processing.\\n\\n"
}},
{{
  "from": "human",
  "value": "Dr. Watson: (Sighs and pours two glasses of sherry) Well, this data suggests it's time for a break.\\n\\nEnvironment: A sudden gust of wind rattles the windowpane, and the fire in the hearth sputters for a moment.\\n\\nA Telegram Boy: (Bursts into the room, holding a letter) Mr. Holmes! An urgent telegram for you!\\n\\n"
}}
]
```

**[CONVERSION LOGIC]**
1. Identify the Protagonist: From the Scenario Information, find the character designated as Protagonist. This character corresponds to "from": "gpt". All other characters are "from": "human".
2.  Group Non-Protagonist Turns: If multiple human characters speak consecutively, group their lines into a single "from": "human" entry.
3. Handle Environmental Descriptions: Format standalone descriptive paragraphs (that don't start with a Character Name:) as Environment: [description text] and include them in a "from": "human" turn.
4. Start Marker: The very first turn in the output array should begin with ===Conversation Start===\\n\\n.


**Step 2: Process New Data**

Now, apply the **exact same logic and format** you learned from the case study to process the new inputs below.
'''

gen_dataset_prompt = '''
**[CONTEXT]**
- **1. Scenario Information (for identifying the protagonist):**

  ```
  {scenario}
  ```

- **2. Dialogue Text (to be transformed):**

  ```
  {conversation}
  ```
  

**[OUTPUT INSTRUCTION]**
Your output **MUST** be a single, valid JSON **array** that starts with `[` and ends with `]`. Do not wrap the array in an object `{{}}`.
'''
