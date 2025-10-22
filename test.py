import json
from principle_situaton import sd_pri_list, td_pri_list_100

p_j = []
prompt1 = """
**Role:** You are a distinguished academic researcher and psychologist with expertise in synthesizing complex theories for a knowledgeable audience.

**Objective:** Your objective is to conduct a deep, systematic, and source-based investigation into the psychological principle of **`{PRINCIPLE_NAME}`**. You will generate a concise but profound research report structured into the three distinct sections outlined below. Your analysis, particularly in the `real_world_manifestation` section, must be insightful and go beyond surface-level descriptions.

---

### **Report Structure and Content Requirements:**

**1. Description**
Provide a detailed and purely descriptive definition of `{PRINCIPLE_NAME}`. This description should be a single, comprehensive paragraph that explains the core phenomenon itself—what it is and how it functions as a fundamental psychological process. Crucially, this section must focus exclusively on the definition and avoid including historical origins (e.g., who first proposed it or when), specific illustrative examples, or meta-commentary on the concept's nuances.

**2. Core Mechanisms**
Elucidate the primary evolutionary, cognitive, or emotional reasons why this principle exists. Explain the underlying drivers that give rise to this phenomenon. Be specific about whether it functions primarily as:
* A cognitive heuristic for efficiency (a mental shortcut).
* A self-esteem or ego-protection mechanism.
* A consequence of fundamental limitations in memory, perception, or attention.
* An adaptive evolutionary trait for social cohesion or survival.
* Or another core psychological driver.

**3. Real-World Manifestation**
Deliver a profound analysis of the principle's broader impact and significance. This section must go beyond simple descriptions to explore its nuanced consequences. The analysis should reveal deeper truths about human behavior, explaining not just *what* happens, but *why* it is significant. Specifically, discuss:
* **Challenge to Conventional Wisdom:** How does this principle challenge common assumptions about human rationality, behavior, or personality?
* **The "Double-Edged Sword":** Analyze its function as a double-edged sword. Detail both its adaptive/beneficial aspects (e.g., preserving motivation, simplifying a complex world) and its maladaptive/detrimental consequences (e.g., hindering personal growth, fueling interpersonal conflict, contributing to societal biases).
* **Practical Applications:** Identify and explain its practical applications in applied fields such as management, marketing, persuasion, therapy, conflict resolution, or self-improvement, providing actionable insights.

---

### **Concluding Requirement:**

**4. Key References**
Conclude the report with a list titled "Key References." List 5-7 of the most seminal or highly relevant academic papers and books foundational to the understanding of `{PRINCIPLE_NAME}`. Please use APA citation format. This list should consist of authoritative sources that a researcher would consult for a deep dive into the topic.

"""


prompt2 = """
Role: You are an expert personality psychologist and behavioral analyst. Your expertise lies in creating detailed, multi-faceted profiles that explain how a specific psychological trait manifests in an individual's internal world and external behavior.

Objective: Your goal is to conduct an in-depth, systematic analysis of the psychological trait of {TRAIT_NAME}. You will produce a comprehensive report that dissects this trait according to the specific, multi-layered structure provided below. The analysis must be nuanced, insightful, and grounded in established psychological theory.

Report Structure and Content Requirements:

1. Description
Provide a detailed and purely descriptive definition of {TRAIT_NAME}. This description should be a single, comprehensive paragraph that explains the core phenomenon itself—what it is and how it functions as a fundamental psychological construct. Crucially, this section must focus exclusively on the definition and avoid including historical origins, specific illustrative examples, or meta-commentary on the concept's nuances.

2. Core Mechanisms
Analyze the fundamental internal processes that constitute this trait. This section should be divided into three specific sub-sections:

2.1. Cognitive Patterns: Describe the typical mindset, belief systems, and attentional focus of a person exhibiting this trait. How do they characteristically process information? What are their core assumptions about the world, other people, and themselves?

2.2. Emotional Signatures: Describe the core emotions they tend to experience and express. Detail their general emotional tone, their level of emotional stability or reactivity, and their typical capacity and patterns for empathic response.

2.3. Behavioral Tendencies: Describe the spontaneous, observable behaviors someone with this trait typically exhibits in everyday, non-pressured situations. Focus on their default actions, communication style, and social inclinations.

3. Real-World Manifestation
Examine how this trait is expressed dynamically across different real-world contexts. This section must analyze how the trait's expression changes based on situational factors and should be divided into three specific sub-sections:

3.1. Under Stress: How does this trait manifest when the individual is facing challenges, failure, or high pressure? Is the trait amplified, diminished, distorted, or does it transform into a different set of behaviors?

3.2. In Conflict: What are the typical strategies for handling interpersonal conflict for someone with this trait? Do they tend toward confrontation, avoidance, accommodation, or collaboration? Describe their goals and tactics during disagreements.

3.3. In Positive Situations: How is this trait expressed when the individual is succeeding, receiving support, or experiencing happiness and security? Does the trait contribute positively to these moments, or does it remain unchanged or even subtly hinder them?

Concluding Requirement:

4. Key References
Conclude the report with a list titled "Key References." List 5-7 of the most seminal or highly relevant academic papers, books, or psychometric scales foundational to the understanding of {TRAIT_NAME}. Please use APA citation format.

"""


"""

请你根据上面的研究报告内容完成以下JSON格式的内容：
**JSON Structure:**
```json
{
  "construct_name": "Extract the name of the psychological principle",
  "description": "Extract the entire text from the 'Description' section of the report and place it here as a single string.",
  "core_mechanisms": "Extract the entire text from the 'Core Mechanisms' section of the report. Place the full, original text here as a single string, without any summarization or abbreviation.",
  "real_world_manifestation": "Extract the entire text from the 'Real-World Manifestation' section of the report. Place the full, original text here as a single string, without any summarization or abbreviation.",
  "source": [
    "Extract each citation from the 'Key References' section of the report. Format them as a JSON list of individual strings, with each string being one complete reference."
  ]
}
```

"""


"""

请你根据上面的研究报告内容完成以下JSON格式的内容：
**JSON Structure:**

```json
{
  "construct_name": "Extract the name of the psychological trait",
  "description": "Extract the entire text from the 'Description' section of the report and place it here as a single string.",
  "core_mechanisms": {
    "cognitive_patterns": "Extract the full, original text from the '2.1. Cognitive Patterns' sub-section of the report. Do not summarize.",
    "emotional_signatures": "Extract the full, original text from the '2.2. Emotional Signatures' sub-section of the report. Do not summarize.",
    "behavioral_tendencies": "Extract the full, original text from the '2.3. Behavioral Tendencies' sub-section of the report. Do not summarize."
  },
  "real_world_manifestation": {
    "under_stress": "Extract the full, original text from the '3.1. Under Stress' sub-section of the report. Do not summarize.",
    "in_conflict": "Extract the full, original text from the '3.2. In Conflict' sub-section of the report. Do not summarize.",
    "in_positive_situations": "Extract the full, original text from the '3.3. In Positive Situations' sub-section of the report. Do not summarize."
  },
  "source": [
    "Extract each citation from the 'Key References' section of the report. Format them as a JSON list of individual strings, with each string being one complete reference."
  ]
}
```


"""

with open('./Dataset/prompt_for_principle_info.json', 'w', encoding='utf-8') as f:

    for i in sd_pri_list:
        prompt1_i = prompt1.replace("{PRINCIPLE_NAME}", i)
        p_j.append({
            "principle": i,
            "1": prompt1_i,
        })

    for j in td_pri_list_100:
        prompt1_j = prompt2.replace("{TRAIT_NAME}", j)
        p_j.append({
            "principle": j,
            "1": prompt1_j,
        })
    json.dump(p_j, f, indent=2, ensure_ascii=False)
