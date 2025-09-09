import pandas as pd
pri_list = ['生存指令', '厌恶反应', '当下偏好', '认知失调与自我辩护', '后视偏见', '乐观/悲观偏见', '决策疲劳', '峰终定律', '自利偏见', '叙事自我', '享乐适应',
            '自我决定理论', '快乐原则与现实原则', '意义寻求', '影响偏见', '共情鸿沟', '聚光灯效应', '控制错觉', '克制偏见', '道德许可效应', '亲缘选择与广义适合度',
            '亲代投资不对称', '互惠式利他主义', '等级序列的形成', '领地意识', '择偶策略', '嫉妒', '父权不确定性', '基本归因错误', '内/外群体偏见', '社会从众', '服从权威',
            '社会比较与地位焦虑', '互惠规范', '社会认同', '群体思维', '旁观者效应', '光环效应', '公正世界假设', '社会助长', '责任分散', '负面偏好', '锚定与调整',
            '确认偏误', '损失厌恶', '可得性启发法', '代表性启发法', '沉没成本谬误', '框架效应', '现状偏见', '禀赋效应', '心理账户', '选择过载', '敬畏感',
            '凡人终死效应与遗产驱动', '心流原则', '感恩机制', '创伤后成长', '皮肤饥渴与触觉律令', '自我妨碍悖论', '禁忌之魅惑', '神话叙事律令', '虐待性快感',
            '自我欺骗的效用原则', '逃避自由原则', '狄俄尼索斯冲动', '灾难性风险偏好原则', '有意选择平凡原则', '非人化开关原则', '涌现式愚蠢原则', '游戏冲动原则',
            '心理状态切换原则', '叙事解脱原则']


pri_list_en = ['Actor observer asymmetry', 'Defensive attribution hypothesis', 'Effort Justification', 'Egocentric bias', 'False consensus effect', 'Forer effect', 'Fundamental attribution error', 'Hard-easy effect', 'Illusion of control', 'Illusory superiority', 'Optimism bias', 'Overconfidence effect', 'Risk compensation', 'Self-serving bias', 'Social desirability bias', 'Third-person effect', 'Decoy effect', 'Reactance', 'Social comparison bias', 'Status quo bias', 'Backfire effect', 'Endowment effect', 'Loss aversion', 'Pseudocertainty effect', 'Sunk cost fallacy', 'Zero-risk bias', 'Hyperbolic discounting', 'Identifiable victim effect', 'Ambiguity bias', 'Belief bias', 'Information bias', 'Less-is-better effect', 'Authority bias', 'Automation bias', 'Bandwagon effect', 'Group attribution error', 'Just-world hypothesis', 'Stereotyping', 'Ultimate attribution error', 'Halo effect', 'In-group bias', 'Out-group homogeneity bias', 'Positivity effect', 'Reactive devaluation', 'Hindsight bias', 'Impact bias', 'Outcome bias', 'Pessimism bias', 'Planning fallacy', 'Projection bias', 'Restraint bias', 'Self-consistency bias', 'Denomination effect', 'Mental accounting', 'Normalcy bias', 'Subadditivity effect', 'Survivorship bias', 'Zero-sum bias', 'Anthropomorphism', 'Illusion of validity', 'Illusory correlation', 'Curse of knowledge', 'Illusion of asymmetric insight', 'Illusion of transparency', 'Spotlight effect', 'Negativity bias', 'Choice-supportive bias', 'Confirmation bias', 'Continued influence effect', 'Expectation bias', 'Observer effect', 'Observer-expectancy effect', 'Ostrich effect', 'Bias blind spot', 'Naive cynicism', 'Naive realism',
               'Attentional bias', 'Availability heuristic', 'Base rate fallacy', 'Context effect', 'Empathy gap', 'Illusory truth effect', 'Mere exposure effect', 'Mood-congruent memory bias', 'Omission bias', 'Anchoring', 'Conservatism', 'Contrast effect', 'Distinction Bias', 'Focusing effect', 'Framing effect', 'Fading affect bias', 'Implicit association', 'Implicit stereotypes', 'False memory', 'Misattribution of memory', 'Source confusion', 'Misinformation effect', 'Peak-end rule', 'Delayed Reciprocity', 'Asymmetrical Investment', 'Survival Imperative', 'Aversion Response', 'Narrative Self', 'Hedonic Adaptation', 'Self-Determination Theory', 'Pleasure Principle & Reality Principle', 'Search for Meaning', 'Moral Licensing Effect', 'Choice Overload', 'Kin Selection & Inclusive Fitness', 'Asymmetrical Parental Investment', 'Formation of Dominance Hierarchies', 'Territoriality', 'Mating Strategies', 'Jealousy', 'Paternity Uncertainty', 'Groupthink', 'Bystander Effect', 'Social Facilitation', 'Diffusion of Responsibility', 'Dehumanization Switch Principle', 'Decision Fatigue', 'Awe', 'Mortality Salience & Legacy Drive', 'Flow Principle', 'Gratitude Mechanism', 'Post-Traumatic Growth', 'Skin Hunger & the Law of Touch', 'Self-Handicapping Paradox', 'The Allure of the Forbidden', 'Mythical Narrative Imperative', 'Sadistic Pleasure', 'The Utility Principle of Self-Deception', 'The Principle of Escaping from Freedom', 'Dionysian Impulse', 'Catastrophic Risk Preference Principle', 'The Principle of Intentionally Choosing Mediocrity', 'Emergent Stupidity Principle', 'Play Impulse Principle', 'Mental State Switching Principle', 'Narrative Relief Principle']

pri_rel = {
    "生存指令": [
        "厌恶反应",
        "共情鸿沟",
        "社会助长",
        "负面偏好",
        "损失厌恶"
    ],
    "厌恶反应": [
        "生存指令",
        "内/外群体偏见"
    ],
    "当下偏好": [
        "快乐原则与现实原则",
        "克制偏见",
        "决策疲劳"
    ],
    "认知失调与自我辩护": [
        "沉没成本谬误",
        "自利偏见",
        "叙事自我",
        "互惠规范"
    ],
    "后视偏见": [
        "控制错觉",
        "叙事自我"
    ],
    "乐观/悲观偏见": [
        "控制错觉",
        "自利偏见",
        "克制偏见"
    ],
    "决策疲劳": [
        "当下偏好",
        "道德许可效应",
        "现状偏见",
        "锚定与调整"
    ],
    "峰终定律": [
        "叙事自我",
        "可得性启发法"
    ],
    "自利偏见": [
        "控制错觉",
        "基本归因错误",
        "乐观/悲观偏见"
    ],
    "叙事自我": [
        "认知失调与自我辩护",
        "峰终定律",
        "后视偏见",
        "意义寻求",
        "社会比较与地位焦虑"
    ],
    "享乐适应": [
        "社会比较与地位焦虑",
        "影响偏见"
    ],
    "自我决定理论": [
        "快乐原则与现实原则",
        "意义寻求",
        "叙事自我"
    ],
    "快乐原则与现实原则": [
        "当下偏好",
        "克制偏见"
    ],
    "意义寻求": [
        "叙事自我",
        "自我决定理论"
    ],
    "影响偏见": [
        "享乐适应",
        "认知失调与自我辩护"
    ],
    "共情鸿沟": [
        "当下偏好",
        "克制偏见"
    ],
    "聚光灯效应": [
        "叙事自我",
        "社会比较与地位焦虑"
    ],
    "控制错觉": [
        "乐观/悲观偏见",
        "自利偏见"
    ],
    "克制偏见": [
        "乐观/悲观偏见",
        "快乐原则与现实原则",
        "共情鸿沟",
        "快乐原则与现实原则"
    ],
    "道德许可效应": [
        "心理账户",
        "自利偏见",
        "决策疲劳"
    ],
    "亲缘选择与广义适合度": [
        "互惠式利他主义",
        "亲代投资不对称",
        "内/外群体偏见"
    ],
    "亲代投资不对称": [
        "等级序列的形成",
        "择偶策略",
        "嫉妒",
        "父权不确定性",
        "社会比较与地位焦虑"
    ],
    "互惠式利他主义": [
        "亲缘选择与广义适合度",
        "公正世界假设",
        "互惠规范"
    ],
    "等级序列的形成": [
        "择偶策略",
        "社会比较与地位焦虑",
        "服从权威"
    ],
    "领地意识": [
        "生存指令",
        "等级序列的形成"
    ],
    "择偶策略": [
        "亲代投资不对称",
        "嫉妒",
        "等级序列的形成"
    ],
    "嫉妒": [
        "择偶策略",
        "亲代投资不对称",
        "父权不确定性"
    ],
    "父权不确定性": [
        "择偶策略",
        "嫉妒",
        "亲代投资不对称"
    ],
    "基本归因错误": [
        "自利偏见",
        "内/外群体偏见",
        "公正世界假设"
    ],
    "内/外群体偏见": [
        "亲缘选择与广义适合度",
        "光环效应",
        "社会从众",
        "基本归因错误",
        "群体思维"
    ],
    "社会从众": [
        "社会认同",
        "群体思维",
        "内/外群体偏见",
        "服从权威"
    ],
    "服从权威": [
        "社会从众",
        "群体思维",
        "等级序列的形成"
    ],
    "社会比较与地位焦虑": [
        "损失厌恶",
        "内/外群体偏见",
        "叙事自我"
    ],
    "互惠规范": [
        "互惠式利他主义",
        "社会从众",
        "认知失调与自我辩护"
    ],
    "社会认同": [
        "社会从众",
        "旁观者效应",
        "责任分散"
    ],
    "群体思维": [
        "社会从众",
        "内/外群体偏见",
        "服从权威"
    ],
    "旁观者效应": [
        "社会认同",
        "责任分散"
    ],
    "光环效应": [
        "确认偏误",
        "代表性启发法"
    ],
    "公正世界假设": [
        "基本归因错误",
        "自利偏见"
    ],
    "社会助长": [
        "聚光灯效应",
        "社会比较与地位焦虑"
    ],
    "责任分散": [
        "内/外群体偏见",
        "旁观者效应"
    ],
    "负面偏好": [
        "生存指令",
        "损失厌恶",
        "可得性启发法"
    ],
    "锚定与调整": [
        "框架效应",
        "确认偏误"
    ],
    "确认偏误": [
        "认知失调与自我辩护"
    ],
    "损失厌恶": [
        "沉没成本谬误",
        "现状偏见"
    ],
    "可得性启发法": [
        "框架效应",
        "负面偏好"
    ],
    "代表性启发法": [
        "基本归因错误"
    ],
    "沉没成本谬误": [
        "认知失调与自我辩护",
        "损失厌恶"
    ],
    "框架效应": [
        "损失厌恶",
        "锚定与调整"
    ],
    "现状偏见": [
        "损失厌恶",
        "决策疲劳",
        "禀赋效应"
    ],
    "禀赋效应": [
        "损失厌恶",
        "现状偏见"
    ],
    "心理账户": [
        "损失厌恶",
        "框架效应"
    ],
    "选择过载": [
        "锚定与调整",
        "决策疲劳",
        "框架效应"
    ],
    "敬畏感": [
        "生存指令",
        "聚光灯效应",
        "自利偏见",
        "叙事自我",
        "意义寻求"
    ],
    "凡人终死效应与遗产驱动": [
        "亲缘选择与广义适合度",
        "内/外群体偏见",
        "意义寻求",
        "叙事自我"
    ],
    "心流原则": [
        "享乐适应",
        "自我决定理论",
        "聚光灯效应",
        "决策疲劳",
        "选择过载"
    ],
    "感恩机制": [
        "负面偏好",
        "自我决定理论",
        "互惠规范",
        "社会比较与地位焦虑",
        "互惠式利他主义",
        "损失厌恶"
    ],
    "创伤后成长": [
        "认知失调与自我辩护",
        "享乐适应",
        "负面偏好",
        "叙事自我"
    ],
    "皮肤饥渴与触觉律令": [
        "生存指令",
        "择偶策略",
        "自我决定理论",
        "亲缘选择与广义适合度",
        "厌恶反应"
    ],
    "自我妨碍悖论": [
        "乐观/悲观偏见",
        "沉没成本谬误",
        "自利偏见",
        "认知失调与自我辩护"
    ],
    "禁忌之魅惑": [
        "可得性启发法",
        "服从权威",
        "快乐原则与现实原则",
        "自我决定理论"
    ],
    "神话叙事律令": [
        "公正世界假设",
        "内/外群体偏见",
        "凡人终死效应与遗产驱动",
        "叙事自我"
    ],
    "虐待性快感": [
        "共情鸿沟",
        "服从权威",
        "内/外群体偏见",
        "等级序列的形成"
    ],
    "自我欺骗的效用原则": [
        "乐观/悲观偏见",
        "自利偏见",
        "控制错觉",
        "锚定与调整",
        "可得性启发法"
    ],
    "逃避自由原则": [
        "社会从众",
        "自我决定理论",
        "服从权威"
    ],
    "狄俄尼索斯冲动": [
        "生存指令",
        "意义寻求",
        "叙事自我"
    ],
    "灾难性风险偏好原则": [
        "狄俄尼索斯冲动",
        "损失厌恶",
        "现状偏见"
    ],
    "有意选择平凡原则": [
        "意义寻求",
        "自我决定理论",
        "社会比较与地位焦虑",
        "等级序列的形成"
    ],
    "非人化开关原则": [
        "虐待性快感",
        "内/外群体偏见",
        "共情鸿沟",
        "服从权威"
    ],
    "涌现式愚蠢原则": [
        "群体思维"
    ],
    "游戏冲动原则": [
        "亲缘选择与广义适合度",
        "生存指令",
        "心流原则",
        "快乐原则与现实原则"
    ],
    "心理状态切换原则": [
        "损失厌恶",
        "内/外群体偏见",
        "感恩机制",
        "灾难性风险偏好原则"
    ],
    "叙事解脱原则": [
        "心流原则",
        "叙事自我",
        "逃避自由原则"
    ]
}


print(len(pri_list_en))


