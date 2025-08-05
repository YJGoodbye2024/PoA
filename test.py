import json

pri_list=['生存指令', '厌恶反应', '当下偏好', '认知失调与自我辩护', '后视偏见', '乐观/悲观偏见', '决策疲劳', '峰终定律', '自利偏见', '叙事自我', '享乐适应', 
          '自我决定理论', '快乐原则与现实原则', '意义寻求', '影响偏见', '共情鸿沟', '聚光灯效应', '控制错觉', '克制偏见', '道德许可效应', '亲缘选择与广义适合度', 
          '亲代投资不对称', '互惠式利他主义', '等级序列的形成', '领地意识', '择偶策略', '嫉妒', '父权不确定性', '基本归因错误', '内/外群体偏见', '社会从众', '服从权威', 
          '社会比较与地位焦虑', '互惠规范', '社会认同', '群体思维', '旁观者效应', '光环效应', '公正世界假设', '社会助长', '责任分散', '负面偏好', '锚定与调整', 
          '确认偏误', '损失厌恶', '可得性启发法', '代表性启发法', '沉没成本谬误', '框架效应', '现状偏见', '禀赋效应', '心理账户', '选择过载', '敬畏感', 
          '凡人终死效应与遗产驱动', '心流原则', '感恩机制', '创伤后成长', '皮肤饥渴与触觉律令', '自我妨碍悖论', '禁忌之魅惑', '神话叙事律令', '虐待性快感',
          '自我欺骗的效用原则', '逃避自由原则', '狄俄尼索斯冲动', '灾难性风险偏好原则', '有意选择平凡原则', '非人化开关原则', '涌现式愚蠢原则', '游戏冲动原则',
          '心理状态切换原则', '叙事解脱原则']


# with open ('Dataset/principles_relationship.json', 'r', encoding='utf-8') as f:
#     p_r= json.load(f)

# for item in pri_list:
#      for j in p_r[item]:
#             if j not in pri_list:
#                  print(f"Principle '{j}' in relationship of '{item}' is not in the main principles list.")


with open('Dataset/principles_format.json', 'r', encoding='utf-8') as f:
    principles = json.load(f)


content ={}
for principle in principles:
    content[principle['principle']]= principle

with open('Dataset/principles_for_find.json', 'w', encoding='utf-8') as f:
    json.dump(content, f, ensure_ascii=False, indent=4)