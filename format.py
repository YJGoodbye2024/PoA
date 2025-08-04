# import json
# import os 
# import re


# def get_pri(text):
#     """提取字符串中所有[]包含的内容"""
#     pattern = r'\[([^\]]*)\]'
#     return re.findall(pattern, text)

# # text="这是“人自身”领域中，[叙事自我] 为了创造连贯故事而采用的一种叙事手法。它通过让我们感觉过去是可预测的，从而加强了 [控制错觉] ，即我们高估了自己对世界的理解和预测能力。"
# # print(get_pri(text))

# def remove_parentheses_content(text):
#     """删除字符串中的()和（）及其包含的内容"""
#     # 同时匹配英文圆括号()和中文圆括号（）
#     pattern = r'\([^)]*\)|（[^）]*）'
#     result = re.sub(pattern, '', text)
#     return result

# def clean_bracket_content(text):
#     """
#     找到[]括起来的内容，删除其中的空格和圆括号内容，保留[]
#     """
#     def clean_content(match):
#         # 获取方括号内的内容
#         content = match.group(1)
        
#         # 删除圆括号及其内容（英文和中文）
#         content = re.sub(r'\([^)]*\)|（[^）]*）', '', content)
        
#         # 删除所有空格
#         content = re.sub(r'\s+', '', content)
        
#         # 返回清理后的内容，保留方括号
#         return f'[{content}]'
    
#     # 找到所有[...]内容并进行处理
#     pattern = r'\[([^\]]*)\]'
#     result = re.sub(pattern, clean_content, text)
    
#     return result


# with open ('PoA/Dataset/principles.json', 'r', encoding='utf-8') as f:
#     principles = json.load(f)

# principles_list=[]
# rela_dict = {}
# content=[]
# for i,item in enumerate(principles):
#     s=remove_parentheses_content(item["principle"])
#     item["principle"] = s
#     principles_list.append(s)
#     rela_list = get_pri(item["relationship"])
#     # print(rela_list)
#     for j, rela in enumerate(rela_list):
#         rela_list[j]= remove_parentheses_content(rela)
#     # print(rela_list)

#     rela_list = list(set(rela_list))  # 去重
#     # print(rela_list)

#     if item["principle"] in rela_list:
#         rela_list.remove(item["principle"])
#     content.append(item)
#     rela_dict[item["principle"]] = rela_list

# with open('PoA/Dataset/principles_format.json', 'w', encoding='utf-8') as f:
#     json.dump(content, f, ensure_ascii=False, indent=4)

# with open('PoA/Dataset/principles_relationship.json', 'w', encoding='utf-8') as f:
#     json.dump(rela_dict, f, ensure_ascii=False, indent=4)

# print(principles_list)
# print(len(principles_list))