

import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import time
import uuid
import json

client_1 = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),  
    base_url="https://api.deepseek.com"
)



def generate_scenes_summary(generated_scenario_json):

    summary_prompt = f"请用一句话，高度凝练地概括以下场景的核心剧情冲突。聚焦于「谁，在什么样的场景下，遇到了什么困境」。\n\n场景JSON:\n{generated_scenario_json}"
    response = client_1.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一位专业的故事梗概作者。"},
        {"role": "user", "content": summary_prompt},
    ],
    stream=False
    )

    summary_text = response.choices[0].message.content

    return summary_text


# --- 配置客户端 ---
# 1. 配置Dashscope (Qwen)
client_2 = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)
QWEN_EMBEDDING_DIMENSION=1024

# 2. 配置Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# 3. 检查并创建Pinecone索引
PINECONE_INDEX_NAME = "scenario-memory-db" # 给你的索引起个名字

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"索引 '{PINECONE_INDEX_NAME}' 不存在，正在创建...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=QWEN_EMBEDDING_DIMENSION, # 维度必须匹配Qwen模型
        metric="cosine", # 使用余弦相似度进行搜索，适合文本语义
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    )
    print("索引创建成功！")
else:
    print(f"索引 '{PINECONE_INDEX_NAME}' 已存在。")

# 4. 连接到你的索引
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

print("\n初始化完成，所有客户端已准备就绪。")




def get_embedding_qwen(text):
    """使用Qwen模型为文本生成向量嵌入"""
    try:
        completion = client_2.embeddings.create(
            model="text-embedding-v4",
            input=text,
            dimensions=QWEN_EMBEDDING_DIMENSION,
            encoding_format="float"
        )

        # 方法1: 直接访问对象属性
        embedding = completion.data[0].embedding
        
        # 或者方法2: 如果需要使用字典形式
        # res_dict = completion.model_dump()  # 注意是 model_dump() 而不是 model_dump_json()
        # embedding = res_dict['data'][0]['embedding']
        
        return embedding
    except Exception as e:
        print(f"Qwen嵌入生成失败: {e}")
        return None

def add_scenario_to_memory(generated_scenario_json):
    """将一个新生成的场景摘要并存入Pinecone记忆数据库"""
    
    # 1. 生成摘要 
    summary_text = generate_scenes_summary(generated_scenario_json)
    summary_text = f"一个关于'{summary_text}'的故事" # 伪实现，用于演示
    print(f"生成摘要: {summary_text}")

    # 2. 生成向量嵌入
    embedding_vector = get_embedding_qwen(summary_text)
    if not embedding_vector:
        return
    
    print(f"generated_scenario_json的内容如下：{generated_scenario_json}")
    # 3. 准备元数据
    metadata = {
        "summary": summary_text,
        "principle_name": generated_scenario_json["principleName"][0], 
        "scenario_title": generated_scenario_json["scenarioTitle"],
        "creation_timestamp": int(time.time())
    }
    
    # 4. 生成唯一ID并存入Pinecone
    scenario_id = str(uuid.uuid4())

    try:
        pinecone_index.upsert(
            vectors=[
                {"id": scenario_id, "values": embedding_vector, "metadata": metadata}
            ]
        )
        print(f"✅ 场景 '{scenario_id}' 的记忆已成功存入Pinecone。")
    except Exception as e:
        print(f"❌ 存入Pinecone失败: {e}")

def find_similar_scenarios(principle_json, top_k=3):
    """根据原则描述，在Pinecone中检索最相似的场景摘要"""
    
    # 1. 确定查询内容
    principle_description = principle_json["description"]
    principle_name = str(principle_json["principle"])
    
    # 2. 生成查询向量
    print(f"\n正在为原则 '{principle_name}' 生成查询向量...")
    query_vector = get_embedding_qwen(principle_description)
    if not query_vector:
        return []

    # 3. 执行向量搜索，并使用元数据过滤
    print(f"正在Pinecone中检索与 '{principle_name}' 最相似的 {top_k} 个场景...")
    try:
        search_results = pinecone_index.query(
            vector=query_vector,
            top_k=top_k,
            filter={
                # 关键过滤：只在同样原则的场景中寻找相似项
                "principle_name": {"$eq": f"['{principle_name}']"} 
            },
            include_metadata=True
        )
        
        # 4. 提取排斥摘要
        exclusion_summaries = [match['metadata']['summary'] for match in search_results['matches']]
        print(f"检索到 {len(exclusion_summaries)} 条相似摘要：")
        for summary in exclusion_summaries:
            print(f"- {summary}")
        return exclusion_summaries
    except Exception as e:
        print(f"❌ 从Pinecone检索失败: {e}")
        return []
    

 