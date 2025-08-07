import os
from pinecone import Pinecone, ServerlessSpec
import uuid
import time
from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)
QWEN_EMBEDDING_DIMENSION = 1024

# 2. 配置Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# 3. 检查并创建Pinecone索引
PINECONE_INDEX_NAME = "scenario-memory-db"  # 给你的索引起个名字

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"索引 '{PINECONE_INDEX_NAME}' 不存在，正在创建...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=QWEN_EMBEDDING_DIMENSION,  # 维度必须匹配Qwen模型
        metric="cosine",  # 使用余弦相似度进行搜索，适合文本语义
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print("索引创建成功！")
else:
    print(f"索引 '{PINECONE_INDEX_NAME}' 已存在。")

# 4. 连接到你的索引
index = pc.Index(PINECONE_INDEX_NAME)

print("\n初始化完成，所有客户端已准备就绪。")


def get_embedding(text):
    """生成文本的向量嵌入"""
    completion = client.embeddings.create(
        model="text-embedding-v4",
        input=text,
        dimensions=1024,
        encoding_format="float"
    )
    return completion.data[0].embedding

# ===== 1. 增加 (CREATE) =====


def add_single_vector(text, metadata=None):
    """添加单个向量"""
    vector_id = str(uuid.uuid4())
    embedding = get_embedding(text)

    if metadata is None:
        metadata = {"text": text, "created_at": int(time.time())}

    index.upsert(vectors=[{
        "id": vector_id,
        "values": embedding,
        "metadata": metadata
    }])

    print(f"✅ 向量已添加，ID: {vector_id}")
    return vector_id


def add_multiple_vectors(data_list):
    """批量添加向量"""
    vectors = []

    for data in data_list:
        vector_id = str(uuid.uuid4())
        text = data.get("text", "")
        metadata = data.get("metadata", {})
        metadata["created_at"] = int(time.time())

        embedding = get_embedding(text)

        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        })

    # 批量插入（建议每批不超过100个）
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"✅ 已添加第 {i//batch_size + 1} 批，共 {len(batch)} 个向量")

    return [v["id"] for v in vectors]

# ===== 2. 查询 (READ) =====


def query_by_text(query_text, top_k=5, filter_dict=None, include_metadata=True):
    """根据文本查询最相似的向量"""
    query_embedding = get_embedding(query_text)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        filter=filter_dict,
        include_metadata=include_metadata,
        include_values=False  # 通常不需要返回向量值
    )

    print(f"🔍 查询到 {len(results['matches'])} 个相似向量:")
    for match in results['matches']:
        print(f"  ID: {match['id']}")
        print(f"  相似度: {match['score']:.4f}")
        if include_metadata:
            print(f"  元数据: {match['metadata']}")
        print("---")

    return results


def query_by_id(vector_ids):
    """根据ID获取向量信息"""
    if isinstance(vector_ids, str):
        vector_ids = [vector_ids]

    results = index.fetch(ids=vector_ids)

    print(f"🔍 获取到 {len(results['vectors'])} 个向量:")
    for vid, vector_data in results['vectors'].items():
        print(f"  ID: {vid}")
        print(f"  元数据: {vector_data.get('metadata', {})}")
        print("---")

    return results


def get_index_stats():
    """获取索引统计信息"""
    stats = index.describe_index_stats()
    print("📊 索引统计信息:")
    print(f"  总向量数: {stats['total_vector_count']}")
    print(f"  维度: {stats['dimension']}")
    print(f"  索引满载情况: {stats['index_fullness']}")
    if 'namespaces' in stats:
        print(f"  命名空间: {list(stats['namespaces'].keys())}")
    return stats

# ===== 3. 更新 (UPDATE) =====


def update_vector_metadata(vector_id, new_metadata):
    """更新向量的元数据（保留原有向量值）"""
    # 先获取原向量
    result = index.fetch(ids=[vector_id])

    if vector_id not in result['vectors']:
        print(f"❌ 向量 {vector_id} 不存在")
        return False

    # 使用 upsert 更新（相同 ID 会覆盖）
    index.upsert(vectors=[{
        "id": vector_id,
        "values": result['vectors'][vector_id]['values'],
        "metadata": new_metadata
    }])

    print(f"✅ 向量 {vector_id} 的元数据已更新")
    return True


def update_vector_completely(vector_id, new_text, new_metadata=None):
    """完全更新向量（重新生成嵌入和元数据）"""
    new_embedding = get_embedding(new_text)

    if new_metadata is None:
        new_metadata = {"text": new_text, "updated_at": int(time.time())}

    index.upsert(vectors=[{
        "id": vector_id,
        "values": new_embedding,
        "metadata": new_metadata
    }])

    print(f"✅ 向量 {vector_id} 已完全更新")
    return True

# ===== 4. 删除 (DELETE) =====


def delete_by_ids(vector_ids):
    """根据ID删除向量"""
    if isinstance(vector_ids, str):
        vector_ids = [vector_ids]

    index.delete(ids=vector_ids)
    print(f"🗑️  已删除 {len(vector_ids)} 个向量")


def delete_by_filter(filter_dict):
    """根据元数据过滤条件删除向量"""
    index.delete(filter=filter_dict)
    print(f"🗑️  已根据过滤条件删除向量: {filter_dict}")


def delete_all_vectors():
    """删除索引中的所有向量（谨慎使用！）"""
    index.delete(delete_all=True)
    print("🗑️  ⚠️  所有向量已被删除！")

# ===== 高级查询功能 =====


def advanced_search(query_text, filters=None, top_k=10):
    """高级搜索示例"""
    query_embedding = get_embedding(query_text)

    # 复杂的过滤条件示例
    if filters is None:
        filters = {
            "$and": [
                {"domain": {"$eq": "职场与事业"}},
                {"principle_name": {"$eq": "生存指令"}},
                {"creation_timestamp": {"$gte": int(
                    time.time()) - 86400}}  # 24小时内
            ]
        }

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        filter=filters,
        include_metadata=True
    )

    return results


def search_with_namespace(query_text, namespace="default", top_k=5):
    """在特定命名空间中搜索"""
    query_embedding = get_embedding(query_text)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )

    return results

# ===== 使用示例 =====


if __name__ == "__main__":

    # # 1. 添加单个向量
    # vector_id = add_single_vector(
    #     text="这是一个测试场景",
    #     metadata={
    #         "principle_name": "测试原则",
    #         "domain": "测试领域",
    #         "scenario_title": "测试场景"
    #     }
    # )

    # # 2. 批量添加
    # test_data = [
    #     {
    #         "text": "职场压力场景1",
    #         "metadata": {"domain": "职场与事业", "principle_name": "生存指令"}
    #     },
    #     {
    #         "text": "恋爱关系场景1",
    #         "metadata": {"domain": "恋爱与婚姻", "principle_name": "社会认同"}
    #     }
    # ]
    # batch_ids = add_multiple_vectors(test_data)

    # # 3. 查询
    # results = query_by_text(
    #     query_text="职场中的压力情况",
    #     top_k=3,
    #     filter_dict={"domain": {"$eq": "职场与事业"}}
    # )

    # # 4. 根据ID获取
    # vector_info = query_by_id(vector_id)

    # # 5. 更新元数据
    # update_vector_metadata(vector_id, {
    #     "principle_name": "更新后的原则",
    #     "domain": "更新后的领域",
    #     "updated": True
    # })

    # # 6. 删除向量
    # delete_by_ids(vector_id)

    # # 7. 根据条件删除
    # delete_by_filter({"domain": {"$eq": "测试领域"}})

    # # 8. 获取统计信息
    # get_index_stats()

    # # 删除所有向量
    delete_all_vectors()

    get_index_stats()

# ===== Pinecone 过滤操作符参考 =====
"""
常用过滤操作符：

1. 等于: {"field": {"$eq": "value"}}
2. 不等于: {"field": {"$ne": "value"}}
3. 大于: {"field": {"$gt": 100}}
4. 大于等于: {"field": {"$gte": 100}}
5. 小于: {"field": {"$lt": 100}}
6. 小于等于: {"field": {"$lte": 100}}
7. 在范围内: {"field": {"$in": ["value1", "value2"]}}
8. 不在范围内: {"field": {"$nin": ["value1", "value2"]}}

逻辑操作符：
1. 与: {"$and": [condition1, condition2]}
2. 或: {"$or": [condition1, condition2]}

示例复杂过滤：
{
    "$and": [
        {"domain": {"$eq": "职场与事业"}},
        {"$or": [
            {"principle_name": {"$eq": "生存指令"}},
            {"principle_name": {"$eq": "社会认同"}}
        ]},
        {"creation_timestamp": {"$gte": 1640995200}}
    ]
}
"""
