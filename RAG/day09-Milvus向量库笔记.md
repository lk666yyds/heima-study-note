# Milvus 向量库学习笔记

> 基于 `day9` 系列 notebook（`01.Milvus`、`02.LangChain-Milvus`）整理，覆盖从定义 Schema、建索引、创建 Collection、保存数据，到混合检索与结果重排的完整流程，以及 LangChain-Milvus 的封装用法。

---

## 零、Milvus 核心概念回顾

Milvus 是专门做**向量相似度检索**的分布式向量数据库，三个核心概念的层级关系：

| 概念 | 关系型数据库类比 | 说明 |
|------|------------------|------|
| **Collection** | 表（Table） | 存储和检索数据的基本容器 |
| **Field** | 列（Column） | 含标量字段和向量字段 |
| **Entity** | 行（Row） | 一条完整记录 |

两种向量字段类型（也是今天学习重点）：

- **稠密向量（Dense Vector）**：`FLOAT_VECTOR`，由 Embedding 模型生成的高维浮点向量，做语义相似度检索。
- **稀疏向量（Sparse Vector）**：`SPARSE_FLOAT_VECTOR`，由分词后的词频（如 BM25）生成，做关键词匹配检索。

> **混合检索** = 稠密向量（语义） + 稀疏向量（关键词）两者结合，取长补短。

---

## 一、定义 Collection 的 Schema 约束

Schema 决定了 Collection 里有哪些字段、每个字段是什么类型、主键是谁。用 `client.create_schema()` 创建，再用 `add_field()` 逐个添加字段。

### 1. 创建 Schema 对象

```python
from pymilvus import MilvusClient, Function, FunctionType, DataType
from app.core.config import settings

client = MilvusClient(uri=settings.rag.milvus_url)

schema = client.create_schema(
    auto_id=False,            # 不自动生成 id，由我们自己指定主键值
    enable_dynamic_field=False # 是否允许插入未在 schema 中声明的动态字段
)
```

### 2. 添加各类字段

```python
# ① 主键字段：INT64 类型，is_primary=True 标记为主键
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)

# ② 稠密向量字段：必须指定维度 dim，与 Embedding 模型输出维度一致
schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=1024)

# ③ 稀疏向量字段：无需指定维度
schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

# ④ 普通元数据字段：VARCHAR，需指定最大长度，用于检索时做过滤
schema.add_field(field_name="h2", datatype=DataType.VARCHAR, max_length=128)

# ⑤ 原始文档内容字段：enable_analyzer=True 允许对该字段分词，用于生成 BM25 稀疏向量
schema.add_field(
    field_name="content",
    datatype=DataType.VARCHAR,
    max_length=2048,
    enable_analyzer=True,                 # 开启分词
    analyzer_params={"type": "chinese"}   # 中文分词
)
```

| 字段名 | 类型 | 作用 |
|--------|------|------|
| `id` | `INT64` | 主键，唯一标识一条 Entity |
| `dense` | `FLOAT_VECTOR` | 稠密向量，语义检索 |
| `sparse` | `SPARSE_FLOAT_VECTOR` | 稀疏向量，关键词检索 |
| `h2` | `VARCHAR` | 元数据，用于过滤 |
| `content` | `VARCHAR` + 分词 | 原始文档内容 |

### 3. 添加 BM25 函数（自动生成稀疏向量）

BM25 函数能把 `content` 字段的原始文本**自动转成稀疏向量**存到 `sparse` 字段，插入时就不需要自己算稀疏向量了：

```python
bm25_function = Function(
    name="text_bm25_emb",            # 函数名
    input_field_names=["content"],   # 输入字段：原始文本
    output_field_names=["sparse"],   # 输出字段：稀疏向量
    function_type=FunctionType.BM25, # 函数类型：BM25
)
schema.add_function(bm25_function)
```

> 这样插入数据时只传 `content`，Milvus 会自动完成分词 → BM25 稀疏向量的转换。

---

## 二、定义稀疏和稠密索引

索引是作用在向量字段上、用来**加速相似度检索**的结构。用 `prepare_index_params()` 创建索引参数容器，再逐个字段 `add_index()`。

### 1. 稠密向量索引

```python
index_params = client.prepare_index_params()

# 稠密向量索引：AUTOINDEX 自动选择最优索引，metric_type 指定相似度度量方式
index_params.add_index(
    field_name="dense",
    index_type="AUTOINDEX",
    metric_type="COSINE"   # 余弦相似度；也可用 IP（内积）、L2（欧氏距离）
)
```

### 2. 稀疏向量索引

```python
# 稀疏向量索引：采用倒排索引，metric_type 用 BM25
index_params.add_index(
    field_name="sparse",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",   # 支持 BM25、IP
)
```

### 3. 普通标量字段索引

```python
# 标量字段索引：用于加速过滤查询
index_params.add_index(field_name="h2", index_type="AUTOINDEX")
```

> 常见 `metric_type` 对比：
> - `COSINE`：余弦相似度，衡量方向，常用于文本向量（已归一化时等价于 `IP`）。
> - `IP`：内积，值越大越相似。
> - `L2`：欧氏距离，值越小越相似。

---

## 三、创建 Collection

Schema 和索引都准备好后，调用 `create_collection()` 一次性创建：

```python
def create_collection(collection_name):
    # 1. 定义 Schema（见第一节）
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="h2", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=2048,
                     enable_analyzer=True, analyzer_params={"type": "chinese"})
    bm25_function = Function(name="text_bm25_emb", input_field_names=["content"],
                             output_field_names=["sparse"], function_type=FunctionType.BM25)
    schema.add_function(bm25_function)

    # 2. 定义索引（见第二节）
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    index_params.add_index(field_name="h2", index_type="AUTOINDEX")

    # 3. 创建 Collection
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

collection_name = 'test_collection_1'
create_collection(collection_name)
```

### 查看 / 删除 Collection

```python
# 查看所有 Collection
res = client.list_collections()
print(res)  # ['test_collection_1']

# 删除 Collection（schema 不能原地加字段，改了结构要删掉重建）
client.drop_collection(collection_name='test_collection_1')
```

---

## 四、保存数据（Entity 操作）

### 1. 插入 insert

插入前先要把文档向量化，再组织成 `list[dict]`（每个 dict 是一条 Entity），`content` 字段会自动转稀疏向量：

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b", dimensions=1024)

# 1. 批量向量化：传入 list[str]，返回 list[list[float]]
vectors = embeddings.embed_documents([doc.page_content for doc in docs])

# 2. 组装数据：每个 dict 对应一条 Entity
data = [
    {
        "id": i + 1,                              # 主键
        "content": doc.page_content,               # 原始内容（会自动转 sparse）
        "h2": doc.metadata.get("h2", ""),          # 元数据
        "dense": vectors[i],                       # 稠密向量
    }
    for i, doc in enumerate(docs)
]

# 3. 插入
res = client.insert(collection_name, data)
print(res)  # {'insert_count': 21, 'ids': [1, 2, ..., 21]}
```

### 2. 更新或插入 upsert

`upsert` 按主键判断：**id 已存在则更新，不存在则插入**。

```python
# id=1 的数据修改 h2 字段 → 模拟更新
data1 = data[0]
data1['h2'] = "测试一下"

# id=2 的数据把 id 改为 88 → 模拟新增
data2 = data[1]
data2['id'] = 88

res = client.upsert(collection_name, [data1, data2])
print(res)  # {'upsert_count': 2, 'ids': [1, 88]}
```

### 3. 删除 delete

按主键 id 删除：

```python
res = client.delete(collection_name, ids=[1])
print(res)  # {'delete_count': 1}
```

---

## 五、检索（Search）

### 稠密向量检索

查询时先把问题向量化，再指定 `anns_field`（检索哪个向量字段）搜索：

```python
def dense_query(query: str) -> list[list[dict]]:
    # 1. 问题向量化
    query_vector = embeddings.embed_query(text=query)
    # 2. 相似度检索
    return client.search(
        collection_name=collection_name,        # 在哪个集合中查
        data=[query_vector],                    # 查询向量（可传多个）
        limit=5,                                # 返回的文档数量
        output_fields=["id", "h2", "content"],  # 返回哪些字段
        search_params={"metric_type": "COSINE"},# 检索度量类型
        anns_field='dense'                      # 检索的向量字段
    )

result = dense_query('教育的负面作用是什么')
for hits in result:
    for hit in hits:
        print(hit)
```

> ⚠️ 注意：`output_fields` 里列出的字段**必须在 Schema 中已定义**，否则报 `field xxx not exist`（code=1100）。切分器如果产出了 `h1`/`h3` 元数据，也要先在 `add_field` 里声明才能查询返回。

---

## 六、LangChain-Milvus 封装

LangChain 提供了 `langchain_milvus` 包，把 Milvus 封装成 `VectorStore` 对象，使用方式和其他 LangChain 向量库（如 `InMemoryVectorStore`）一致，省去手写 Schema/索引的麻烦。

### 1. 初始化 VectorStore

```python
from langchain_milvus import Milvus, BM25BuiltInFunction
from app.core.config import settings

vector_store = Milvus(
    embedding_function=embeddings,               # 稠密向量模型
    collection_name="langchain_collection",      # Collection 名称
    builtin_function=BM25BuiltInFunction(        # 生成稀疏向量的函数
        analyzer_params={"type": "chinese"}      # 中文分词
    ),
    vector_field=["dense", "sparse"],            # 向量字段：稠密 + 稀疏
    connection_args={
        "uri": settings.rag.milvus_url,          # Milvus 地址
    },
    drop_old=True,                               # 是否删除旧 Collection，避免重复创建
)
```

> `Milvus` 会自动完成 Schema 定义、稀疏/稠密索引、建 Collection 等底层工作，只需关注业务字段。

### 2. 批量新增文档

```python
# 按每批 20 个分批添加
batch_documents = [docs[i:i + 20] for i in range(0, len(docs), 20)]

for batch in batch_documents:
    result = vector_store.add_documents(documents=batch)
    print(result)
```

### 3. 删除文档

```python
# 传入文档 id 列表删除
vector_store.delete([468847736313318163])
```

---

## 七、混合检索

混合检索 = **稠密向量（语义相似）+ 稀疏向量（BM25 关键词）** 两个检索结果融合。LangChain-Milvus 通过 `similarity_search_with_score()` 配合 `ranker_type` 实现，返回 `(文档, 得分)` 二元组。

### 1. 加权混合检索（weighted）

给稠密和稀疏两路结果分配不同权重，线性加权融合：

```python
result = vector_store.similarity_search_with_score(
    query='教育的负面作用是什么',
    k=3,                                  # 最终返回的 TopK
    fetch_k=5,                            # 初筛时每路召回的文档数量
    ranker_type='weighted',
    ranker_params={'weights': [0.5, 0.5]} # [稠密权重, 稀疏权重]
)

for document, score in result:
    print(f'得分:{score}')
    print(document.model_dump_json(indent=2))
```

### 2. RRF 混合检索（Reciprocal Rank Fusion）

基于**排名倒数**融合，不关心具体得分只看排名，对异常分数更鲁棒：

```python
result = vector_store.similarity_search_with_score(
    query='教育的负面作用是什么',
    k=3,
    fetch_k=5,
    ranker_type='rrf',
    ranker_params={'k': 60}   # RRF 平滑参数
)
```

### 3. 过滤搜索

通过 `expr` 表达式对元数据字段做过滤，先过滤再检索：

```python
result = vector_store.similarity_search_with_score(
    query='教育的负面作用是什么',
    k=5,
    fetch_k=5,
    ranker_type='rrf',
    ranker_params={'k': 60},
    expr='h2 like "%第一节%"'   # 过滤条件：h2 字段包含"第一节"
)
```

---

## 八、查询结果重排（Cross-Encoder Rerank）

初筛（`fetch_k`）召回一批候选后，用一个 **Cross-Encoder 重排模型**（如阿里云 `gte-rerank-v2`）对候选做精排，比纯向量相似度更准确地判断"查询-文档"相关性，最终只返回精排后的 TopK。

### 1. 定义 reranker 函数

```python
from pymilvus import Function, FunctionType

def create_cross_encoder_ranker(queries: list[str]):
    return Function(
        name="dashscope_semantic_ranker",   # reranker 名称，唯一即可
        input_field_names=["text"],         # 原始文档字段
        function_type=FunctionType.RERANK,  # 函数类型：RERANK
        params={
            "reranker": "model",            # 基于模型 rerank（cross-encoder）
            "provider": "ali",              # rerank 模型提供者
            "model_name": "gte-rerank-v2",  # rerank 模型名称
            "queries": queries,             # 查询条件
            "max_client_batch_size": 5,     # 向模型发请求的批处理限制
        },
    )
```

### 2. 混合检索 + 重排

```python
def cross_encoder_hybrid_search(query: str, filter: str = ""):
    return vector_store.similarity_search_with_score(
        query,
        k=3,                                   # 精排后最终返回的 TopK
        fetch_k=5,                             # 初筛召回的候选数
        expr=filter,                           # 可选过滤条件
        reranker=create_cross_encoder_ranker([query])  # 传入重排器
    )

query = "教育的负面作用是什么"
result = cross_encoder_hybrid_search(query, 'h2 like "%教育的功能%"')

for document, score in result:
    print(f'得分:{score}')
    print(document.model_dump_json(indent=2))
```

> 流程总结：`fetch_k` 初筛召回 → `reranker`（Cross-Encoder）精排 → 返回 `k` 条最相关结果。

---

## 九、原生 pymilvus 与 LangChain-Milvus 对比

| 维度 | 原生 pymilvus | LangChain-Milvus |
|------|---------------|------------------|
| Schema | 手动 `add_field` 逐个定义 | 自动生成 |
| 索引 | 手动 `prepare_index_params` + `add_index` | 自动生成 |
| 建 Collection | 手动 `create_collection` | 初始化 `Milvus` 时自动建 |
| 插入 | `client.insert(collection_name, data)` | `vector_store.add_documents(docs)` |
| 检索 | `client.search(...)` | `vector_store.similarity_search_with_score(...)` |
| 混合检索 / 重排 | 需自行组合 | 内置 `ranker_type`、`reranker` 参数 |
| 适用场景 | 需要精细控制底层细节 | 与 LangChain 生态集成、快速开发 |

---

## 十、核心要点总结

| 目标 | 关键 API / 参数 |
|------|----------------|
| 定义 Schema | `create_schema()` + `add_field()`，主键 `is_primary=True`，稠密 `dim`，内容 `enable_analyzer` |
| 生成稀疏向量 | `Function(function_type=FunctionType.BM25)` + `add_function()` |
| 稠密索引 | `add_index(index_type="AUTOINDEX", metric_type="COSINE")` |
| 稀疏索引 | `add_index(index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")` |
| 创建 Collection | `client.create_collection(collection_name, schema, index_params)` |
| 保存数据 | `insert`（新增）、`upsert`（存在更新/不存在插入）、`delete`（按 id） |
| 稠密检索 | `client.search(data=[query_vector], anns_field='dense', output_fields=[...])` |
| 混合检索 | `similarity_search_with_score(ranker_type='weighted'/'rrf', fetch_k=..., k=...)` |
| 过滤检索 | `expr='h2 like "%xx%"'` |
| 结果重排 | `reranker=Function(function_type=FunctionType.RERANK)` + `gte-rerank-v2` |
| LangChain-Milvus | `Milvus(embedding_function, collection_name, builtin_function, vector_field, connection_args, drop_old)` |
