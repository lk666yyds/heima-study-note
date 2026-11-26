# LangChain 实现 RAG 技术笔记

> 基于 `day08` 系列 notebook 整理，覆盖从文档加载、拆分、向量化到检索问答的完整 RAG 流程。

---

## 一、LangChain 实现 RAG 的代码流程

RAG（Retrieval-Augmented Generation，检索增强生成）的核心思路：**先从外部知识库检索相关文档片段，再把片段和用户问题一起交给大模型生成答案**，从而解决模型知识更新困难、事实性错误、缺乏专业领域知识等问题。

LangChain 实现 RAG 的整体流程分为**离线构建知识库**和**在线知识检索问答**两个阶段：

```text
┌───────────────────── 离线阶段：构建知识库 ─────────────────────┐
│                                                              │
│  ① 文档加载  →  ② 文档拆分  →  ③ 向量化  →  ④ 存入向量库      │
│  (PyPDFLoader) (TextSplitter) (Embeddings)  (VectorStore)    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────── 在线阶段：知识检索问答 ───────────────────┐
│                                                              │
│  ① 用户提问  →  ② 相似度检索  →  ③ 拼接提示词  →  ④ 模型生成   │
│               (similarity_search)  (context+query)   (LLM)   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

一个最简 RAG 完整代码骨架如下（对应 `01.RAG预览`）：

```python
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# ===== 一、构建知识库阶段 =====
# 1. 加载文档
loader = PyPDFLoader(file_path="resources/贵州茅台研报.pdf", mode="single")
docs = loader.load()

# 2. 切分文档
splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# 3. 向量化 + 4. 存入向量库
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b", dimensions=1024)
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)

# ===== 二、在线问答阶段 =====
model = init_chat_model("deepseek-v4-flash",
                        extra_body={"thinking": {"type": "disabled"}})

def try_rag(query: str):
    # 1. 检索文档
    retrieved_docs = vectorstore.similarity_search(query, k=2)
    # 2. 拼接上下文提示词
    content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = f"""你基于我提供的报告回答用户问题，报告中没提及的就说不知道，不要自己编造答案。
    report: ```{content}```
    query: {query}"""
    # 3. 调用模型生成答案
    return model.invoke(prompt).content

print(try_rag("茅台收盘价多少"))
```

---

## 二、构建知识库（离线阶段）

知识库构建由四个步骤组成：**文档加载 → 文档拆分 → 向量化 → 存入向量库**。

### 1. 文档加载

#### 1.1 PyPDFLoader 加载 PDF（了解）

`PyPDFLoader` 来自 `langchain_community.document_loaders`，返回 `list[Document]`，每个 `Document` 包含 `page_content`（正文）和 `metadata`（元数据）两个属性。

```python
from langchain_community.document_loaders import PyPDFLoader

# 初始化并配置文档加载器
loader = PyPDFLoader(
    file_path='resources/贵州茅台研报.pdf',
    mode="page"  # single: 整个文件作为 1 个文档；page：按 PDF 分页
)

docs = loader.load()
print(f'文档数量: {len(docs)}')

for i, doc in enumerate(docs):
    print(doc.metadata)      # 元数据
    print(doc.page_content)  # 正文内容
```

#### 1.2 MinerU 加载（更高解析质量）

MinerU 支持 Flash（免 Token）、Precision（需 Token）和 OCR 三种模式，解析结果可直接导出 Markdown：

```python
from mineru import MinerU

# Flash 模式，无需 Token
client = MinerU()
result = client.flash_extract("./resources/贵州茅台研报.pdf")
print(result.markdown)
result.save_markdown("./resources/r2.md")

# Precision 模式，需要 Token
from app.core.config import settings
client = MinerU(settings.rag.mineru_token)
result = client.extract("./resources/贵州茅台研报.pdf")
result.save_markdown("./resources/r3.md", with_images=True)  # 保存图片

# OCR 功能
result = client.extract('./resources/small_ocr.pdf', ocr=True)
```

### 2. 文档拆分（了解）

#### 2.1 固定长度拆分 `CharacterTextSplitter`

使用指定单个分隔符分割，按固定字符数或 token 数切块。

```python
from langchain_text_splitters import CharacterTextSplitter

# 按字符数拆分
text_splitter = CharacterTextSplitter(
    separator='',      # 分隔符，空表示任意位置都可分隔（固定块大小）
    chunk_size=200,    # 每个块的大小
    chunk_overlap=40   # 相邻块重叠大小（滑动窗口）
)
chunks = text_splitter.split_text(doc)

# 按 token 数拆分
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name='cl100k_base',  # 编码器
    separator='',
    chunk_size=200,    # 200 token
    chunk_overlap=40
)
chunks = text_splitter.split_text(doc)
```

#### 2.2 递归拆分 `RecursiveCharacterTextSplitter`

持有一个分隔符优先级列表，递归尝试不同粒度的分隔符，是处理结构复杂文本的**官方推荐默认拆分器**。

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    separators=['\n\n', '\n', '。', '!', ""],  # 按优先级从大到小尝试
)
chunks = recursive_splitter.split_text(doc)
```

#### 2.3 结构感知拆分 `MarkdownHeaderTextSplitter`

按 Markdown 标题层级拆分，保留文档结构信息。

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

text_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ('#', 'header 1'),
        ('##', 'header 2'),
        ('###', 'header 3'),
    ],
    strip_headers=False  # 是否自动去除块正文中的标题
)
docs = text_splitter.split_text(md_doc)
```

### 3. 向量化（Embedding）

#### 3.1 OllamaEmbeddings（本地）

```python
from langchain_ollama import OllamaEmbeddings

ollama_embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",  # 性价比高的模型
    dimensions=1024                # 可选：减少维度以节省存储
)

# 批量向量化
vectors = ollama_embeddings.embed_documents(["我要躺平", "我爱工作", "拒绝加班"])
# 单条向量化（用于 query）
vector = ollama_embeddings.embed_query("我爱上班")
```

#### 3.2 OpenAIEmbeddings（阿里云百炼，OpenAI 规范）

```python
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
import os
from pydantic import SecretStr

dashscope_base_url = os.getenv('DASHSCOPE_BASE_URL')
dashscope_api_key = os.getenv('DASHSCOPE_API_KEY', '')

embeddings = OpenAIEmbeddings(
    base_url=dashscope_base_url,
    api_key=SecretStr(dashscope_api_key),
    model='qwen3.7-text-embedding-flash',
    dimensions=1024,
    check_embedding_ctx_length=False  # 关闭向量文本长度检查
)
```

#### 3.3 衡量相似度（余弦相似度）

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """余弦相似度：点积 / (模长乘积)"""
    dot = np.dot(vec1, vec2)
    return dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

for i, v in enumerate(vectors):
    print(f"{texts[i]}:", cosine_similarity(vector, v))
```

### 4. 向量数据库

LangChain 中向量库统一抽象为 `VectorStore`，本例使用内存向量库 `InMemoryVectorStore`。

```python
from langchain_core.vectorstores import InMemoryVectorStore

# 初始化向量库，指定向量模型
vector_store = InMemoryVectorStore(embedding=embeddings)

# 添加文档（自动完成向量化）
vector_store.add_documents(documents)

# 批量添加（每批 20 个）
batch_documents = [documents[i:i + 20] for i in range(0, len(documents), 20)]
for batch in batch_documents:
    ids = vector_store.add_documents(batch)
    print(f'文档 ID 列表: {ids}')

# 删除文档
vector_store.delete(["doc_1", "doc_2"])
```

> 也可用 `InMemoryVectorStore.from_documents(chunks, embeddings)` 一步完成「向量化 + 入库」。

---

## 三、RAG 知识检索（在线阶段）

### 1. 相似度检索

```python
query = "论语中教育的目的是什么"

# 相似度检索，返回 list[Document]
results = vector_store.similarity_search(query=query, k=5)

for doc in results:
    print(doc.model_dump_json(indent=2))
```

### 2. 基础 RAG 问答

检索到片段后，拼接成上下文提示词，交给模型生成答案：

```python
from langchain.chat_models import init_chat_model
from app.core.config import settings

model = init_chat_model(
    "deepseek-v4-flash",
    api_key=settings.llm.api_key,
    extra_body={'thinking': {'type': 'disabled'}}
)

def rag_chat(user_input: str):
    # 1. 检索知识片段
    docs = vector_store.similarity_search(user_input, k=2)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 2. 拼接提示词
    prompt = f'''
    你是一个教资考试专家，你根据文档回答用户问题，文档中没有的不要编造，直接说不知道。
    回答直接引用文档中原文，不要自己发挥！
    context: {context}
    query: {user_input}
    '''

    # 3. 调用模型
    return model.invoke(prompt).content

print(rag_chat("论语中教育的目的是什么？"))
```

### 3. Agentic RAG（工具调用）

把「检索」封装成工具 `@tool`，交给 Agent 自主决定何时调用：

```python
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def retrieve_docs(user_input: str) -> str:
    """知识库检索，输入用户问题，返回可用于回答的上下文片段。"""
    docs = vector_store.similarity_search(user_input, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context

agent = create_agent(
    model=model,
    tools=[retrieve_docs],
    system_prompt='你是一个教资考试专家，你需要根据提供的上下文回答用户的问题，不能自己编造答案。答案要简短，直接引用原文'
)

response = agent.invoke({'messages': [HumanMessage(content='你好!')]})
print(response['messages'][-1].text)
```

### 4. Hybrid 混合 RAG 检索（扩展）

在工具内部加入**查询改写（增强）**和**检索充分性判断**，检索不充分时自动改写并加大召回数量重试，提升检索质量：

```python
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

def enhance_query(original: str) -> str:
    """把用户问题改写为更适合向量检索的查询"""
    prompt = [
        SystemMessage(content='把用户问题改写为更适合向量检索的查询，只返回改写结果。'),
        HumanMessage(content=original),
    ]
    return model.invoke(prompt).content.strip()

def is_sufficient(original: str, context: str) -> bool:
    """判断上下文能否回答用户问题"""
    prompt = [
        SystemMessage(content='上下文能否回答用户问题？只回答 yes 或 no。'),
        HumanMessage(content=f'问题：{original}\n\n上下文：\n{context}'),
    ]
    return model.invoke(prompt).content.strip().lower().startswith('yes')

@tool
def retrieve_docs(user_input: str) -> str:
    """知识库检索。输入用户问题，返回可用于回答的上下文片段。"""
    query = enhance_query(user_input)                      # 查询改写
    docs = vector_store.similarity_search(query, k=5)      # 向量检索
    context = "\n\n".join(d.page_content for d in docs)

    attempt = 1
    # 检索不充分时重试（改写 + 增加召回数量），最多 3 次
    while not is_sufficient(user_input, context) and attempt < 3:
        query = enhance_query(f"{user_input}（请给出更具体的关键词）")
        docs = vector_store.similarity_search(query, k=5 + attempt)
        context = "\n\n".join(d.page_content for d in docs)
        attempt += 1

    return context
```

---

## 四、核心要点总结

| 阶段 | 步骤 | 关键类 / 方法 |
|------|------|--------------|
| 构建知识库 | 文档加载 | `PyPDFLoader`、`MinerU` |
| 构建知识库 | 文档拆分 | `CharacterTextSplitter`、`RecursiveCharacterTextSplitter`、`MarkdownHeaderTextSplitter` |
| 构建知识库 | 向量化 | `OllamaEmbeddings`、`OpenAIEmbeddings` |
| 构建知识库 | 存入向量库 | `InMemoryVectorStore` |
| 知识检索 | 相似度检索 | `vector_store.similarity_search(query, k)` |
| 知识检索 | 问答 | `init_chat_model` + `model.invoke(prompt)` |
| 知识检索（进阶） | Agentic RAG | `@tool` + `create_agent` |
| 知识检索（进阶） | Hybrid RAG | 查询改写 + 充分性判断 + 重试 |
