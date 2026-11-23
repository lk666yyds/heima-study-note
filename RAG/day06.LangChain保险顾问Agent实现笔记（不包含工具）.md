# 用 LangChain 框架实现保险顾问 Agent（学习笔记）

> 项目：hm-insurance / agent-service
> 核心库：LangChain（`create_agent`）+ LangGraph（`CompiledStateGraph`）+ FastAPI + SQLAlchemy(async) + psycopg(async)
> 四大目标：
> 1. PostgreSQL 实现 Agent 短期记忆
> 2. FastAPI 管理 Agent 生命周期
> 3. 基于 SSE 实现前端流式对话
> 4. 利用 Checkpointer 实现会话历史管理

---

## 一、整体架构总览

```
前端 (Vue, localhost:5173)
        │  HTTP / SSE
        ▼
FastAPI 应用 (main.py)
   ├── lifespan 生命周期：启动时初始化数据库、Checkpointer、Agent
   ├── 路由层 (router)     —— 接收请求、依赖注入
   ├── 业务层 (service)    —— 编排业务逻辑
   ├── 数据访问层 (repository) —— 操作 chat_threads 表
   └── Agent (CompiledStateGraph) ──绑定──▶ Checkpointer (AsyncPostgresSaver)
                                                      │
                                                      ▼
                                            PostgreSQL (checkpoints 表)
```

**分层结构（`app/` 目录）：**

| 目录 | 职责 |
|------|------|
| `app/core/` | 配置(`config.py`)、日志、异常基类 |
| `app/infra/` | 基础设施：`database.py`(业务库)、`checkpointer.py`(检查点) |
| `app/agents/` | Agent 定义：`insurance_advisor.py` |
| `app/common/` | 通用 ORM 基类、时间戳混入 |
| `app/modules/<模块>/` | 按业务分模块，每个模块内含 `router / service / repository / schemas / models` |

---

## 二、目标一：PostgreSQL 实现 Agent 短期记忆

**「短期记忆」的本质**：Agent 每执行一步，LangGraph 会把当前状态（尤其是 `messages` 列表）自动写入 PostgreSQL 的 `checkpoints` 表；下次带着同一个 `thread_id` 调用时，从表里读回历史消息拼到上下文，Agent 就「记得」之前聊了什么。

### 2.1 配置连接串（`app/core/config.py`）

```python
@computed_field
@property
def checkpoint_url(self) -> str:
    """checkpointer 使用的连接字符串"""
    return self.build_url("postgresql")   # psycopg 用 postgresql:// 协议

@computed_field
@property
def url(self) -> str:
    """SQLAlchemy 使用的连接字符串"""
    return self.build_url("postgresql+asyncpg")  # SQLAlchemy 用 asyncpg 驱动
```

> 关键点：同一个数据库，但 **两套连接驱动**。
> - 业务库（SQLAlchemy）走 `postgresql+asyncpg`
> - Checkpointer（psycopg）走 `postgresql`

### 2.2 业务库引擎（`app/infra/database.py`）

```python
engine = create_async_engine(DATABASE_URL, pool_size=..., pool_pre_ping=True, ...)
AsyncSessionFactory = async_sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

async def get_session():
    """每个请求一个独立 Session，用于依赖注入"""
    async with AsyncSessionFactory() as session:
        yield session
```

### 2.3 Checkpointer 专用连接池（`app/infra/checkpointer.py`）

```python
checkpoint_pool = AsyncConnectionPool(
    conninfo=settings.db.checkpoint_url,
    min_size=1, max_size=5,
    kwargs={
        "autocommit": True,          # 自动提交
        "prepare_threshold": 0,      # 不做 prepared statement 缓存
        "row_factory": dict_row,     # 结果以 dict 返回
    },
    open=False,                      # 先不建连接，启动时手动 open
)

async def init_checkpointer() -> AsyncPostgresSaver:
    await checkpoint_pool.open()     # 真正建立连接
    await checkpoint_pool.wait()     # 等连接就绪
    checkpointer = AsyncPostgresSaver(checkpoint_pool)
    await checkpointer.setup()       # 自动建 checkpoints 等表
    return checkpointer
```

> **为什么用连接池？** 每次连库很贵；池子里预维护一组连接，用时取出、用完归还。
> **为什么 `autocommit=True`？** LangGraph 检查点写入需要即时落库，不手动管事务。

---

## 三、目标二：FastAPI 管理 Agent 生命周期

生命周期用 FastAPI 的 **`lifespan` 上下文管理器**实现（`app/main.py:16-33`）：

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.infra.database import check_database, close_database
    from app.infra.checkpointer import init_checkpointer, close_checkpointer
    try:
        await check_database()                        # ① 检查业务库连接
        checkpointer = await init_checkpointer()      # ② 初始化 Checkpointer
        app.state.agent = init_insurance_agent(checkpointer)  # ③ 初始化 Agent 并挂到 app.state
        yield                                          # 应用运行中……
    finally:
        await close_database()                        # 关闭时释放
        await close_checkpointer()
```

**启动顺序（`yield` 前 = 启动时，`yield` 后 = 关闭时）：**

| 步骤 | 动作 | 作用 |
|------|------|------|
| ① | `check_database()` | `SELECT 1` 验证业务库连通 |
| ② | `init_checkpointer()` | 打开连接池 + 建 checkpoints 表 |
| ③ | `init_insurance_agent(checkpointer)` | 组装 Agent，存入 `app.state.agent` |

**Agent 定义（`app/agents/insurance_advisor.py`）：**

```python
def init_insurance_agent(checkpointer: AsyncPostgresSaver):
    model = init_chat_model(
        model=settings.llm.chat_model,
        model_provider=settings.llm.provider,
        api_key=settings.llm.api_key,
    )
    agent = create_agent(
        model=model,
        tools=[],                       # 暂无工具
        system_prompt=SYSTEM_PROMPT,    # 保险顾问人设
        checkpointer=checkpointer,      # 关键：绑定检查点
    )
    return agent
```

> **`app.state` 是全局单例**：Agent 只在启动时创建一次，所有请求通过 `request.app.state.agent` 共享，避免每次请求重复初始化（模型初始化很贵）。

---

## 四、目标三：基于 SSE 实现前端流式对话

### 4.1 路由层（`app/modules/chat/router.py`）

```python
@router.post("", response_class=EventSourceResponse)
async def chat(
    request: ChatRequest,
    user_id: Annotated[int, Header(alias="x-user-id")],
    service: ChatService = Depends(get_chat_service),
) -> AsyncGenerator[ServerSentEvent, Any]:
    async for sse in service.chat_stream(user_id, request):
        yield sse
```

要点：
- `response_class=EventSourceResponse` 声明返回 SSE 流
- 返回类型是 `AsyncGenerator[ServerSentEvent, Any]`，用 `yield` 不断推送事件
- `user_id` 从请求头 `x-user-id` 读取（`Header(alias=...)`）

### 4.2 业务层（`app/modules/chat/service.py`）

```python
async def chat_stream(self, user_id, request):
    # 1. 校验会话归属
    chat_thread = await self.chat_thread_repository.find_owned(request.thread_id, user_id)
    if chat_thread is None:
        raise ChatThreadNotFoundError

    # 2. 组装输入 & 配置（thread_id 是关键）
    _input = {'messages': [HumanMessage(content=request.message)]}
    _config = RunnableConfig(configurable={'thread_id': str(request.thread_id)})

    # 3. v3 协议获取异步事件流
    stream = await self.agent.astream_events(_input, config=_config, version="v3")

    # 4. 只取文本增量，逐块推给前端
    async for message in stream.messages:
        async for text in message.text:
            yield ServerSentEvent(data=text, event="message")

    # 5. 结束标识
    yield ServerSentEvent(data="[DONE]", event="done")
```

### 4.3 SSE 工作原理

- 普通接口：一次性返回完整响应。
- SSE：响应保持连接，服务端用 `event:`（事件名）+ `data:`（数据）逐条推送。
- 前端（`EventSource`）监听 `message` 事件累加文本，收到 `done` 事件结束。
- 这里 `event="message"` 是文本增量，`event="done"` + `data="[DONE]"` 是结束信号（模仿 OpenAI 流式协议）。

> **`astream_events(version="v3")`**：返回 `stream.messages` 与 `stream.text`，逐 token 产出文本增量，是实现「打字机效果」的核心。

---

## 五、目标四：利用 Checkpointer 实现会话历史管理

这里要区分两个概念：

| 概念 | 存哪 | 由谁管 | 内容 |
|------|------|--------|------|
| **会话元数据** | `chat_threads` 表 | SQLAlchemy ORM | 会话 id、标题、用户、时间 |
| **会话消息（短期记忆）** | `checkpoints` 表 | Checkpointer | 完整的 `messages` 历史 |

### 5.1 会话元数据（`chat_threads` 表）

```python
class ChatThread(Base, CreateAtMixin, UpdateAtMixin):
    __tablename__ = "chat_threads"
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[int] = mapped_column(BigInteger)
    title: Mapped[str] = mapped_column(String(200), default="新会话", server_default="新会话")
```

- `id` 用 UUID，同时作为 LangGraph 的 `thread_id`（两套存储靠它关联）。
- 增删改查在 `ChatThreadRepository`（`repository.py`）中完成。

### 5.2 读取历史消息（`app/modules/chat_thread/service.py:get_history`）

```python
async def get_history(self, thread_id, user_id):
    chat_thread = await self.repository.find_owned(thread_id, user_id)
    if chat_thread is None:
        raise ChatThreadNotFoundError

    # 从 Checkpointer 拿会话快照
    _config = RunnableConfig(configurable={'thread_id': str(thread_id)})
    snapshot = await self.agent.aget_state(config=_config)

    # 从快照里取 messages，过滤出 user / assistant 的文本消息
    messages = snapshot.values.get('messages', [])
    for message in messages:
        if isinstance(message, HumanMessage):
            role = 'user'
        elif isinstance(message, AIMessage) and not message.tool_calls:
            role = 'assistant'
        else:
            continue
        if message.text:
            chat_message_list.append(ChatMessageResponse(role=role, content=message.text))
    return ChatHistoryResponse(thread_id=thread_id, messages=chat_message_list)
```

关键 API：
- `agent.aget_state(config=...)`：按 `thread_id` 读取最新快照（异步版）。
- `snapshot.values['messages']`：状态里的消息列表。
- 用 `isinstance` 区分 `HumanMessage`（用户）和 `AIMessage`（助手），并跳过带 `tool_calls` 的中间消息。

### 5.3 删除会话时同步清理检查点（`service.py:delete`）

```python
async def delete(self, thread_id, user_id):
    async with self.session.begin():
        thread = await self.repository.find_owned(thread_id, user_id)
        if thread is None:
            raise ChatThreadNotFoundError
        # 先删 Checkpointer 里的状态，再删元数据
        await self.agent.checkpointer.adelete_thread(str(thread_id))
        await self.repository.delete(thread)
```

> **必须两步删**：只删 `chat_threads` 会留下 checkpoints 里的「孤儿消息」，下次查询或复用该 `thread_id` 会出问题。

### 5.4 会话管理接口（`chat_thread/router.py`）

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/v1/chat-threads` | 创建会话 |
| GET | `/api/v1/chat-threads` | 查询用户会话列表 |
| PATCH | `/api/v1/chat-threads/{thread_id}` | 重命名 |
| DELETE | `/api/v1/chat-threads/{thread_id}` | 删除（含检查点） |
| GET | `/api/v1/chat-threads/{thread_id}/messages` | 查历史消息 |

---

## 六、完整调用链路（一次对话）

```
前端 POST /api/v1/chat  (header: x-user-id, body: {thread_id, message})
  → chat.router: 注入 ChatService + Session
  → service.chat_stream:
      ① find_owned(thread_id, user_id)  校验归属
      ② astream_events(输入, thread_id) 调用 Agent
      ③ Agent 内部：checkpointer 按 thread_id 读旧消息 → 拼上下文 → 模型推理 → 写回快照
      ④ 逐 token yield ServerSentEvent(message)
      ⑤ yield ServerSentEvent(done)
  → 前端累加 message 片段，收到 done 后结束
```

---

## 七、易错点与要点总结

1. **两套数据库驱动不要混**：SQLAlchemy 用 `+asyncpg`，Checkpointer(psycopg) 用裸 `postgresql`。
2. **`thread_id` 是纽带**：`chat_threads.id`（UUID）同时是 LangGraph 的 `thread_id`，全程以字符串形式传入 `RunnableConfig`。
3. **启动失败先查库**：`lifespan` 里三步（check_database → init_checkpointer → init_agent）任何一步连不上 PostgreSQL 都会导致 `Application startup failed`。
4. **删除会话要两步**：先 `adelete_thread()` 清检查点，再删 ORM 元数据。
5. **SSE 用异步生成器**：路由函数返回 `AsyncGenerator[ServerSentEvent, Any]`，内部 `async for` + `yield`。
6. **Agent 是单例**：挂在 `app.state`，避免重复初始化模型。
7. **`aget_state` 读历史时过滤消息**：跳过 `tool_calls` 的中间步骤，只保留纯文本的 user/assistant 消息。
