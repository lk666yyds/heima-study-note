
# 今日总结：使用 LangChain 搭建一个简单的 Agent

> 日期：2026-11-22
> 目标：掌握用 LangChain 框架从零搭建一个 Agent 的完整流程，并理解工具定义、消息、结构化输出、记忆、消息摘要、MCP 集成等核心能力。

---

## 一、整体流程概览

搭建一个 Agent 的通用步骤：

```
1. 加载环境变量
2. 初始化模型
3. 定义工具（@tool / Pydantic / MCP）
4. 创建 Agent（绑定模型 + 工具 + 可选配置）
5. 调用 Agent（invoke / ainvoke / stream）
```

一个最简版 Agent 长这样：

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()

model = init_chat_model("deepseek-v4-flash")

agent = create_agent(model=model, tools=[])

response = agent.invoke({"messages": [HumanMessage("你是谁？")]})
print(response["messages"][-1].content)
```

---

## 二、核心步骤详解

### 1. 加载环境变量

```python
from dotenv import load_dotenv
load_dotenv()   # 加载 .env 文件里的内容到环境变量
```

### 2. 初始化模型 `init_chat_model`

有两种方式：

**（1）默认初始化** —— 靠模型名自动推断厂商、自动读环境变量：

```python
model = init_chat_model(
    model="deepseek-v4-flash",
    temperature=1.5,
    top_p=0.75,
    extra_body={"thinking": {"type": "disabled"}},  # 额外参数，合并进请求体
)
```

**（2）自定义初始化** —— 手动指定 `model_provider`、`base_url`、`api_key`：

```python
model = init_chat_model(
    model='qwen3.8-max',
    model_provider='openai',
    base_url=dashscope_base_url,
    api_key=dashscope_api_key,
    temperature=1.5,
    top_p=0.75,
    extra_body={"thinking": {"type": "disabled"}},
)
```

> 说明：框架不直接支持的参数（如 `thinking`）统一放 `extra_body` 里，调用时会合并进请求体一起发给模型。

### 3. 定义消息 Message

用消息对象代替纯字符串，可区分角色：

```python
from langchain.messages import HumanMessage, SystemMessage

response = model.invoke([
    SystemMessage(content="你扮演小团团，用她的身份回答问题"),
    HumanMessage(content="你是谁？"),
])
```

### 4. 定义工具 Tool

工具就是「加了 `@tool` 装饰器的普通函数」，有三种定义方式：

**（1）方式一：直接 @tool，靠 docstring 描述**（最简单）：

```python
@tool
def get_weather(location: str) -> str:
    """查询指定 location 的天气"""
    return f"Current weather in {location} is sunny"
```

**（2）方式二：靠函数类型注解 + docstring 自动推断参数**。

**（3）方式三：用 Pydantic 模型描述参数（重点，可加约束校验）**：

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    location: str = Field(..., description='城市名称', min_length=1, max_length=10)
    units: Literal['celsius', 'fahrenheit'] = Field(default='celsius', description='温度单位')
    include_forecast: bool = Field(default=False, description='是否包含未来天气预报')

@tool(args_schema=WeatherInput, description='获取指定位置当前时间的天气')
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    ...
```

> `args_schema` 用一个 Pydantic 类声明工具接收的参数、类型和约束，既能引导模型正确传参，也能在调用前做参数校验。

### 5. 创建 Agent

```python
agent = create_agent(
    model=model,       # 绑定模型
    tools=tools,       # 绑定工具列表
)
```

可选参数：

- `system_prompt`：系统提示词，如 `system_prompt="像海盗一样说话"`
- `checkpointer`：检查点保存器，用于记忆（见下文）
- `middleware`：中间件，如消息摘要（见下文）

### 6. 调用 Agent

| 方法 | 说明 |
|------|------|
| `invoke` | 同步阻塞调用 |
| `ainvoke` | 异步调用（`await`），MCP 工具是异步的，必须用 `ainvoke` |
| `stream` | 流式调用，逐片段返回 |
| `stream_events` | 流式事件，返回 token 级文本流 |

```python
# 同步
response = agent.invoke({"messages": [HumanMessage("你是谁？")]})

# 异步（MCP 工具必须用）
response = await agent.ainvoke({"messages": [HumanMessage("现在是什么时间")]})

# 流式事件
stream = agent.stream_events({"messages": [HumanMessage("你是谁？")]}, version="v3")
for message in stream.messages:
    for text in message.text:
        print(text, end="", flush=True)
```

> 返回结果是字典，`response["messages"]` 是消息列表，`[-1]` 取最后一条（Agent 的最终回复），`.content` 是纯文本。

---

## 三、进阶能力

### 1. 结构化输出 `with_structured_output`

让模型按指定结构输出，并自动解析成 Pydantic 对象：

```python
class CapitalInfo(BaseModel):
    name: str = Field(description="首都的名字")
    location: str = Field(description="首都位置")
    ...

model_with_structured_output = model.with_structured_output(schema=CapitalInfo)
response = model_with_structured_output.invoke([HumanMessage("帮我创造月球首都城市背景")])
# response 类型是 CapitalInfo，可直接 response.name
```

### 2. 短期记忆（checkpointer + thread_id）

用 `checkpointer` 保存对话状态，用 `thread_id` 区分会话：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

agent = create_agent(model="deepseek-v4-flash", tools=[], checkpointer=InMemorySaver())

config = RunnableConfig(configurable={"thread_id": "thread-1"})

agent.invoke({"messages": [HumanMessage("我是小团团，我喜欢海盗")]}, config=config)
agent.invoke({"messages": [HumanMessage("我是谁？我喜欢什么？")]}, config=config)  # 记得之前的内容
```

> 同一个 `thread_id` = 同一个会话，记忆在相同 `thread_id` 之间延续。

### 3. 消息摘要中间件 `SummarizationMiddleware`

对话过长时自动总结历史，控制上下文长度：

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="deepseek-v4-flash",
            trigger=("messages", 6),   # 消息数超过 6 时触发总结
            keep=("messages", 1),      # 保留最近 1 条消息
        )
    ],
)
```

> 注意：`middleware` 要传给 `create_agent`，不能传给 `init_chat_model`。

---

## 四、MCP 集成

### 1. 连接多个 MCP 服务器 `MultiServerMCPClient`

一次连接多个 MCP 服务，统一收集工具：

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "time-mcp": {                       # stdio：本地启动子进程
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "time-mcp"],
    },
    "12306-mcp": {                      # streamable_http：连接远程服务
        "transport": "streamable_http",
        "url": "https://mcp.api-inference.modelscope.net/xxxx/mcp",
    },
})

tools = await client.get_tools()   # 拉取所有工具（异步）
agent = create_agent(model=model, tools=tools)
```

### 2. 自定义 MCP 服务（用 FastMCP）

```python
# math_mcp_server.py
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="http", show_banner=True, host='0.0.0.0', port=8000)
```

客户端连接这个自定义服务：

```python
client = MultiServerMCPClient({
    "math": {
        "transport": "streamable_http",
        "url": "http://127.0.0.1:8000/mcp",
    }
})

async def run():
    tools = await client.get_tools()
    agent = create_agent(model=init_chat_model("deepseek-v4-flash"), tools=tools)
    response = await agent.ainvoke({"messages": [HumanMessage("429 和 327 的平方根的和是多少")]})

asyncio.run(run())
```

---

## 五、常见问题

### 1. Windows + Jupyter 下 MCP 报 fileno() 错误

在启动 MCP 子进程前，把 `sys.stderr` 换回原始 stderr：

```python
import sys
if sys.platform == "win32":
    if "ipykernel" in sys.modules:
        sys.stderr = sys.__stderr__
```

> 原因：ipykernel 的 `sys.stderr` 没有有效的 `fileno()`，启动子进程时会报错。这段代码必须放在「启动 MCP 子进程（`get_tools()`）」之前。

### 2. `invoke` vs `ainvoke`

- 普通 `@tool` 工具是同步的，用 `invoke`。
- MCP 工具（`get_tools()` 返回的）是异步的，必须用 `await agent.ainvoke(...)`。

### 3. `messages` 参数是列表

调用时 `{"messages": [...]}` 里的 `messages` 必须是**消息列表**（如 `[HumanMessage(...)]`），不能直接传字符串。

---

## 六、一句话记忆

> 搭 Agent = 加载环境变量 → `init_chat_model` 初始化模型 → `@tool` 或 MCP 定义工具 → `create_agent` 绑定 → `invoke`/`ainvoke` 调用。
> 记忆靠 `checkpointer + thread_id`；控制上下文靠 `SummarizationMiddleware`；扩展工具靠 MCP；规范输出靠 `with_structured_output`。
