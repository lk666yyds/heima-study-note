# 保险工具定义与人工确认（HITL）学习笔记

> 项目：hm-insurance / agent-service
> 核心库：LangChain（`@tool` + `create_agent`）+ LangGraph（`interrupt` / `Command` / `GraphBubbleUp`）+ SQLAlchemy(async)
> 三大目标：
> 1. 保险查询工具：让 Agent 在推荐产品前先查数据库，基于真实数据回答
> 2. 生成并保存保险方案：Agent 选产品、分预算、写理由，方案落库
> 3. 保险方案保存人工确认（HITL）：关键动作前暂停，把决策权交给人类，确认后从暂停处继续

---

## 一、工具总览与 Agent 挂载

工具都定义在 `app/agents/tools.py`，最终在 `app/agents/insurance_advisor.py` 里挂到 Agent 上。

```python
# app/agents/insurance_advisor.py
tools = [query_candidate_products, save_insurance_plan]

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    context_schema=InsuranceAgentContext,     # 运行时上下文
    middleware=[handle_tool_errors],          # 工具错误兜底
)
```

### 1.1 运行时上下文（`app/agents/context.py`）

```python
@dataclass(frozen=True)
class InsuranceAgentContext:
    """保险顾问Agent运行时上下文"""
    user_id: int
```

- `frozen=True`：不可变，创建后不能改字段，保证上下文稳定。
- 通过 `context_schema=InsuranceAgentContext` 告诉 Agent「工具里可以用 `ToolRuntime[InsuranceAgentContext]` 拿到这个上下文」。
- 在 `chat/service.py:36` 创建：`InsuranceAgentContext(user_id=user_id)`，user_id 来自请求头 `x-user-id`。

### 1.2 工具错误兜底中间件（`insurance_advisor.py`）

```python
@wrap_tool_call
async def handle_tool_errors(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
    try:
        return await handler(request)  # 正常执行工具
    except GraphBubbleUp:
        raise  # 中断异常必须向外抛，不能吞
    except Exception as e:
        return ToolMessage(            # 普通错误转成模型能读的消息
            content=f'工具执行失败:{str(e)}',
            tool_call_id=request.tool_call['id']
        )
```

关键点：**`GraphBubbleUp`（工具中断）必须 `raise` 透传**，否则 HITL 的中断信号会被当成普通错误吃掉，前端永远收不到确认请求。

---

## 二、目标一：保险查询工具 `query_candidate_products`

作用：用户咨询具体产品或需要推荐时，Agent 先查数据库拿真实候选产品，再组织回答，避免「编造产品」。

### 2.1 返回类型模型 `ProductToolResult`

```python
class ProductToolResult(BaseModel):
    id: int = Field(..., description='产品唯一标识')
    name: str = Field(..., description='产品名称')
    clause_name: str = Field(..., description='保险条款名称')
    category: str = Field(..., description='保险分类')
    insurer: str = Field(..., description='承保保险公司')
    image_url: Optional[str] = Field(None, description='产品展示图片地址')
    description: Optional[str] = Field(None, description='保险产品简介')
    min_premium: Optional[Decimal] = Field(None, description='最低年缴保费参考')
    max_premium: Optional[Decimal] = Field(None, description='最高保费参考')
    target_group: Optional[str] = Field(None, description='适用人群说明')
    highlights: Optional[List[str]] = Field(None, description='产品亮点列表')
    model_config = {'from_attributes': True}   # 支持从 ORM 对象直接转换
```

> 标准做法：**给工具定义专门的 Pydantic 返回模型，详细描述每个字段含义**，让模型清楚知道拿到的数据是什么，回答更准。

### 2.2 工具本体

```python
@tool
async def query_candidate_products(
    categories: list[Literal['critical_illness','medical','accident','life']],
    premium_min: Decimal | None = None,
    limit_per_category: int = 5,
) -> List[ProductToolResult]:
    """查询候选推荐保险产品列表，在用户咨询保险产品或要求推荐时调用"""
    async with AsyncSessionFactory() as session:
        product_service = ProductService(session)
        products = await product_service.list_candidates(categories, premium_min, limit_per_category)
        return [ProductToolResult.model_validate(p) for p in products]
```

- `@tool` 装饰器把普通 async 函数变成 LangChain 工具，函数 docstring 就是给模型的「使用说明」。
- `Literal[...]` 约束分类枚举，模型只会传合法险种：重疾 `critical_illness`、百万医疗 `medical`、意外 `accident`、寿险 `life`。
- 结果通过 `model_validate` 把 ORM 对象转成 Pydantic（配合 `from_attributes`）。

### 2.3 底层查询逻辑（`product/service.py`）

```python
async def list_candidates(self, categories, premium_min, limit_per_category):
    candidate_products = []
    for category in dict.fromkeys(categories):       # dict.fromkeys 去重且保序
        products = await self.repository.find_limited_by_category(
            category=category, premium_min=premium_min, limit=limit_per_category)
        candidate_products.extend(products)
    return candidate_products
```

- `dict.fromkeys(categories)`：去掉重复分类，同时保持顺序。
- 每个分类独立取 `limit_per_category` 条，再汇总，保证多分类下每个险种都有候选。

---

## 三、目标二：生成并保存保险方案 `save_insurance_plan`

作用：Agent 根据用户画像和预算选产品、分配预算、生成推荐理由，把完整方案写库。

### 3.1 方案数据结构（`insurance_plan/schemas.py`）

```python
class InsurancePlanItemCreate(BaseModel):
    product_id: int                      # 推荐产品
    product_name: str                    # 产品名（用于展示）
    category: str                        # 险种类型（用于展示）
    priority: int = Field(default=1, ge=1)              # 展示顺序
    recommendation_reason: Optional[str] = None         # 推荐理由
    annual_premium_budget: Optional[Decimal] = Field(None, ge=0)  # 产品预算

class InsurancePlanCreate(BaseModel):
    plan_name: str = Field(..., min_length=1, max_length=120)
    summary: Optional[str] = None
    insured_profile: dict = Field(default_factory=dict)   # 被保险人画像
    items: list[InsurancePlanItemCreate] = Field(default_factory=list)
```

### 3.2 数据模型（`insurance_plan/models.py`）

两张表：方案主表 `insurance_plans` + 方案项表 `insurance_plan_items`（一对多，`plan_id` 外键 `ON DELETE CASCADE`）。

```python
class InsurancePlan(Base, CreateAtMixin, UpdateAtMixin):
    __tablename__ = "insurance_plans"
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"))
    plan_name: Mapped[str] = mapped_column(String(120))
    summary: Mapped[str | None] = mapped_column(Text)
    insured_profile: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"))
    annual_premium_budget: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    status: Mapped[Literal["uninsured","applying","insured"]] = mapped_column(String(30), default="uninsured")

class InsurancePlanItem(Base, CreateAtMixin, UpdateAtMixin):
    __tablename__ = "insurance_plan_items"
    __table_args__ = (UniqueConstraint("plan_id","product_id", name="uk_insurance_plan_items_plan_product"),)
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    plan_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("insurance_plans.id", ondelete="CASCADE"))
    product_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("products.id"))
    priority: Mapped[int] = mapped_column(Integer, default=1)
    recommendation_reason: Mapped[str | None] = mapped_column(Text)
    annual_premium_budget: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    status: Mapped[Literal["uninsured","insured"]] = mapped_column(String(30), default="uninsured")
```

- 金额字段用 `Numeric(14,2)` 映射成 `Decimal`，避免浮点误差。
- `id` 用 `uuid4` 客户端生成，`default=uuid4` 创建对象时就有值（不用等 flush）。
- `status` 用 `Literal` 收紧类型，DB 层再配 CHECK 约束兜底。
- `(plan_id, product_id)` 唯一约束：同一方案里同一产品不重复。

### 3.3 保存逻辑（repository + service）

```python
# repository.py
async def create(self, user_id, data):
    # 1. 累加各产品预算 → 组合年缴预算参考
    items_budget = [i.annual_premium_budget for i in data.items if i.annual_premium_budget is not None]
    annual_premium_budget = sum(items_budget, Decimal('0')) if items_budget else None
    # 2. 保存方案
    insurance_plan = InsurancePlan(user_id=user_id, plan_name=..., insured_profile=..., annual_premium_budget=...)
    self.session.add(insurance_plan)
    await self.session.flush()          # 发 INSERT 拿 plan.id，事务未提交
    # 3. 保存方案项（用 plan.id 做外键）
    for item in data.items:
        self.session.add(InsurancePlanItem(plan_id=insurance_plan.id, product_id=item.product_id, ...))
    return insurance_plan

# service.py
async def create_plan(self, user_id, data):
    async with self.session.begin():     # 事务：正常退出 commit，异常 rollback
        plan = await self.repository.create(user_id, data)
        return plan.id
```

要点：
- `sum(items_budget, Decimal('0'))`：起始值是 `Decimal('0')`，保证金额求和全程精确、类型一致。
- `flush()` 只发 SQL 不提交，`begin()` 上下文退出时才 commit。
- 方案与方案项在同一事务里，任一失败整体回滚。

### 3.4 工具本体（保存分支）

```python
@tool
async def save_insurance_plan(data: InsurancePlanCreate, runtime: ToolRuntime[InsuranceAgentContext]) -> dict[str, str]:
    # ... 先 interrupt 人工确认（见目标三）...
    async with AsyncSessionFactory() as session:
        plan_id = await InsurancePlanService(session).create_plan(
            user_id=runtime.context.user_id, data=data)
        return {'plan_id': str(plan_id), 'message': '保险方案保存成功'}
```

- `runtime: ToolRuntime[InsuranceAgentContext]`：拿到运行时上下文，取 `runtime.context.user_id` 作为方案归属用户。
- `data: InsurancePlanCreate`：模型按 schema 构造结构化方案数据（含 items）。

---

## 四、目标三：保险方案保存人工确认（HITL）

核心思想：**在 Agent 执行关键动作（保存方案）前，用 `interrupt()` 把流程暂停，把决策权交给人类；人类确认/拒绝后，用 `Command(resume=...)` 从暂停处继续。**

### 4.1 工具内发起中断（`tools.py`）

```python
from langgraph.types import interrupt

@tool
async def save_insurance_plan(data, runtime):
    decision = interrupt({
        "plan": data.model_dump(mode="json"),       # 给前端看的方案内容
        "confirm_info": "是否确认保存这份保险推荐方案？",
        "action": ["approve", "reject"],            # 允许的决策
    })
    # 流程在此暂停，等人类返回 decision 后才继续向下
    if decision.get("action") != "approve":
        return {"message": f"用户取消保存保险方案, 原因:{decision.get('reject_reason')}"}
    # ... 确认通过，继续保存 ...
```

- `interrupt(payload)`：LangGraph 的中断原语，执行到这一行会**抛出一个特殊的中断信号**并暂停；等外部用 `Command(resume=value)` 恢复后，`interrupt()` 把 resume 的值当作返回值，代码从下一行继续。
- payload 会原样传给前端，让前端渲染确认弹窗。

### 4.2 中断信号必须透传（中间件）

```python
except GraphBubbleUp:
    raise   # 中断异常一定要向外抛出，否则前端收不到中断
```

中断本质是内部抛 `GraphBubbleUp`；中间件只兜普通 `Exception`，遇到中断必须原样 `raise`。

### 4.3 服务端把中断推给前端（`chat/service.py`）

```python
stream = await self.agent.astream_events(_input, config=_config, version="v3", context=_context)

async for message in stream.messages:       # 正常文本流
    ...

interrupts = await stream.interrupts()       # 拿本次运行产生的中断
if interrupts:
    for interrupt in interrupts:
        yield ServerSentEvent(event='interrupt', data=interrupt.value)   # 推给前端
```

- `stream.interrupts()`：v3 事件流结束后，返回本次运行挂起的中断列表。
- 通过 SSE 的 `interrupt` 事件把 `interrupt.value`（即 payload）发给前端，前端据此弹确认框。

### 4.4 前端决策回传，从暂停处继续

```python
# chat/schemas.py
class ChatRequest(BaseModel):
    thread_id: UUID
    message: Optional[str] = None
    decision: Optional[dict[str, Any]] = None   # 用户决策（approve/reject）

# chat/service.py
if request.decision:
    _input = Command(resume=request.decision)   # 用决策恢复中断
else:
    _input = {'messages': [HumanMessage(content=request.message)]}
```

- 用户点「确认/拒绝」后，前端带着 `decision`（如 `{"action": "approve"}`）再次调 `/api/v1/chat`。
- `Command(resume=request.decision)`：让 LangGraph 从上次 `interrupt()` 处恢复，并把 `decision` 作为 `interrupt()` 的返回值，工具继续走「确认则保存 / 拒绝则取消」的分支。

### 4.5 刷新页面时恢复待处理中断（`chat_thread/service.py`）

```python
snapshot = await self.agent.aget_state(config=_config)
interrupts = tuple()
if hasattr(snapshot, 'interrupts') and snapshot.interrupts:
    interrupts = snapshot.interrupts
elif hasattr(snapshot, 'tasks') and snapshot.tasks:
    for task in snapshot.tasks:
        if hasattr(task, 'interrupts') and task.interrupts:
            interrupts = task.interrupts
            break
return ChatHistoryResponse(..., interrupt=interrupts[0].value if interrupts else None)
```

- 从检查点快照里读「当前挂起的中断」，随历史消息一起返回，让前端刷新后仍能恢复确认框。
- 兼容新旧结构：优先 `snapshot.interrupts`，否则从 `snapshot.tasks[].interrupts` 取。

---

## 五、HITL 完整链路

```
模型决定保存方案 → 调用 save_insurance_plan
    │  interrupt({plan, confirm_info, action})
    ▼
LangGraph 抛 GraphBubbleUp（中断）
    ▼
handle_tool_errors 捕获 → 重新 raise 透传
    ▼
chat/service.py: stream.interrupts() 拿到中断
    ▼
SSE event="interrupt" 推给前端 → 前端弹确认框
    ▼
用户点【确认 / 拒绝】→ 前端再次 POST /chat，body 带 decision
    ▼
chat/service.py: Command(resume=decision) 恢复
    ▼
interrupt() 返回 decision → 工具判断
    ├─ action != approve → 返回「取消保存」消息
    └─ action == approve → 继续 create_plan 落库 → 返回 plan_id
```

---

## 六、易错点与要点总结

1. **`interrupt()` 的中断异常必须透传**：中间件里 `except GraphBubbleUp: raise` 不能省，否则前端永远收不到确认请求。
2. **恢复要用 `Command(resume=decision)`**：不能用普通 `{messages: [...]}` 输入，否则会被当成新对话，而不是从暂停点继续。
3. **`id` 用 `uuid4` 客户端生成**：方案对象创建时 `plan.id` 就有值，无需等数据库回填；这也是能直接给方案项赋 `plan_id` 的前提。
4. **金额统一用 `Decimal`**：`sum(..., Decimal('0'))` 起始值也要是 Decimal，避免 int/float 混入导致类型不一致或精度误差。
5. **外键约束别忽视**：`user_id` 必须存在于 `users` 表（种子只有 id=1），`product_id` 必须存在于 `products` 表（id=1~12）；模型幻觉出不存在的 id 会触发外键错误导致整个事务回滚。
6. **方案与方案项同一事务**：`async with session.begin()` 里先 flush 拿 plan.id，再批量 add 方案项，退出时统一 commit；任一步失败全部回滚。
7. **`ToolRuntime[InsuranceAgentContext]` 拿用户身份**：保存方案时用 `runtime.context.user_id`，不要另传或硬编码 user_id。
8. **工具返回模型要写清楚字段描述**：`ProductToolResult` 每个字段都带 `description`，帮助模型准确理解并组织回答。
