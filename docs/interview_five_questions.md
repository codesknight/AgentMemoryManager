# 五道核心面试题深度解答

> 面向技术面试官，每道题都有"一句话答案 → 展开讲 → 代码证明 → 反问/延伸"四层结构

---

## Q1. 五类长期记忆的区别以及你是如何融合的

### 一句话定位

| 策略 | 一句话 | 需要 LLM | 论文来源 |
|------|--------|----------|----------|
| **SlidingWindow** | 保留最近 N 条原始对话 | ❌ | — |
| **Summarize** | 超限时把旧对话压缩成一段摘要 | ✅ 1次 | — |
| **AtomicFacts** | 提取不重复的原子事实并做增删改决策 | ✅ 2次/事实 | Mem0 arXiv:2504.19413 |
| **Reflection** | 积累到阈值后从已有记忆里合成高阶洞见 | ✅ 1次/触发 | Generative Agents UIST 2023 |
| **Zettelkasten** | 每条记忆是原子笔记，通过语义相似度建双向链接 | ✅ 2次 | A-MEM arXiv:2502.12110 |
| **StreamingCompress** | 每次 add() 都实时检查并压缩，始终保持低 token 状态 | ✅ 按需 | StreamingLLM ICLR 2024 |

---

### 展开讲：每类策略的核心差异

**① SlidingWindowStrategy — 零 LLM 成本，牺牲长程记忆**

```
存储：每条 Message → MemoryRecord，直接 backend.save()
检索：取最近 window_size=20 条，从新到旧填满 token_budget
```

本质是 FIFO 队列。优点是延迟接近 0ms，缺点是 20 轮之前的信息永久丢失，适合延迟极敏感、对话不超过 20 轮的场景。

---

**② SummarizeStrategy — 批量压缩，但有延迟峰值**

```python
# process() 流程
for msg in messages:
    await backend.save(record)                   # 先存

all_records = await backend.list_by_session()
if sum(r.token_estimate() for r in all_records) > summarize_threshold:
    # 超限时：取最旧的一批 → LLM 摘要 → 删旧记录 → 存摘要
    summary = await llm.generate(SUMMARIZE_PROMPT.format(...))
    summary_record = MemoryRecord(content=f"[Summary] {summary}", importance_score=7.0)
```

问题：超限那次 `add()` 调用会多花 1-3 秒等 LLM 摘要，其余次调用很快。有明显的延迟抖动。

---

**③ AtomicFactsStrategy — 两阶段 Extract+Update，核心策略** ⭐

这是项目最重要的策略，85%+ 的 token 压缩率来自这里。

```
Phase 1 — Extract（1次 LLM 调用）
  输入：本轮对话
  输出：[{"fact": "用户叫 Alice", "importance": 8}, ...]
  过滤：importance < 3.0 的事实直接丢弃

Phase 2 — Update（每条事实 1次 LLM 调用）
  对每条新 fact，把已有 memories 的前 30 条传给 LLM
  LLM 返回：{"action": "add/update/delete/skip", "target_id": "..."}
```

**四种操作语义（关键面试点）**：
- `add`：全新事实，不在已有记忆里 → 直接 `backend.save()`
- `update`：修正或扩充已有记忆 → `backend.update(target_id, {...})`，旧记录被覆盖而非新增
- `delete`：新事实使旧记忆失效（如"用户已离职"） → `backend.delete(target_id)`
- `skip`：已有记忆已经覆盖此事实 → 什么都不做

**为什么不直接追加，而是做去重判断**：避免同一个事实（"Alice 是工程师"）被重复存储十几次，导致检索时噪声极大。

---

**④ ReflectionStrategy — 两层记忆，高阶洞见**

```python
# 每次 process() 后累加 importance score
self._accumulators[session_id] += sum(r.importance_score for r in newly_added)

# 超过 150 分才触发反思（避免频繁 LLM 调用）
if self._accumulators[session_id] >= self.reflection_threshold:
    insights = await self._synthesize(recent_memories, llm)
    # 存为 MemoryType.REFLECTION，标记来源（evidence_indices）
    self._accumulators[session_id] = 0.0  # 重置
```

**两类记忆的区别**：
- `EPISODIC`：具体事件，如 "2026-05-20 用户提到在做 RAG 项目"
- `REFLECTION`：规律洞见，如 "用户持续关注 AI 工程领域，倾向开源方案"

Reflection 记录的 `importance_score` 通常在 7-9，检索时被优先返回，相当于"压缩后的长期记忆"。

---

**⑤ ZettelkastenStrategy — 知识网络，检索跟随链接**

```python
# 存储时：找相似笔记建双向链接
candidates = await backend.search_by_vector(embedding, top_k=20)
links = [r.id for r in candidates
         if cosine_similarity(r.embedding, embedding) >= 0.75][:5]

record = MemoryRecord(links=links, keywords=["rag", "techcorp"])
await backend.save(record)

# 建立反向链接
for linked_id in links:
    linked = await backend.get(linked_id)
    await backend.update(linked_id, {"links": linked.links + [record.id]})
```

**检索时的 link-hop**（区别于其他策略）：
```python
# 先找语义直接匹配的 top-10
top_notes = await backend.search_by_vector(query_emb, top_k=10)

# 再沿链接跳跃，找"相关但不直接匹配"的笔记
for note in frontier:
    for lid in note.links:          # 跟随链接
        linked = await backend.get(lid)
```

这能发现"上个月提到的项目背景"这类向量相似度低但语义关联强的记忆。

---

### 融合方式：StrategyPipeline

**关键代码**：

```python
class StrategyPipeline(MemoryStrategy):
    def __init__(self, strategies: list[MemoryStrategy]) -> None:
        self.strategies = strategies

    async def process(self, messages, session_id, backend, embedder, llm):
        merged = ProcessResult()
        for strategy in self.strategies:
            result = await strategy.process(...)   # 顺序执行
            merged.added.extend(result.added)
            merged.updated.extend(result.updated)
            merged.deleted.extend(result.deleted)
            merged.compressed = merged.compressed or result.compressed
            merged.reflected = merged.reflected or result.reflected
        return merged

    async def build_context(self, query, session_id, backend, embedder, token_budget):
        # 检索委托给最后一个策略（通常语义最丰富）
        return await self.strategies[-1].build_context(...)
```

**典型组合**（推荐）：

```python
# 组合 1：AtomicFacts + Reflection
# 先提取原子事实，定期触发高阶反思
pipeline = StrategyPipeline([
    AtomicFactsStrategy(min_importance=3.0),
    ReflectionStrategy(
        reflection_threshold=150.0,
        max_insights=5,
    )
])

# 组合 2：AtomicFacts 嵌入 Reflection（delegate 模式）
# ReflectionStrategy 内部调用 delegate，避免重复存储
strategy = ReflectionStrategy(
    delegate=AtomicFactsStrategy(),  # 先提取事实，再触发反思
    reflection_threshold=100.0,
)
```

**Pipeline vs delegate 模式的区别**：
- Pipeline：两个策略各自独立存储，记忆条数 = A + B 的总和
- Delegate（ReflectionStrategy 内置）：Reflection 直接消费 AtomicFacts 的结果，不重复存储，节省后端 I/O

**为什么 build_context 委托给最后一个策略**：Pipeline 里最后的策略通常是语义最丰富的（Reflection > AtomicFacts > SlidingWindow），由它来决定检索逻辑最合理。

---

### 面试可能追问

**"五种策略共用同一个 backend，不会互相干扰吗？"**

不会，因为 `MemoryType` 枚举区分了存储类型（EPISODIC / REFLECTION），`list_by_session` 可以按类型过滤。ReflectionStrategy 在合成洞见时会显式排除已有的 REFLECTION 记录，避免"对反思再反思"的无限循环。

**"用户的业务场景怎么选策略？"**

| 场景 | 推荐策略 |
|------|----------|
| 客服机器人，每次对话 < 20 轮 | SlidingWindow |
| 通用助手，对话不长 | SummarizeStrategy |
| 个人助理，需要记住用户偏好 | AtomicFacts + Reflection |
| 知识管理场景 | ZettelkastenStrategy |
| 实时流式对话，延迟敏感 | StreamingCompressStrategy |

---

## Q2. 四种存储后端的功能和区别，工作中如何调用的

### 一句话定位

| 后端 | 适用场景 | 向量检索 | 持久化 | 外部依赖 |
|------|----------|----------|--------|----------|
| **InMemory** | 测试、演示 | Python 全量扫描 | ❌ | 无 |
| **SQLite** | 单机生产、低并发 | Python 精排 | ✅ 文件 | aiosqlite |
| **Chroma** | 中等规模，开发友好 | HNSW ANN | ✅ 可选 | chromadb |
| **Qdrant** | 大规模生产 | HNSW + 过滤 ANN | ✅ 可选 | qdrant-client |
| **pgvector** (v2.0) | PostgreSQL 生产环境 | IVFFlat ANN + SQL 过滤 | ✅ PostgreSQL | asyncpg |

---

### 展开讲

**① InMemoryBackend — 纯 Python 字典**

```python
class InMemoryBackend(MemoryBackend):
    def __init__(self):
        self._store: dict[str, MemoryRecord] = {}

    async def save(self, record):
        self._store[record.id] = record
        return record.id

    async def search_by_vector(self, embedding, top_k, filters=None):
        candidates = list(self._store.values())
        # 过滤条件
        if filters:
            if sid := filters.get("session_id"):
                candidates = [r for r in candidates if r.session_id == sid]
            if uid := filters.get("user_id"):
                candidates = [r for r in candidates if r.user_id == uid]
        # 全量扫描：O(N) 余弦相似度计算
        scored = [(r, cosine_similarity(r.embedding, embedding)) for r in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored[:top_k]]
```

**限制**：进程退出数据消失；N > 10万条时全量扫描 O(N) 性能下降。

---

**② SQLiteBackend — 生产可用的轻量持久化**

```python
# embedding 序列化：list[float] → JSON 字符串存 TEXT 列
async def save(self, record):
    emb_str = json.dumps(record.embedding) if record.embedding else None
    await conn.execute(
        "INSERT OR REPLACE INTO memories VALUES (?, ?, ?, ...)",
        record.id, record.session_id, ..., emb_str, ...
    )
```

**向量检索：两阶段精排**（因为 SQLite 没有原生向量索引）
```python
async def search_by_vector(self, embedding, top_k, filters):
    # 第一阶段：SQL 过滤缩小范围（利用 idx_session 索引）
    rows = await conn.fetchall(
        "SELECT * FROM memories WHERE session_id = ? LIMIT ?",
        session_id, top_k * 3
    )
    # 第二阶段：Python 层余弦相似度精排
    records = [_row_to_record(r) for r in rows]
    scored = [(r, cosine_similarity(json.loads(r.embedding or "[]"), embedding))
              for r in records]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:top_k]]
```

**关键索引设计**：
```sql
CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_user    ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_type    ON memories(memory_type);
```

**为什么用 aiosqlite 而不是 sqlite3**：普通 `sqlite3` 模块的 I/O 是同步阻塞的，在异步事件循环中会阻塞其他协程。`aiosqlite` 把 SQLite 操作放到线程池执行，返回 awaitable，不阻塞事件循环。

---

**③ ChromaBackend — HNSW 向量索引，开发友好**

```python
# 三种部署模式
ChromaBackend(mode="ephemeral")                     # 纯内存，测试用
ChromaBackend(mode="persistent", path="./chroma_db") # 本地文件持久化
ChromaBackend(mode="remote", host="localhost", port=8000)  # 远程服务

# 检索：Chroma 原生 HNSW
results = collection.query(
    query_embeddings=[embedding],
    n_results=top_k * 3,          # 多取一些用于 Python 精排
    where={"session_id": session_id}  # metadata 过滤
)
```

**与 SQLite 的本质区别**：Chroma 内置 HNSW（Hierarchical Navigable Small World）索引，检索复杂度从 O(N) 降到 O(log N)，10 万条记忆下性能差距显著。

---

**④ QdrantBackend — 生产级分布式向量数据库**

```python
# Qdrant 支持在 ANN 的同时做结构化过滤（不需要两阶段）
from qdrant_client.models import Filter, FieldCondition, MatchValue

filter_ = Filter(must=[
    FieldCondition(key="session_id", match=MatchValue(value=session_id))
])
results = client.search(
    collection_name="memories",
    query_vector=embedding,
    query_filter=filter_,     # ANN + 过滤同时进行，性能优于 Chroma
    limit=top_k,
)
```

**与 Chroma 的区别**：
- Qdrant 支持**过滤条件下的 ANN**（filtered HNSW），不需要先过滤再搜索；Chroma 过滤是 post-filter，性能略差
- Qdrant 支持分布式部署（多节点）、持久化存储、Rust 实现的高性能
- Chroma 更适合本地开发，Qdrant 更适合生产

---

**⑤ PgVectorBackend — PostgreSQL + pgvector（v2.0）**

```python
# 表的 embedding 列使用原生 vector 类型
CREATE TABLE memories (
    embedding vector,   -- pgvector 专有类型，非 TEXT
    ...
);
# IVFFlat 索引：把向量空间分为 100 个 cluster，ANN 只搜附近的 cluster
CREATE INDEX idx_pg_embedding ON memories
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

# 检索：使用 <=> 余弦距离算子
SELECT * FROM memories
WHERE session_id = $2
ORDER BY embedding <=> $1::vector  -- 原生 ANN 操作符
LIMIT 30
```

**为什么还需要 Python 精排**：IVFFlat 是近似算法，可能漏掉精确最近邻（特别是 cluster 边界附近的点）。数据库先取 top_k * 3，Python 再用精确余弦相似度取 top_k，两阶段保证准确性。

---

### 工作中如何调用

**统一接口，后端透明**——这是最重要的设计点：

```python
# 开发阶段：InMemory，无需任何配置
manager = MemoryManager(
    backend=InMemoryBackend(),
    strategy=AtomicFactsStrategy(),
    ...
)

# 单机部署：SQLite，只换 backend 参数
manager = MemoryManager(
    backend=SQLiteBackend("./memory.db"),
    ...
)

# 生产部署：pgvector，其余代码完全不变
manager = MemoryManager(
    backend=PgVectorBackend(dsn="postgresql://...", vector_dim=384),
    ...
)

# MemoryManager 的所有上层代码（add/search/build_prompt）
# 完全不感知后端类型，这就是 MemoryBackend 抽象的价值
```

**`backend.search_by_vector(embedding, top_k, filters)` 的 filters 参数**：
```python
# session 内检索
filters = {"session_id": "s1"}

# 跨 session 检索（v2.0 跨会话功能）
filters = {"user_id": "alice"}  # 不设 session_id，跨所有会话

# 类型过滤
filters = {"memory_type": "reflection"}
```

---

## Q3. 为什么 + 怎么构建时序知识图谱，支持实体关系、多跳查询与时间语义管理

### 为什么要构建知识图谱？

向量检索解决了"找相似"的问题，但无法回答**结构化问题**：
- "Alice 在哪个公司工作？"（实体属性查询）
- "DataCo 的员工都有谁？"（反向关系查询）
- "Alice 上个月还在 DataCo，现在呢？"（时间语义问题）

这三类问题需要知识图谱，参考了 Zep/Graphiti (arXiv:2501.13956) 的时序图设计。

---

### 怎么构建：三层架构

```
GraphExtractor（LLM 提取）
    ↓  entities + relations
SemanticMemory（NetworkX 有向图，内存运行）
    ↓  JSON 序列化
GraphStore（SQLite 持久化）
```

---

### 第一层：GraphExtractor — LLM 驱动的信息提取

**Prompt 设计**（ENTITY_EXTRACTION_PROMPT）：

```
Extract named entities and their relationships from the following conversation.
Only extract entities that are clearly stated — do not infer.

Return JSON:
{
  "entities": [
    {"name": "Alice", "type": "person", "attributes": {"role": "ML engineer"}},
    {"name": "DataCo", "type": "organization", "attributes": {}}
  ],
  "relations": [
    {"subject": "Alice", "predicate": "works_at", "object": "DataCo", "confidence": 0.95}
  ]
}
```

**实体去重（merge 而非 duplicate）**：
```python
existing = graph.get_entity(name)
if existing:
    # 已有实体 → 属性合并（新属性覆盖旧属性）
    existing.attributes.update(e_data.get("attributes", {}))
    graph.add_entity(existing)
else:
    # 全新实体 → 创建节点
    entity = Entity(name=name, entity_type=..., attributes=...)
    graph.add_entity(entity)
```

**关系去重（完全匹配跳过）**：
```python
@staticmethod
def _relation_exists(graph, subj, pred, obj):
    for rel in graph.get_current_relations():
        if (rel.subject_id.lower() == subj.lower()
                and rel.predicate == pred
                and rel.object_id.lower() == obj.lower()):
            return True
    return False
```

---

### 第二层：SemanticMemory — NetworkX 有向图

**图结构**：
```python
self._graph: nx.DiGraph = nx.DiGraph()  # 有向图
self._entities: dict[str, Entity] = {}  # name.lower() → Entity
self._relations: list[Relation] = []    # 全部关系（含已失效的）
```

**节点 = 实体，边 = 关系**：
```python
def add_entity(self, entity):
    self._entities[entity.name.lower()] = entity
    self._graph.add_node(
        entity.name.lower(),
        id=entity.id, name=entity.name, entity_type=entity.entity_type,
        attributes=entity.attributes,
    )

def add_relation(self, relation):
    self._graph.add_edge(
        subj, obj,
        predicate=relation.predicate,
        confidence=relation.confidence,
        valid_from=relation.valid_from.isoformat(),
        valid_to=relation.valid_to.isoformat() if relation.valid_to else None,
    )
```

---

### 时间语义管理（核心难点）

**Relation 数据模型**：
```python
class Relation(BaseModel):
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0
    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: Optional[datetime] = None  # None 表示当前仍然有效

    @property
    def is_current(self) -> bool:
        return self.valid_to is None
```

**软删除（关系失效，而非物理删除）**：
```python
def invalidate_relation(self, relation_id: str) -> bool:
    for rel in self._relations:
        if rel.id == relation_id and rel.valid_to is None:
            rel.valid_to = datetime.now(timezone.utc)  # 打时间戳，而非 delete
            # 同步更新图中的边数据
            self._graph[subj][obj]["valid_to"] = now.isoformat()
            return True
```

**为什么软删除而不是物理删除**：
- 保留历史记录——"Alice 曾在 DataCo 工作"这个历史事实不应消失
- 支持时间回溯查询——可以查询某个时间点的图谱状态
- `get_current_relations()` 返回 `valid_to is None` 的关系，过滤出当前有效的

---

### 多跳查询实现（BFS 广度优先搜索）

```python
def get_neighbours(self, name, hops=1, current_only=True):
    start = name.lower()
    if start not in self._graph:
        return []

    results = []
    visited = {start}
    frontier = [(start, depth=0)]        # BFS 队列

    while frontier:
        node, depth = frontier.pop(0)
        if depth >= hops:
            continue
        for neighbour in self._graph.successors(node):   # 只走出边
            if neighbour in visited:
                continue
            edge_data = self._graph[node][neighbour]
            if current_only and edge_data.get("valid_to") is not None:
                continue                  # 跳过已失效的关系
            entity = self._entities.get(neighbour)
            results.append({
                "entity": entity,
                "relation": edge_data.get("predicate"),
                "confidence": edge_data.get("confidence"),
                "distance": depth + 1,
            })
            visited.add(neighbour)
            frontier.append((neighbour, depth + 1))

    return results
```

**调用示例**：
```python
graph = await manager.query_graph("Alice", session_id="s1", hops=2)
for n in graph.neighbours:
    print(f"Alice --{n['relation']}--> {n['entity'].name} (距离 {n['distance']})")
# 输出：
# Alice --works_at--> DataCo (距离 1)
# DataCo --located_in--> Beijing (距离 2)
```

---

### 第三层：GraphStore — SQLite 持久化

```python
# 整张图序列化为 JSON 存一行
await conn.execute(
    "INSERT OR REPLACE INTO graphs (session_id, graph_json, updated_at) VALUES (?,?,?)",
    session_id,
    json.dumps(graph.to_dict()),   # entities + relations 全序列化
    datetime.now(timezone.utc).isoformat()
)
```

**MemoryManager 中的懒加载策略**：
```python
async def _get_graph(self, session_id):
    if session_id not in self._graphs:
        # 冷启动：从 SQLite 恢复图，避免重复提取
        if self._graph_store:
            graph = await self._graph_store.load(session_id) or SemanticMemory(session_id)
        else:
            graph = SemanticMemory(session_id)
        self._graphs[session_id] = graph
    return self._graphs[session_id]  # 后续直接返回内存中的图
```

---

### 为什么选 NetworkX 而不是 Neo4j？

| | NetworkX | Neo4j |
|--|--|--|
| 部署 | 纯 Python，零外部服务 | 需要独立进程 |
| 数据量 | 单 session 通常 < 1000 节点 | 适合亿级节点 |
| 查询语言 | Python API | Cypher |
| 序列化 | JSON → SQLite | 需要 APOC 导出 |

当前场景（单 session 知识图谱）NetworkX 完全够用，而且没有网络开销。规模达到百万节点时才需要 Neo4j，已在 v3.0 Roadmap 中规划。

---

## Q4. 如何适配 LangChain / LlamaIndex Memory 接口

### 核心思路：零侵入适配

两个框架都有自己的 Memory 抽象接口：
- LangChain：`BaseMemory`（`langchain-core`）
- LlamaIndex：`BaseMemory`（`llama-index-core`）

实现两个 Adapter 类分别继承这两个接口，在方法内部调用 `MemoryManager` 的 API，Agent 不需要修改任何代码。

---

### LangChain 适配

**接口要求**（来自 langchain-core）：
- `memory_variables: list[str]` — 声明提供哪些变量给 Prompt
- `load_memory_variables(inputs) -> dict` — 读取记忆
- `save_context(inputs, outputs)` — 保存对话
- `clear()` — 清除记忆

**实现**：
```python
class AgentMemoryManagerAdapter(BaseMemory):
    manager: Any              # MemoryManager 实例
    session_id: str
    token_budget: int = 2000
    memory_key: str = "history"

    @property
    def memory_variables(self):
        return [self.memory_key]   # 告诉 LangChain：我提供 "history" 变量

    def load_memory_variables(self, inputs):
        query = inputs.get("input") or inputs.get("human_input", "")
        # LangChain 的接口是同步的，但 MemoryManager 是异步的
        loop = _get_event_loop()
        ctx = loop.run_until_complete(
            self.manager.build_context(query=query, session_id=self.session_id, ...)
        )
        return {self.memory_key: ctx.context}

    def save_context(self, inputs, outputs):
        user_content = inputs.get("input", "")
        ai_content = outputs.get("response", "")
        messages = [Message(role=Role.USER, content=user_content),
                    Message(role=Role.ASSISTANT, content=ai_content)]
        loop = _get_event_loop()
        loop.run_until_complete(
            self.manager.add(messages=messages, session_id=self.session_id)
        )

    def clear(self):
        loop.run_until_complete(self.manager.delete_session(self.session_id))
```

**使用方式（对 LangChain 代码零改动）**：
```python
memory = AgentMemoryManagerAdapter(manager=manager, session_id="user-123")
chain = ConversationChain(llm=llm, memory=memory)   # 直接替换内置 Memory
```

---

### LlamaIndex 适配

**接口要求**（来自 llama-index-core）：
- `get(input) -> list[ChatMessage]` — 获取记忆注入 messages
- `put(message: ChatMessage)` — 保存一条消息
- `get_all() -> list[ChatMessage]` — 获取全部历史
- `reset()` — 清除

**实现关键：记忆作为 System Message 注入**
```python
def get(self, input=None, **kwargs):
    query = input or ""
    loop = _get_loop()
    ctx = loop.run_until_complete(
        self._manager.build_context(query=query, session_id=self._session_id, ...)
    )
    result = list(self._buffer)    # 当前窗口消息
    if ctx.context.strip():
        # 把记忆上下文作为 System Message 插入到消息列表最前面
        result.insert(0, ChatMessage(
            role=MessageRole.SYSTEM,
            content=f"Relevant memory from past conversations:\n{ctx.context}"
        ))
    return result

def put(self, message: ChatMessage):
    self._buffer.append(message)
    internal = _to_internal(message)   # ChatMessage → Message
    loop = _get_loop()
    loop.run_until_complete(
        self._manager.add(messages=[internal], session_id=self._session_id)
    )
```

---

### 同步/异步适配的技术细节

**问题**：LangChain 和 LlamaIndex 的 Memory 接口是同步的（`def load_memory_variables`），但 `MemoryManager` 全部是 `async def`。

**解决方案**：使用 `asyncio.get_event_loop().run_until_complete()`：

```python
def _get_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        # 在没有事件循环的线程中（如测试），新建一个
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
```

**潜在问题**：如果 LangChain 本身在异步上下文中调用 `load_memory_variables`，`run_until_complete` 会因为"在已运行的事件循环里再次 run"而报错。这是当前版本的已知限制，完整异步版本（支持 `async def load_memory_variables`）在 LangChain v0.3+ 的新接口中实现，已在 Roadmap 中规划。

---

### 与直接使用 MemoryManager 的区别

| | 直接使用 SDK | LangChain Adapter |
|--|--|--|
| 代码修改 | 需要改 Agent 代码 | 零侵入 |
| 异步支持 | 完整 async | run_until_complete 桥接 |
| 功能完整性 | 全部功能 | Memory 接口子集 |
| 适用场景 | 自研 Agent | 现有 LangChain 项目迁移 |

---

## Q5. 编写 174 个单元测试，核心代码覆盖率 86%，数字怎么来的？

### 测试数量：174 个，怎么数的

```bash
$ python -m pytest tests/unit/ --co -q
174 tests collected
174 passed, 2 skipped in 3.37s
```

**2 个跳过**：集成测试中需要真实 Chroma/Qdrant 服务的测试，用 `pytest.mark.skip` 标记，不计入核心覆盖。

---

### 覆盖率：86%，怎么测出来的

```bash
$ python -m pytest tests/unit/ --cov=agent_memory_manager --cov-report=term
TOTAL    1613 statements    226 missed    86%
```

**命令解释**：
- `--cov=agent_memory_manager`：只统计主包的覆盖率，不含测试代码
- `1613 total statements`：主包里可执行的代码行总数
- `226 missed`：没有被任何测试覆盖到的行
- `86% = (1613 - 226) / 1613`

---

### 哪些地方没覆盖（14%）？

```
# 主要未覆盖区域：
agent_memory_manager/server/entrypoint.py      0%   # Docker 启动脚本，需要真实环境
agent_memory_manager/integrations/__init__.py  0%   # 懒加载的可选依赖
agent_memory_manager/backends/sqlite.py       61%   # 部分错误分支（数据库异常）
agent_memory_manager/strategies/atomic_facts.py 78% # LLM 返回异常格式的分支
```

**entrypoint.py 为 0% 的原因**：它是读取环境变量、创建真实 backend/llm 的入口脚本，需要真实 OpenAI API Key 和文件系统，单元测试环境不具备。这是合理的豁免。

---

### 测试策略：Mock 一切外部依赖

**核心原则**：单元测试不能依赖真实的 LLM、数据库、网络。用 `unittest.mock.AsyncMock` 替代所有 I/O。

**策略测试模板**：
```python
@pytest.mark.asyncio
async def test_atomic_facts_adds_new_fact():
    # Arrange：Mock 所有依赖
    backend = AsyncMock()
    backend.list_by_session = AsyncMock(return_value=[])  # 已有记忆为空
    backend.save = AsyncMock()

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

    llm = AsyncMock()
    # 控制 LLM 的精确输出，测试特定分支
    llm.generate = AsyncMock(side_effect=[
        '[{"fact": "用户叫 Alice", "importance": 8}]',  # Phase 1: 提取
        '{"action": "add", "target_id": null}',         # Phase 2: 决策
    ])

    strategy = AtomicFactsStrategy()

    # Act
    result = await strategy.process(
        messages=[Message(role=Role.USER, content="我叫 Alice")],
        session_id="s1",
        backend=backend, embedder=embedder, llm=llm,
    )

    # Assert
    assert len(result.added) == 1
    assert result.added[0].content == "用户叫 Alice"
    backend.save.assert_called_once()
```

---

### pgvector 的特殊 Mock 方案

`pgvector.py` 在模块加载时就 `import asyncpg`，没装包就报 `ImportError`。解决方案是在测试文件顶部注入假的 asyncpg：

```python
import sys
from unittest.mock import MagicMock

# 在 pgvector.py 被导入之前，注入假的 asyncpg
_fake_asyncpg = MagicMock()
_fake_asyncpg.Pool = MagicMock
sys.modules.setdefault("asyncpg", _fake_asyncpg)

# 现在可以安全导入了
from agent_memory_manager.backends.pgvector import PgVectorBackend
```

然后用 mock pool 替换真实连接池：
```python
@pytest.fixture
def backend_with_pool():
    pool, conn = _make_pool_and_conn()  # 手工构造 mock pool
    backend = PgVectorBackend(dsn="postgresql://fake/db")
    backend._pool = pool    # 直接注入，跳过 initialize()
    return backend, pool, conn
```

---

### FastAPI 服务器测试（ASGITransport）

不启动真实 HTTP 服务器，用 `httpx.AsyncClient` + `ASGITransport`：

```python
from httpx import AsyncClient, ASGITransport

@pytest.fixture
async def client():
    mock_manager = AsyncMock()
    mock_manager.add = AsyncMock(return_value=AddResult(added=[...]))
    app = create_app(mock_manager)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as c:
        yield c

async def test_add_memories(client):
    resp = await client.post("/sessions/s1/memories", json={
        "messages": [{"role": "user", "content": "hello"}]
    })
    assert resp.status_code == 200
```

**好处**：测试完整的 HTTP 路由、JSON 序列化、状态码，但不需要真实网络端口。

---

### 覆盖率 86% 是好是坏？

**行业参考**：
- 一般项目：60-70%
- 良好工程实践：80%+
- 本项目：86%（核心业务逻辑代码）

**主动说明低覆盖区域是合理豁免**（面试时展示思考深度）：
1. `entrypoint.py`（0%）：需要真实环境变量和外部服务，单元测试不适合
2. `integrations/__init__.py`（0%）：懒加载的可选依赖，需要真实 LangChain/LlamaIndex
3. `sqlite.py` 的部分异常分支（61%）：需要模拟数据库崩溃等极端情况

**真正关键的代码覆盖率**：
```
strategies/*          ≈ 90%    # 核心策略，每个都有专项测试
backends/in_memory    95%      # 最常用的测试 backend
memory/graph_*        90-98%   # 知识图谱
models/*              95-100%  # 数据模型
```

---

### 测试文件分布（体现分层）

```
tests/unit/
├── test_atomic_facts.py   # AtomicFacts 两阶段 pipeline
├── test_reflection.py     # 累加器触发机制
├── test_zettelkasten.py   # 双向链接构建
├── test_streaming.py      # 流式压缩
├── test_summarize.py      # 摘要策略
├── test_strategies.py     # SlidingWindow + Pipeline
├── test_graph_extractor.py # LLM 实体提取 + 去重
├── test_graph_store.py    # SQLite 序列化
├── test_manager.py        # MemoryManager 集成（全 mock）
├── test_manager_user.py   # 跨会话 API
├── test_user_profile.py   # UserProfile 序列化
├── test_server.py         # FastAPI 路由（ASGITransport）
├── test_pgvector.py       # pgvector backend（asyncpg mock）
├── test_token_counter.py  # token 计数工具
└── test_scoring.py        # 余弦相似度 + 综合评分
```
