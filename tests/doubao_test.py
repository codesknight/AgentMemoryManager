"""Integration smoke test for v1.5 Graph Memory with Doubao (豆包) API.

Usage:
    $env:DOUBAO_API_KEY="your-key-here"
    python tests/doubao_test.py

API key is read from environment variable DOUBAO_API_KEY.
Do NOT hardcode the key — never commit secrets to git.
"""

import asyncio
import os
import tempfile
from pathlib import Path

from agent_memory_manager import MemoryManager, Message, Role
from agent_memory_manager.backends import InMemoryBackend
from agent_memory_manager.embedders.ollama_embedder import OllamaEmbedder
from agent_memory_manager.llm.openai import OpenAIClient
from agent_memory_manager.strategies import AtomicFactsStrategy

# ── 豆包 API 配置 ─────────────────────────────────────────────────────────────

DOUBAO_API_KEY  = os.environ.get("DOUBAO_API_KEY", "")
DOUBAO_BASE_URL = "https://ark.volces.com/api/v3"
DOUBAO_MODEL    = "ep-m-20260129120145-kv6dc"   # doubao-seed-1-6-251015 推理接入点

# Embedder 仍用本地 nomic-embed-text（向量化无需大模型）
EMBED_MODEL  = "nomic-embed-text"
OLLAMA_BASE  = "http://localhost:11434"

# ─────────────────────────────────────────────────────────────────────────────

def make_llm():
    if not DOUBAO_API_KEY:
        raise RuntimeError("请先设置环境变量 DOUBAO_API_KEY")
    return OpenAIClient(
        model=DOUBAO_MODEL,
        api_key=DOUBAO_API_KEY,
        base_url=DOUBAO_BASE_URL,
        timeout=60.0,
        trust_env=True,   # 外部 API 需走系统代理（与本地 Ollama 相反）
    )

def make_embedder():
    return OllamaEmbedder(model=EMBED_MODEL, base_url=OLLAMA_BASE)


# ── 测试函数 ──────────────────────────────────────────────────────────────────

async def test_llm_connectivity():
    print("\n[1] 豆包 LLM 连通性测试")
    llm = make_llm()
    reply = await llm.generate("请用一句话介绍你自己", max_tokens=100)
    print(f"    LLM 回复: {reply[:300]}")
    assert len(reply) > 0
    print("    OK")


async def test_embed_connectivity():
    print("\n[2] Embedder 连通性测试（本地 nomic-embed-text）")
    emb = make_embedder()
    vec = await emb.embed("测试文本")
    print(f"    向量维度: {len(vec)}")
    assert len(vec) > 0
    print("    OK")


async def test_atomic_facts_pipeline():
    print("\n[3] AtomicFacts 策略 + 记忆检索")
    manager = MemoryManager(
        backend=InMemoryBackend(),
        strategy=AtomicFactsStrategy(),
        llm=make_llm(),
        embedder=make_embedder(),
        enable_graph=False,
    )
    await manager.initialize()

    msgs = [
        Message(role=Role.USER, content="我叫李明，是一名后端工程师，最近在做 RAG 项目。"),
        Message(role=Role.ASSISTANT, content="你好李明！RAG 项目是做知识库检索吗？"),
    ]
    result = await manager.add(messages=msgs, session_id="s-facts")
    print(f"    新增记忆: {len(result.added)} 条")
    for r in result.added:
        print(f"      - {r.content}")

    prompt = await manager.build_prompt("这个用户在做什么项目？", "s-facts")
    print(f"    增强 Prompt:\n{prompt}")
    await manager.close()
    print("    OK")


async def test_graph_extraction():
    print("\n[4] Graph Memory 提取（v1.5）")
    from agent_memory_manager.utils.prompts import ENTITY_EXTRACTION_PROMPT
    from agent_memory_manager.utils.json_utils import extract_json

    # 单独验证 LLM 对实体提取提示词的响应质量
    llm = make_llm()
    test_conv = "USER: 我叫王芳，在字节跳动担任算法工程师，负责推荐系统项目。\nASSISTANT: 您好王芳！推荐系统是大规模机器学习的典型场景。"
    raw = await llm.generate(
        ENTITY_EXTRACTION_PROMPT.format(conversation=test_conv),
        max_tokens=512,
        temperature=0.0,
    )
    print(f"    LLM 原始输出: {raw[:500]!r}")
    parsed = extract_json(raw)
    if isinstance(parsed, dict):
        print(f"    实体: {[e.get('name') for e in parsed.get('entities', [])]}")
        print(f"    关系: {[(r.get('subject'), r.get('predicate'), r.get('object')) for r in parsed.get('relations', [])]}")

    # 完整 manager 流程
    manager = MemoryManager(
        backend=InMemoryBackend(),
        strategy=AtomicFactsStrategy(),
        llm=make_llm(),
        embedder=make_embedder(),
        enable_graph=True,
    )
    await manager.initialize()

    msgs = [
        Message(role=Role.USER, content="我叫王芳，在字节跳动担任算法工程师，负责推荐系统项目。"),
        Message(role=Role.ASSISTANT, content="您好王芳！推荐系统是大规模机器学习的典型场景。"),
    ]
    result = await manager.add(messages=msgs, session_id="s-graph")
    print(f"    提取实体: {result.entities_extracted} 个 / 关系: {result.relations_extracted} 条")

    stats = await manager.get_stats("s-graph")
    print(f"    图谱统计: {stats.graph_entity_count} 实体 / {stats.graph_relation_count} 关系")
    await manager.close()
    print("    OK")


async def test_graph_query_api():
    print("\n[5] Graph 查询 API（v1.5）")
    manager = MemoryManager(
        backend=InMemoryBackend(),
        strategy=AtomicFactsStrategy(),
        llm=make_llm(),
        embedder=make_embedder(),
        enable_graph=True,
    )
    await manager.initialize()

    for content in [
        "我叫张伟，是清华大学的 NLP 研究员，研究大语言模型对齐问题。",
        "我们团队在开发基于 RLHF 的对齐框架，和 Anthropic 的方向类似。",
    ]:
        await manager.add(
            messages=[Message(role=Role.USER, content=content)],
            session_id="s-query",
        )

    result = await manager.query_graph("张伟", session_id="s-query", hops=1)
    print(f"    '张伟' 的邻居 ({len(result.neighbours)} 个):")
    for n in result.neighbours:
        target = n["entity"].name if n["entity"] else "?"
        print(f"      → {n['relation']} → {target} (conf={n['confidence']:.2f})")

    entity = await manager.get_entity("张伟", session_id="s-query")
    if entity:
        print(f"    实体详情: {entity.name} [{entity.entity_type}] attrs={entity.attributes}")
    else:
        print("    '张伟' 未提取到实体")

    persons = await manager.list_entities("s-query", entity_type="person")
    print(f"    Person 实体列表: {[e.name for e in persons]}")
    await manager.close()
    print("    OK")


async def test_graph_persistence():
    print("\n[6] GraphStore SQLite 持久化（v1.5）")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        m1 = MemoryManager(
            backend=InMemoryBackend(),
            strategy=AtomicFactsStrategy(),
            llm=make_llm(),
            embedder=make_embedder(),
            enable_graph=True,
            graph_db_path=db_path,
        )
        await m1.initialize()
        result = await m1.add(
            messages=[Message(role=Role.USER, content="我叫陈静，是北京大学的计算机博士生，研究知识图谱。")],
            session_id="s-persist",
        )
        print(f"    写入图谱：{result.entities_extracted} 实体")
        await m1.close()

        m2 = MemoryManager(
            backend=InMemoryBackend(),
            strategy=AtomicFactsStrategy(),
            llm=make_llm(),
            embedder=make_embedder(),
            enable_graph=True,
            graph_db_path=db_path,
        )
        await m2.initialize()
        entity = await m2.get_entity("陈静", session_id="s-persist")
        if entity:
            print(f"    SQLite 恢复成功：'{entity.name}' [{entity.entity_type}]")
        else:
            print("    实体未从 SQLite 恢复（图谱提取为空时属正常）")
        await m2.close()
    finally:
        Path(db_path).unlink(missing_ok=True)
    print("    OK")


# ── 主入口 ───────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print(" AgentMemoryManager v1.5 — 豆包 API 集成测试")
    print(f" LLM: {DOUBAO_MODEL}  |  Embedder: {EMBED_MODEL}(local)")
    print("=" * 60)

    tests = [
        test_llm_connectivity,
        test_embed_connectivity,
        test_atomic_facts_pipeline,
        test_graph_extraction,
        test_graph_query_api,
        test_graph_persistence,
    ]

    passed = failed = 0
    for t in tests:
        try:
            await t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"    FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f" 结果：{passed} 通过 / {failed} 失败")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
