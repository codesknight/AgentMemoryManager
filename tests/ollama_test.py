import asyncio
from agent_memory_manager import MemoryManager, Message, Role
from agent_memory_manager.backends import InMemoryBackend
from agent_memory_manager.llm.openai import OpenAIClient
from agent_memory_manager.embedders.ollama_embedder import OllamaEmbedder
from agent_memory_manager.strategies import SlidingWindowStrategy, AtomicFactsStrategy

async def main():
    # 第一步：用 SlidingWindow（不调用 LLM）验证 embedder 和基础流程
    manager = MemoryManager(
        backend=InMemoryBackend(),
        strategy=AtomicFactsStrategy(),
        llm=OpenAIClient(
            model="qwen3:0.6b",
            base_url="http://localhost:11434/v1",
        ),
        embedder=OllamaEmbedder(model="nomic-embed-text"),
    )
    await manager.initialize()

    msgs = [
        Message(role=Role.USER, content="我叫李明，是一名后端工程师，最近在做 RAG 项目。"),
        Message(role=Role.ASSISTANT, content="你好李明！RAG 项目是做知识库检索吗？"),
    ]
    result = await manager.add(messages=msgs, session_id="test-001")
    print(f"[SlidingWindow] 存入 {len(result.added)} 条记忆")

    prompt = await manager.build_prompt("这个用户在做什么项目？", "test-001")
    print(prompt)

    # 第二步：单独测 LLM 是否可达
    print("\n[LLM 直连测试]")
    llm = OpenAIClient(model="qwen3:0.6b", base_url="http://localhost:11434/v1")
    reply = await llm.generate("请用一句话介绍自己")
    print(f"LLM 回复: {reply[:200]}")

asyncio.run(main())