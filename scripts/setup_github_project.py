"""
GitHub Project Setup Script
============================
Creates Milestones and Issues for AgentMemoryManager project management.

Usage:
    GITHUB_TOKEN=ghp_xxx python scripts/setup_github_project.py

Or:
    python scripts/setup_github_project.py --token ghp_xxx

Requires: requests (pip install requests)
"""

import argparse
import os
import sys

try:
    import requests
except ImportError:
    sys.exit("Install requests first: pip install requests")

REPO = "codesknight/AgentMemoryManager"
API = "https://api.github.com"


def make_headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def create_milestone(token: str, title: str, description: str, due: str) -> int:
    r = requests.post(
        f"{API}/repos/{REPO}/milestones",
        headers=make_headers(token),
        json={"title": title, "description": description, "due_on": due},
    )
    r.raise_for_status()
    mid = r.json()["number"]
    print(f"  ✓ Milestone '{title}' created (#{mid})")
    return mid


def create_issue(token: str, title: str, body: str, labels: list, milestone: int) -> int:
    r = requests.post(
        f"{API}/repos/{REPO}/issues",
        headers=make_headers(token),
        json={"title": title, "body": body, "labels": labels, "milestone": milestone},
    )
    r.raise_for_status()
    num = r.json()["number"]
    print(f"  ✓ Issue '{title}' created (#{num})")
    return num


def create_label(token: str, name: str, color: str, description: str = "") -> None:
    r = requests.post(
        f"{API}/repos/{REPO}/labels",
        headers=make_headers(token),
        json={"name": name, "color": color, "description": description},
    )
    if r.status_code == 422:
        print(f"  (label '{name}' already exists)")
    else:
        r.raise_for_status()
        print(f"  ✓ Label '{name}' created")


def main(token: str) -> None:
    print("\n=== Setting up GitHub project for AgentMemoryManager ===\n")

    # ── Labels ──────────────────────────────────────────────────────────
    print("Creating labels...")
    labels = [
        ("phase:1", "0075ca", "Phase 1 - Core scaffold"),
        ("phase:2", "e4e669", "Phase 2 - Core strategies"),
        ("phase:3", "d93f0b", "Phase 3 - Enhancement & eval"),
        ("component:backend", "c5def5", "Storage backend related"),
        ("component:strategy", "bfd4f2", "Memory strategy related"),
        ("component:llm", "f9d0c4", "LLM client related"),
        ("component:integration", "fef2c0", "Framework integration"),
    ]
    for name, color, desc in labels:
        create_label(token, name, color, desc)

    # ── Milestones ───────────────────────────────────────────────────────
    print("\nCreating milestones...")
    m1 = create_milestone(token, "v1.0 MVP",
        "Working memory + episodic memory + SQLite/Chroma backends + LangChain integration",
        "2026-06-30T00:00:00Z")
    m2 = create_milestone(token, "v1.5 Graph Memory",
        "Knowledge graph backend + Reflection + Zettelkasten + LlamaIndex integration",
        "2026-08-31T00:00:00Z")
    m3 = create_milestone(token, "v2.0 Advanced",
        "Temporal KG + Multi-modal + Multi-agent shared memory + Auto strategy selection",
        "2026-12-31T00:00:00Z")

    # ── Issues: v1.0 ────────────────────────────────────────────────────
    print("\nCreating v1.0 issues...")
    v1_issues = [
        (
            "[v1.0] Add ChromaDB backend",
            "Implement `ChromaBackend(MemoryBackend)` using `chromadb` client.\n\n"
            "- Persistent vector store\n- Support `search_by_vector` with metadata filters\n"
            "- Add to `backends/__init__.py` and `MemoryConfig.backend` enum\n\n"
            "Ref: `agent_memory_manager/backends/sqlite.py` as reference implementation.",
            ["enhancement", "component:backend", "phase:1"],
        ),
        (
            "[v1.0] Implement ReflectionStrategy",
            "Implement `ReflectionStrategy` (Generative Agents, Park et al. 2023).\n\n"
            "When accumulated importance score exceeds threshold, synthesize higher-level "
            "insights from recent memories.\n\n"
            "- Track importance accumulator per session\n"
            "- Use `REFLECTION_PROMPT` from `utils/prompts.py`\n"
            "- Store insights as `MemoryType.REFLECTION` records\n"
            "- Unit test with mock LLM",
            ["enhancement", "component:strategy", "phase:2"],
        ),
        (
            "[v1.0] Integration tests with real LLM (Anthropic)",
            "Add integration tests that call real Anthropic API.\n\n"
            "- Gated behind `ANTHROPIC_API_KEY` env var\n"
            "- Test `AtomicFactsStrategy` extracts facts correctly\n"
            "- Test `SummarizeStrategy` produces coherent summaries\n"
            "- Add to CI as optional job (`if: env.ANTHROPIC_API_KEY != ''`)",
            ["testing", "component:llm", "phase:2"],
        ),
        (
            "[v1.0] LOCOMO benchmark evaluation script",
            "Implement `tests/benchmarks/locomo_eval.py`.\n\n"
            "Download LOCOMO dataset and evaluate against:\n"
            "- Single-hop QA accuracy\n"
            "- Multi-hop QA accuracy\n"
            "- Retrieval latency P95\n"
            "- Token compression ratio\n\n"
            "Compare: SlidingWindow vs Summarize vs AtomicFacts strategies.",
            ["testing", "phase:3"],
        ),
        (
            "[v1.0] LlamaIndex memory module adapter",
            "Implement `agent_memory_manager/integrations/llamaindex.py`.\n\n"
            "Adapt `MemoryManager` as a LlamaIndex memory module compatible with "
            "`llama_index.core.memory.BaseMemory`.",
            ["enhancement", "component:integration", "phase:2"],
        ),
    ]
    for title, body, issue_labels in v1_issues:
        create_issue(token, title, body, issue_labels, m1)

    # ── Issues: v1.5 ────────────────────────────────────────────────────
    print("\nCreating v1.5 issues...")
    v15_issues = [
        (
            "[v1.5] ZettelkastenStrategy implementation",
            "Implement `ZettelkastenStrategy` (A-MEM, arXiv:2502.12110, NeurIPS 2025).\n\n"
            "Each interaction becomes a structured note. Similar notes are automatically linked.\n\n"
            "- Generate structured note: content + keywords + context description\n"
            "- Auto-link: search similar existing memories, set `links` field\n"
            "- Optionally update related memories' context descriptions",
            ["enhancement", "component:strategy"],
        ),
        (
            "[v1.5] NetworkX-based SemanticMemory (lightweight graph)",
            "Implement `SemanticMemory` using NetworkX (no Neo4j dependency).\n\n"
            "- Entity extraction via LLM\n"
            "- Store as directed graph (Entity nodes, Relation edges)\n"
            "- Support temporal relations (`valid_from`, `valid_to`)\n"
            "- Multi-hop query: `get_related_entities(entity_name, hops=2)`",
            ["enhancement", "component:backend"],
        ),
    ]
    for title, body, issue_labels in v15_issues:
        create_issue(token, title, body, issue_labels, m2)

    print("\n=== Done! Visit https://github.com/codesknight/AgentMemoryManager/issues ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    args = parser.parse_args()

    if not args.token:
        sys.exit(
            "GitHub token required.\n"
            "Set GITHUB_TOKEN env var or pass --token ghp_xxx\n"
            "Create a token at: https://github.com/settings/tokens (needs 'repo' scope)"
        )
    main(args.token)
