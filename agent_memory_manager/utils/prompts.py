"""Built-in prompt templates for memory operations."""

IMPORTANCE_SCORING_PROMPT = """\
Rate the importance of the following memory on a scale of 1–10.

Scoring guide:
- 1–3: Casual chitchat, no lasting value (e.g. "okay", "thanks")
- 4–6: Moderately useful information (preferences, temporary plans)
- 7–9: Important personal info, commitments, decisions (name, job, key agreements)
- 10: Critical event, high lasting value (emergencies, core goals)

Memory: {memory_content}

Return only the integer (1–10), no explanation.
"""

ATOMIC_FACTS_EXTRACTION_PROMPT = """\
Extract atomic facts worth remembering long-term from the following conversation.

Rules:
1. Each fact must be a concise, self-contained statement (≤ 20 words)
2. Only extract information with lasting value: preferences, background, decisions, commitments, to-dos
3. Ignore transient content and small talk
4. Describe the user in third person ("The user...")
5. If there is nothing worth remembering, return an empty array

Conversation:
{conversation}

Respond in JSON:
[
  {{"fact": "The user is a Python backend engineer working on an AI assistant project.", "importance": 9}},
  {{"fact": "The user prefers open-source solutions over proprietary ones.", "importance": 7}}
]
"""

SUMMARIZE_PROMPT = """\
Summarize the following conversation into a concise paragraph (≤ 150 words).
Preserve all key information: names, decisions, action items, and important context.
Write in third person.

Conversation:
{conversation}

Summary:
"""

REFLECTION_PROMPT = """\
Based on the following recent memories, synthesize up to {max_insights} high-level insights or patterns.
Each insight should be more general and durable than a single memory.

Recent memories:
{recent_memories}

Respond in JSON:
[
  {{
    "insight": "The user consistently favors open-source tools for technical decisions.",
    "evidence_indices": [0, 2, 4],
    "importance": 8
  }}
]
"""

MEMORY_INJECTION_TEMPLATE = """\
## Relevant Memory (from conversation history)

{memory_context}

---

{base_prompt}"""

MEMORY_CONTEXT_ITEM_TEMPLATE = "- {content}"

ZETTELKASTEN_NOTE_PROMPT = """\
Create a structured Zettelkasten note from the following conversation.

Rules:
1. "content": one concise atomic statement capturing the key insight (≤ 25 words)
2. "keywords": 2–5 lowercase tags for indexing (topics, entities, concepts)
3. "context": one sentence explaining WHY this is noteworthy

Conversation:
{conversation}

Respond in JSON:
{{
  "content": "The user is building a RAG pipeline for internal documentation at TechCorp.",
  "keywords": ["rag", "documentation", "techcorp", "project"],
  "context": "User revealed a concrete ongoing project with specific technical scope."
}}
"""

ENTITY_EXTRACTION_PROMPT = """\
Extract named entities and their relationships from the following conversation.
Only extract entities that are clearly stated — do not infer.

Conversation:
{conversation}

Respond in JSON:
{{
  "entities": [
    {{"name": "Alex", "type": "person", "attributes": {{"role": "data scientist"}}}},
    {{"name": "TechCorp", "type": "organization", "attributes": {{}}}}
  ],
  "relations": [
    {{"subject": "Alex", "predicate": "works_at", "object": "TechCorp", "confidence": 0.95}}
  ]
}}
"""

DEDUP_CHECK_PROMPT = """\
Given a new fact and a list of existing memories, determine the correct action.

New fact: {new_fact}

Existing memories:
{existing_memories}

Choose ONE action:
- "add"    — the new fact is genuinely new information
- "update" — the new fact updates/corrects an existing memory (provide the memory ID)
- "delete" — the new fact makes an existing memory obsolete (provide the memory ID)
- "skip"   — the new fact is already covered by an existing memory

Respond in JSON:
{{"action": "add"|"update"|"delete"|"skip", "target_id": "<memory_id or null>"}}
"""
