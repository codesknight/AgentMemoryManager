"""LLM-driven entity and relation extraction into SemanticMemory.

Parses conversations to populate the knowledge graph automatically.
Called by MemoryManager.add() when graph extraction is enabled.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agent_memory_manager.models.entity import Entity, Relation
from agent_memory_manager.utils.json_utils import extract_json
from agent_memory_manager.utils.prompts import ENTITY_EXTRACTION_PROMPT

if TYPE_CHECKING:
    from agent_memory_manager.llm.base import LLMClient
    from agent_memory_manager.memory.semantic_memory import SemanticMemory
    from agent_memory_manager.models import Message

logger = logging.getLogger(__name__)


class GraphExtractor:
    """Extracts entities and relations from conversation turns via LLM.

    Results are merged into the provided SemanticMemory instance.
    Existing entities are updated (not duplicated); new relations are added.
    """

    async def extract(
        self,
        messages: list[Message],
        session_id: str,
        graph: SemanticMemory,
        llm: LLMClient,
    ) -> tuple[int, int]:
        """Run extraction and merge results into `graph`.

        Returns:
            (entities_added_or_updated, relations_added)
        """
        conversation = "\n".join(
            f"{m.role.value.upper()}: {m.content}" for m in messages
        )
        raw = await self._call_llm(conversation, llm)
        if not raw:
            return 0, 0

        entities_raw: list[dict] = raw.get("entities", [])
        relations_raw: list[dict] = raw.get("relations", [])

        entity_count = 0
        for e_data in entities_raw:
            name = e_data.get("name", "").strip()
            if not name:
                continue
            existing = graph.get_entity(name)
            if existing:
                # Merge new attributes into existing entity
                existing.attributes.update(e_data.get("attributes", {}))
                graph.add_entity(existing)
            else:
                entity = Entity(
                    session_id=session_id,
                    name=name,
                    entity_type=e_data.get("type", "concept"),
                    attributes=e_data.get("attributes", {}),
                )
                graph.add_entity(entity)
            entity_count += 1

        relation_count = 0
        for r_data in relations_raw:
            subj = r_data.get("subject", "").strip()
            pred = r_data.get("predicate", "").strip()
            obj  = r_data.get("object", "").strip()
            if not (subj and pred and obj):
                continue
            # Skip if this exact relation already exists
            if self._relation_exists(graph, subj, pred, obj):
                continue
            relation = Relation(
                session_id=session_id,
                subject_id=subj,
                predicate=pred,
                object_id=obj,
                confidence=float(r_data.get("confidence", 0.9)),
            )
            graph.add_relation(relation)
            relation_count += 1

        logger.debug(
            "GraphExtractor: +%d entities, +%d relations for session=%s",
            entity_count, relation_count, session_id,
        )
        return entity_count, relation_count

    # ── private ──────────────────────────────────────────────────────────────

    async def _call_llm(self, conversation: str, llm: LLMClient) -> dict:
        prompt = ENTITY_EXTRACTION_PROMPT.format(conversation=conversation)
        try:
            response = await llm.generate(prompt, max_tokens=512, temperature=0.0)
            result = extract_json(response)
            if not isinstance(result, dict):
                return {}
            return result
        except Exception as exc:
            logger.warning("GraphExtractor LLM call failed: %s", exc)
            return {}

    @staticmethod
    def _relation_exists(graph: SemanticMemory, subj: str, pred: str, obj: str) -> bool:
        """Check if an equivalent current relation already exists."""
        for rel in graph.get_current_relations():
            if (rel.subject_id.lower() == subj.lower()
                    and rel.predicate == pred
                    and rel.object_id.lower() == obj.lower()):
                return True
        return False
