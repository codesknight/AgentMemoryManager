"""SQLite persistence for UserProfile objects."""
from __future__ import annotations

import logging
from typing import Optional

from agent_memory_manager.models.user_profile import UserProfile

try:
    import aiosqlite
except ImportError as exc:
    raise ImportError("Install 'aiosqlite': pip install aiosqlite") from exc

logger = logging.getLogger(__name__)

_CREATE = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id     TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


class UserProfileStore:
    """Async SQLite store for synthesized UserProfile objects.

    One row per user_id; the full profile is stored as a JSON blob.
    """

    def __init__(self, db_path: str = "user_profiles.db") -> None:
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_CREATE)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("UserProfileStore not initialized")
        return self._db

    async def save(self, profile: UserProfile) -> None:
        from datetime import datetime, timezone
        await self._conn().execute(
            """
            INSERT OR REPLACE INTO user_profiles (user_id, profile_json, updated_at)
            VALUES (?, ?, ?)
            """,
            (profile.user_id, profile.to_json(), datetime.now(timezone.utc).isoformat()),
        )
        await self._conn().commit()

    async def load(self, user_id: str) -> Optional[UserProfile]:
        async with self._conn().execute(
            "SELECT profile_json FROM user_profiles WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        try:
            return UserProfile.from_json(row["profile_json"])
        except Exception as exc:
            logger.warning("Failed to deserialize UserProfile for %s: %s", user_id, exc)
            return None

    async def delete(self, user_id: str) -> bool:
        cursor = await self._conn().execute(
            "DELETE FROM user_profiles WHERE user_id = ?", (user_id,)
        )
        await self._conn().commit()
        return cursor.rowcount > 0

    async def list_users(self) -> list[str]:
        async with self._conn().execute("SELECT user_id FROM user_profiles") as cur:
            return [row["user_id"] for row in await cur.fetchall()]
