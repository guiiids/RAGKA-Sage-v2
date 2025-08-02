"""Session memory abstractions with Redis and Postgres implementations."""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Any, Dict

from db_manager import DatabaseManager

logger = logging.getLogger()


class SessionMemory:
    """Interface for session memory backends."""

    def store_turn(self, session_id: str, user_msg: str, bot_msg: str, summary: Optional[str] = None) -> None:
        raise NotImplementedError

    def get_history(self, session_id: str, last_n_turns: int = 10) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def clear(self, session_id: str) -> None:
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        return {}


class PostgresSessionMemory(SessionMemory):
    """PostgreSQL-backed implementation keeping last N turns per session."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._ensure_table()
        logger.debug("PostgresSessionMemory initialized with max_turns=%s", max_turns)

    def _ensure_table(self) -> None:
        query = (
            "CREATE TABLE IF NOT EXISTS session_memory ("
            "session_id TEXT,"
            "user_msg TEXT,"
            "bot_msg TEXT,"
            "summary TEXT,"
            "created_at TIMESTAMP DEFAULT NOW()"
            ")"
        )
        index = (
            "CREATE INDEX IF NOT EXISTS idx_session_memory_session_created_at "
            "ON session_memory (session_id, created_at DESC)"
        )
        conn = DatabaseManager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query)
                cur.execute(index)
                conn.commit()
        finally:
            conn.close()

    def store_turn(self, session_id: str, user_msg: str, bot_msg: str, summary: Optional[str] = None) -> None:
        logger.debug("Storing turn for session %s", session_id)
        conn = DatabaseManager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO session_memory (session_id, user_msg, bot_msg, summary) VALUES (%s, %s, %s, %s)",
                    (session_id, user_msg, bot_msg, summary),
                )
                cur.execute(
                    "DELETE FROM session_memory "
                    "WHERE session_id = %s AND ctid NOT IN ("
                    " SELECT ctid FROM session_memory "
                    " WHERE session_id = %s ORDER BY created_at DESC LIMIT %s"
                    ")",
                    (session_id, session_id, self.max_turns),
                )
                conn.commit()
                logger.debug("Stored turn and trimmed history for session %s", session_id)
        except Exception:
            logger.exception("Failed storing conversation turn")
            conn.rollback()
        finally:
            conn.close()

    def get_history(self, session_id: str, last_n_turns: int = 10) -> List[Tuple[str, str]]:
        logger.debug("Fetching last %s turns for session %s", last_n_turns, session_id)
        conn = DatabaseManager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_msg, bot_msg FROM session_memory "
                    "WHERE session_id = %s ORDER BY created_at DESC LIMIT %s",
                    (session_id, last_n_turns),
                )
                rows = cur.fetchall()
            rows.reverse()
            logger.debug("Fetched %s turns for session %s", len(rows), session_id)
            return [(r[0], r[1]) for r in rows]
        except Exception:
            logger.exception("Failed retrieving conversation history")
            return []
        finally:
            conn.close()

    def clear(self, session_id: str) -> None:
        logger.debug("Clearing history for session %s", session_id)
        conn = DatabaseManager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM session_memory WHERE session_id = %s", (session_id,))
                conn.commit()
                logger.debug("Cleared history for session %s", session_id)
        except Exception:
            logger.exception("Failed clearing session history")
            conn.rollback()
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        conn = DatabaseManager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM session_memory")
                total = cur.fetchone()[0]
            return {"total_rows": total}
        except Exception:
            logger.exception("Failed retrieving stats")
            return {}
        finally:
            conn.close()


# Keep Redis implementation available through existing module for backward compatibility
try:
    from services.simple_redis_memory import SimpleRedisMemory as RedisSessionMemory
except Exception:  # pragma: no cover
    RedisSessionMemory = None