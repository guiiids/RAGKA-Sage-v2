"""
A minimal Redis-backed conversation memory service for simple contextual RAG.
- Stores and retrieves conversation history by session.
- No classification, truncation, or aggregation – always returns last N (Q, A) turns.
"""

import os
import json
from typing import List, Tuple, Optional
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASS = os.getenv("REDIS_PASSWORD", None)
REDIS_EXPIRATION = int(os.getenv("REDIS_DEFAULT_EXPIRATION", "604800"))  # 7 days

class SimpleRedisMemory:
    def __init__(self):
        self._client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASS,
            decode_responses=True,
            socket_timeout=5
        )

    def _key(self, session_id: str) -> str:
        return f"simple_history:{session_id}"

    def store_turn(self, session_id: str, user_query: str, assistant_response: str) -> None:
        turn = {
            "user": user_query,
            "assistant": assistant_response
        }
        self._client.rpush(self._key(session_id), json.dumps(turn))
        self._client.expire(self._key(session_id), REDIS_EXPIRATION)

    def get_history(self, session_id: str, last_n_turns: int = 5) -> List[Tuple[str, str]]:
        # Returns a list of (user_query, assistant_response) tuples, oldest to newest (last N only)
        raw = self._client.lrange(self._key(session_id), -last_n_turns, -1)
        history: List[Tuple[str, str]] = []
        for entry in raw:
            try:
                turn = json.loads(entry)
                history.append((turn.get("user", ""), turn.get("assistant", "")))
            except Exception:
                continue
        return history

    def clear(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))

memory_service = SimpleRedisMemory()
