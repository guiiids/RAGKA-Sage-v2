import json
import time
import os
from threading import Lock
from typing import Optional

_log_lock = Lock()


def truncate_text(text: Optional[str], max_chars: int = 200) -> Optional[str]:
    """Return ``text`` truncated to ``max_chars`` characters."""
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _extract_user_message(request: dict) -> Optional[str]:
    messages = request.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user" and "content" in msg:
            return msg["content"]
    return None


def _extract_assistant_message(response_dict: dict) -> Optional[str]:
    try:
        return response_dict["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def log_openai_call(
    request: dict,
    response,
    latency: float = 0.0,
    include_full_payload: bool = False,
) -> None:
    """Log an OpenAI request/response pair to ``logs/openai_calls.jsonl``."""
    os.makedirs("logs", exist_ok=True)

    response_dict = response.to_dict() if hasattr(response, "to_dict") else dict(response)
    user_msg = _extract_user_message(request)
    assistant_msg = _extract_assistant_message(response_dict)

    record = {
        "timestamp": time.time(),
        "user_message": truncate_text(user_msg),
        "assistant_message": truncate_text(assistant_msg),
        "usage": response_dict.get("usage"),
        "latency": latency,
    }

    if include_full_payload:
        record["request"] = request
        record["response"] = response_dict

    path = os.path.join("logs", "openai_calls.jsonl")
    with _log_lock, open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_openai_usage(request: dict, response, latency: float) -> None:
    """Log usage information to ``logs/openai_usage.jsonl``."""
    os.makedirs("logs", exist_ok=True)

    response_dict = response.to_dict() if hasattr(response, "to_dict") else dict(response)
    user_query = _extract_user_message(request)
    response_text = _extract_assistant_message(response_dict)

    record = {
        "timestamp": time.time(),
        "user_query": truncate_text(user_query),
        "response_text": truncate_text(response_text),
        "usage": response_dict.get("usage"),
        "latency": latency,
    }

    path = os.path.join("logs", "openai_usage.jsonl")
    with _log_lock, open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

