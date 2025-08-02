import json
import time
import os
from threading import Lock

_log_lock = Lock()

def log_openai_call(request: dict, response) -> None:
    """
    Append each OpenAI request and response as a JSON object
    (one per line) into logs/openai_calls.jsonl.
    """
    os.makedirs('logs', exist_ok=True)
    record = {
        "timestamp": time.time(),
        "request": request,
        # response may be an OpenAI response object with to_dict()
        "response": response.to_dict() if hasattr(response, "to_dict") else dict(response)
    }
    path = os.path.join('logs', 'openai_calls.jsonl')
    with _log_lock, open(path, 'a') as f:
        f.write(json.dumps(record) + "\\n")

def log_openai_usage(request: dict, response, latency: float) -> None:
    """
    Append user query, response text, usage info, and latency as a JSON object
    (one per line) into logs/openai_usage.jsonl.
    """
    os.makedirs('logs', exist_ok=True)

    # Extract user query from request messages (last user message content)
    user_query = None
    messages = request.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user" and "content" in msg:
            user_query = msg["content"]
            break

    # Extract response text from response object
    response_dict = response.to_dict() if hasattr(response, "to_dict") else dict(response)
    response_text = None
    try:
        response_text = response_dict["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        response_text = None

    # Extract usage info if available
    usage = response_dict.get("usage")

    record = {
        "timestamp": time.time(),
        "user_query": user_query,
        "response_text": response_text,
        "usage": usage,
        "latency": latency,
    }
    path = os.path.join('logs', 'openai_usage.jsonl')
    with _log_lock, open(path, 'a') as f:
        f.write(json.dumps(record) + "\\n")
