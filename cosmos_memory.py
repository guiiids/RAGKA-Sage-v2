import os, uuid, datetime as dt
from azure.cosmos import CosmosClient, PartitionKey

COSMOS_URI = os.environ["COSMOS_URI"]
COSMOS_KEY = os.environ["COSMOS_KEY"]
DB_NAME = "rag"
CONTAINER_NAME = "sessions"

_client = CosmosClient(COSMOS_URI, COSMOS_KEY)
_db = _client.create_database_if_not_exists(DB_NAME)
_container = _db.create_container_if_not_exists(
    id=CONTAINER_NAME,
    partition_key=PartitionKey(path="/sessionId"),
    default_ttl=86400
)

def _now_iso():
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00","Z")

def save_turn(session_id: str, role: str, content: str, order: int, ttl: int|None=None, tokens: int|None=None):
    doc = {
        "id": str(uuid.uuid4()),
        "type": "message",
        "sessionId": session_id,
        "role": role,
        "content": content,
        "order": order,
        "createdAt": _now_iso()
    }
    if ttl:
        doc["ttl"] = ttl
    if tokens is not None:
        doc["tokens"] = tokens
    _container.upsert_item(doc)

def load_context(session_id: str, max_tokens: int = 1200, max_turns: int = 12):
    query = """
    SELECT c.id, c.role, c.content, c.order
    FROM c
    WHERE c.sessionId = @sid AND c.type = 'message'
    ORDER BY c.order DESC
    """
    items = list(_container.query_items(
        query=query,
        parameters=[{"name":"@sid","value":session_id}],
        enable_cross_partition_query=False
    ))
    context, tokens = [], 0
    for it in items:
        context.append({"role": it["role"], "content": it["content"]})
        tokens += max(1, len(it["content"])//4)
        if len(context) >= max_turns or tokens >= max_tokens:
            break
    return list(reversed(context))

def update_summary(session_id: str, content: str, turns: int):
    _container.upsert_item({
        "id": f"summary::{session_id}",
        "type": "summary",
        "sessionId": session_id,
        "content": content,
        "turnsCovered": turns,
        "updatedAt": _now_iso()
    })

def next_order_for(session_id: str) -> int:
    query = """
    SELECT VALUE MAX(c.order) FROM c WHERE c.sessionId=@sid AND c.type='message'
    """
    items = list(_container.query_items(
        query=query,
        parameters=[{"name":"@sid","value":session_id}],
        enable_cross_partition_query=False
    ))
    max_order = items[0] if items else None
    return (max_order or 0) + 1
