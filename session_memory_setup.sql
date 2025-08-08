CREATE TABLE IF NOT EXISTS session_memory (
    session_id TEXT,
    user_msg TEXT,
    bot_msg TEXT,
    summary TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_session_memory_session_created_at
    ON session_memory (session_id, created_at DESC);
