-- Multiple chats per user: each conversation has its own message history.

CREATE TABLE IF NOT EXISTS conversations (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT 'New chat',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);

-- Link existing chats table to conversations.
ALTER TABLE chats
    ADD COLUMN IF NOT EXISTS conversation_id BIGINT
    REFERENCES conversations(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_chats_conversation ON chats(conversation_id);

-- Backfill: move any pre-existing (ungrouped) chats into one conversation per user
-- so old history is not lost once the app starts filtering by conversation_id.
DO $$
DECLARE
    rec RECORD;
    cid BIGINT;
BEGIN
    FOR rec IN SELECT DISTINCT user_id FROM chats WHERE conversation_id IS NULL LOOP
        INSERT INTO conversations (user_id, title)
        VALUES (rec.user_id, 'Imported history')
        RETURNING id INTO cid;

        UPDATE chats
        SET conversation_id = cid
        WHERE user_id = rec.user_id AND conversation_id IS NULL;
    END LOOP;
END $$;
