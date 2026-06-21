-- Optional metadata for assistant reply provenance (FAQ / document / multi-FAQ).

ALTER TABLE chats
    ADD COLUMN IF NOT EXISTS response_source TEXT;

ALTER TABLE chats
    ADD COLUMN IF NOT EXISTS response_document_name TEXT;
