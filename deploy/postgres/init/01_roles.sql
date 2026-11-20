-- Runs once on an empty volume. The agent's role can only read, with a hard timeout (ADR-003).
CREATE DATABASE lexicon_state;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'lexicon_ro') THEN
    CREATE ROLE lexicon_ro LOGIN PASSWORD 'lexicon_ro_local';
  END IF;
END $$;
ALTER ROLE lexicon_ro SET default_transaction_read_only = on;
ALTER ROLE lexicon_ro SET statement_timeout = '5s';
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
