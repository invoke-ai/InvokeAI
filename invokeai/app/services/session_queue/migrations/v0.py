import sqlite3


def v0(cursor: sqlite3.Cursor) -> None:
    """
    Migration for `session_queue` table v0
    https://github.com/invoke-ai/InvokeAI/pull/4502

    Creates the `session_queue` table, indicies and triggers for the session_queue service.
    """
    cursor.execute(
        """--sql
        CREATE TABLE IF NOT EXISTS session_queue (
            item_id INTEGER PRIMARY KEY AUTOINCREMENT, -- used for ordering, cursor pagination
            batch_id TEXT NOT NULL, -- identifier of the batch this queue item belongs to
            queue_id TEXT NOT NULL, -- identifier of the queue this queue item belongs to
            session_id TEXT NOT NULL UNIQUE, -- duplicated data from the session column, for ease of access
            field_values TEXT, -- NULL if no values are associated with this queue item
            session TEXT NOT NULL, -- the session to be executed
            status TEXT NOT NULL DEFAULT 'pending', -- the status of the queue item, one of 'pending', 'in_progress', 'completed', 'failed', 'canceled'
            priority INTEGER NOT NULL DEFAULT 0, -- the priority, higher is more important
            error TEXT, -- any errors associated with this queue item
            created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')), -- updated via trigger
            started_at DATETIME, -- updated via trigger
            completed_at DATETIME -- updated via trigger, completed items are cleaned up on application startup
            -- Ideally this is a FK, but graph_executions uses INSERT OR REPLACE, and REPLACE triggers the ON DELETE CASCADE...
            -- FOREIGN KEY (session_id) REFERENCES graph_executions (id) ON DELETE CASCADE
        );
        """
    )

    cursor.execute(
        """--sql
        CREATE UNIQUE INDEX IF NOT EXISTS idx_session_queue_item_id ON session_queue(item_id);
        """
    )

    cursor.execute(
        """--sql
        CREATE UNIQUE INDEX IF NOT EXISTS idx_session_queue_session_id ON session_queue(session_id);
        """
    )

    cursor.execute(
        """--sql
        CREATE INDEX IF NOT EXISTS idx_session_queue_batch_id ON session_queue(batch_id);
        """
    )

    cursor.execute(
        """--sql
        CREATE INDEX IF NOT EXISTS idx_session_queue_created_priority ON session_queue(priority);
        """
    )

    cursor.execute(
        """--sql
        CREATE INDEX IF NOT EXISTS idx_session_queue_created_status ON session_queue(status);
        """
    )

    cursor.execute(
        """--sql
        CREATE TRIGGER IF NOT EXISTS tg_session_queue_completed_at
        AFTER UPDATE OF status ON session_queue
        FOR EACH ROW
        WHEN
            NEW.status = 'completed'
            OR NEW.status = 'failed'
            OR NEW.status = 'canceled'
        BEGIN
            UPDATE session_queue
            SET completed_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
            WHERE item_id = NEW.item_id;
        END;
        """
    )

    cursor.execute(
        """--sql
        CREATE TRIGGER IF NOT EXISTS tg_session_queue_started_at
        AFTER UPDATE OF status ON session_queue
        FOR EACH ROW
        WHEN
            NEW.status = 'in_progress'
        BEGIN
            UPDATE session_queue
            SET started_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
            WHERE item_id = NEW.item_id;
        END;
        """
    )

    cursor.execute(
        """--sql
        CREATE TRIGGER IF NOT EXISTS tg_session_queue_updated_at
        AFTER UPDATE
        ON session_queue FOR EACH ROW
        BEGIN
            UPDATE session_queue
            SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
            WHERE item_id = old.item_id;
        END;
        """
    )
