"""Give each system prompt its own optional cap on the Expand Prompt LLM's output length.

Expand Prompt asks the text LLM for at most ``EXPAND_PROMPT_MAX_TOKENS_DEFAULT`` (300) new
tokens. That suits the one-paragraph rewrites the original seeded prompts produce, but truncates
prompts that are structurally longer by design -- the MiniMax H3 Ref2VA prompt emits six labelled
sections and routinely needs three to five times as much.

The cap therefore becomes a property of the system prompt: ``max_tokens`` is NULL for every
existing row (meaning "use the default"), and is set to 1000 for the Ref2VA prompt.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_07_add_minimax_h3_ref2va_system_prompt import (
    MINIMAX_H3_REF2VA_PROMPT_ID,
)
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration

MINIMAX_H3_REF2VA_MAX_TOKENS = 1000
"""Headroom for the six-section structured prompt.

Measured with the Qwen BPE tokenizer these models use, a realistic output costs roughly
75 tokens of fixed structure, +55 per reference (a `subject_definitions` bullet plus a
`retention_analysis` bullet) and +155 per shot, so three references and one shot land near 470
and five references with two shots near 745. The two audio sections come last and are cheap
(~45 combined), which is why a cap sized off the prompt's own terse worked example (187 tokens)
truncated them first. `max_new_tokens` is a safety stop rather than a target -- generation ends
at EOS -- so this is sized for the worst realistic request, not the median one.
"""


class AddSystemPromptMaxTokensCallback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("PRAGMA table_info(system_prompts);")
        # SQLite has no ADD COLUMN IF NOT EXISTS. The migrator already guarantees this callback
        # runs at most once per database and rolls the whole thing back on failure, so this guard
        # is for a hand-modified database only -- not a sequence the app can produce.
        if "max_tokens" not in {row[1] for row in cursor.fetchall()}:
            cursor.execute("ALTER TABLE system_prompts ADD COLUMN max_tokens INTEGER;")

        # Every row is NULL immediately after the ALTER above, so the IS NULL scope only matters
        # if the column already existed -- see the guard above. Kept so that re-running the
        # callback by hand cannot overwrite a cap the user has since chosen.
        cursor.execute(
            """--sql
            UPDATE system_prompts SET max_tokens = ? WHERE id = ? AND max_tokens IS NULL;
            """,
            (MINIMAX_H3_REF2VA_MAX_TOKENS, MINIMAX_H3_REF2VA_PROMPT_ID),
        )


def build_migration() -> Migration:
    """Add ``system_prompts.max_tokens`` and give the Ref2VA prompt the room it needs.

    Depends on the Ref2VA seed rather than on the table creation: the row it raises the cap on
    must already exist.
    """
    return Migration(
        id="2026_09_09_add_system_prompt_max_tokens",
        depends_on="2026_09_07_add_minimax_h3_ref2va_system_prompt",
        callback=AddSystemPromptMaxTokensCallback(),
    )
