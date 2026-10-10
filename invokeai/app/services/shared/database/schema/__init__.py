"""The application schema as SQLAlchemy metadata, one module per domain.

It is the schema the SQLite migrations build (`tests/app/services/shared/database/test_schema_parity.py` holds
the two to each other), and the schema a server database is created with. A schema change is a migration and
the matching change here, together.
"""

# Every domain module is imported, so that `metadata` holds every table.
from invokeai.app.services.shared.database.schema import (  # noqa: F401
    app_settings,
    boards,
    client_state,
    fonts,
    image_index,
    image_moves,
    images,
    intermediates,
    locks,
    media_references,
    migrator,
    models,
    projects,
    session_queue,
    style_presets,
    system_prompts,
    users,
    videos,
    wildcards,
    workflows,
)
from invokeai.app.services.shared.database.schema.metadata import metadata

__all__ = ["metadata"]
