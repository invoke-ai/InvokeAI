"""One-shot migration of an existing InvokeAI SQLite database to an external backend.

The local SQLite DB (built by the raw-SQL migrations) is copied row-for-row into a freshly
created external backend whose schema is built by ``SQLModel.metadata.create_all()``. DB-
generated columns are recomputed by the target.

Usage:

    uv run --extra cuda python scripts/migrate_db_to_external.py \
        --target-url "mysql+pymysql://invoke:secret@127.0.0.1:3306/invokeai"

The target database must already exist and be empty. Requires the matching driver in the venv
(e.g. ``uv pip install pymysql`` for MySQL/MariaDB). The source is read-only and untouched.
"""

import argparse
from unittest import mock

from invokeai.app.services.config.config_default import get_config
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.sqlite.db_copy import copy_database
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.backend.util.logging import InvokeAILogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy the local SQLite database to an external backend.")
    parser.add_argument(
        "--target-url",
        required=True,
        help="SQLAlchemy URL of the target backend, e.g. mysql+pymysql://user:pass@host:3306/invokeai",
    )
    args = parser.parse_args()

    logger = InvokeAILogger.get_logger("db-migrate")
    config = get_config()

    # Source: the existing local SQLite database, built/upgraded by the migrations. The
    # image_files service is only used by migration 2, which won't re-run on an already-
    # migrated DB, so a stub is sufficient.
    image_files = mock.Mock(spec=ImageFileStorageBase)
    source_config = config.model_copy(update={"db_url": None})
    source = init_db(config=source_config, logger=logger, image_files=image_files)

    # Target: the external backend, schema created from SQLModel.metadata.
    target_config = config.model_copy(update={"db_url": args.target_url})
    target = init_db(config=target_config, logger=logger, image_files=image_files)

    logger.info(f"Copying SQLite database -> {args.target_url.split('@')[-1]}")
    counts = copy_database(source=source, target=target, logger=logger)

    total = sum(counts.values())
    logger.info(f"Done. Copied {total} rows across {len(counts)} tables.")


if __name__ == "__main__":
    main()
