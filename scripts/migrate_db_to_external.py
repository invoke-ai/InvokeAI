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
from logging import Logger
from pathlib import Path
from unittest import mock

from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import make_url

from invokeai.app.services.config.config_default import get_config
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.sqlite.db_copy import copy_database
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.backend.util.logging import InvokeAILogger


def _ensure_database_exists(target_url: str, logger: Logger) -> None:
    """Create the target database if it does not exist (MySQL/MariaDB).

    SQLite auto-creates its file; Postgres needs a maintenance connection and is left to the
    user. For MySQL/MariaDB we connect to the server without selecting a database and issue a
    ``CREATE DATABASE IF NOT EXISTS`` with utf8mb4 so JSON/text payloads are stored safely.
    """
    url = make_url(target_url)
    if not url.get_backend_name().startswith("mysql"):
        return
    db_name = url.database
    if not db_name:
        return
    # Connect to the server itself (no database selected) to issue CREATE DATABASE. URL.set()
    # does not drop the database component, so build a fresh URL without it.
    server_url = URL.create(
        drivername=url.drivername,
        username=url.username,
        password=url.password,
        host=url.host,
        port=url.port,
        query=url.query,
    )
    server_engine = create_engine(server_url)
    try:
        with server_engine.connect() as conn:
            conn.execute(
                text(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            )
            conn.commit()
        logger.info(f"Ensured target database `{db_name}` exists (utf8mb4)")
    finally:
        server_engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy the local SQLite database to an external backend.")
    parser.add_argument(
        "--target-url",
        required=True,
        help="SQLAlchemy URL of the target backend, e.g. mysql+pymysql://user:pass@host:3306/invokeai",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="InvokeAI root directory holding databases/invokeai.db (the same value you pass to "
        "invokeai-web --root). Defaults to $INVOKEAI_ROOT / the standard root resolution.",
    )
    args = parser.parse_args()

    logger = InvokeAILogger.get_logger("db-migrate")
    config = get_config()
    # This is a standalone script, so the app never parsed --root for us; apply it (or fall
    # back to $INVOKEAI_ROOT via root_path) and pin the resolved root onto both config copies.
    if args.root:
        config._root = Path(args.root).expanduser().resolve()
    root = config.root_path
    logger.info(f"InvokeAI root: {root}")

    # Source: the existing local SQLite database, built/upgraded by the migrations. The
    # image_files service is only used by migration 2, which won't re-run on an already-
    # migrated DB, so a stub is sufficient.
    image_files = mock.Mock(spec=ImageFileStorageBase)
    source_config = config.model_copy(update={"db_url": None})
    source_config._root = config._root
    logger.info(f"Source SQLite database: {source_config.db_path}")
    if not source_config.db_path.exists():
        raise SystemExit(f"No SQLite database at {source_config.db_path} — is --root correct?")
    source = init_db(config=source_config, logger=logger, image_files=image_files)

    # Target: the external backend, schema created from SQLModel.metadata.
    _ensure_database_exists(args.target_url, logger)
    target_config = config.model_copy(update={"db_url": args.target_url})
    target_config._root = config._root
    target = init_db(config=target_config, logger=logger, image_files=image_files)

    logger.info(f"Copying SQLite database -> {args.target_url.split('@')[-1]}")
    counts = copy_database(source=source, target=target, logger=logger)

    total = sum(counts.values())
    logger.info(f"Done. Copied {total} rows across {len(counts)} tables.")


if __name__ == "__main__":
    main()
