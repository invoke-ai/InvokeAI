"""The application schema on the backend under test: what its constraints, generated columns and defaults do.

`test_schema_parity.py` shows that the metadata is the migrated SQLite schema; these show that the tables it
creates behave alike on every backend.
"""

import json
import re
from typing import Any

import pytest
from sqlalchemy import (
    ForeignKeyConstraint,
    Table,
    UniqueConstraint,
    create_mock_engine,
    delete,
    func,
    insert,
    select,
    update,
)

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.engines import MARIADB_BINARY_COLLATION, MYSQL_BINARY_COLLATION
from invokeai.app.services.shared.database.errors import (
    CheckViolation,
    ForeignKeyViolation,
    NotNullViolation,
    UniqueViolation,
)
from invokeai.app.services.shared.database.schema import metadata
from invokeai.app.services.shared.database.schema.boards import board_images, boards
from invokeai.app.services.shared.database.schema.fonts import fonts
from invokeai.app.services.shared.database.schema.image_index import image_embeddings, image_index_vocab_terms
from invokeai.app.services.shared.database.schema.image_moves import (
    image_subfolder_move_items,
    image_subfolder_move_jobs,
)
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.models import models
from invokeai.app.services.shared.database.schema.projects import projects
from invokeai.app.services.shared.database.schema.session_queue import session_queue
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.shared.database.schema.videos import videos
from invokeai.app.services.shared.database.schema.wildcards import wildcards
from invokeai.app.services.shared.database.schema.workflows import workflow_library
from invokeai.app.services.shared.database.types import Key, LongText, NoCaseKey
from tests.fixtures.database import external_test_db_url

server_only = pytest.mark.skipif(
    external_test_db_url() is None, reason="needs a MySQL or MariaDB server (INVOKEAI_TEST_DB_URL)"
)

MODEL_CONFIG: dict[str, Any] = {
    "hash": "blake3:0123456789abcdef",
    "base": "sdxl",
    "type": "main",
    "path": "sdxl/main/model.safetensors",
    "format": "checkpoint",
    "name": "Model",
    "description": None,
    "source": "https://example.com/model.safetensors",
    "source_type": "url",
    "file_size": 6_938_078_334,
    "trigger_phrases": ["a photo"],
}


def _create(database: Database, *tables: Table) -> None:
    with database.begin(write=True) as conn:
        metadata.create_all(conn, tables=list(tables))


def _image(name: str) -> dict[str, Any]:
    return {"image_name": name, "image_origin": "internal", "image_category": "general", "width": 8, "height": 8}


@pytest.mark.parametrize("url", ["mysql+pymysql://", "mariadb+pymysql://", "postgresql://"])
def test_the_schema_compiles_for_each_server_backend(url: str) -> None:
    # Without a server or a driver: every type and construct compiles for the backend, PostgreSQL (not yet
    # supported) included, and an index that differs per backend is created once.
    statements: list[str] = []

    def compile_statement(statement: Any, *args: Any, **kwargs: Any) -> None:
        statements.append(str(statement.compile(dialect=engine.dialect)))

    engine = create_mock_engine(url, compile_statement)
    metadata.create_all(engine, checkfirst=False)

    creates_index = re.compile(r"\s*CREATE (?:UNIQUE )?INDEX (\S+)")
    indexes = [match.group(1) for match in map(creates_index.match, statements) if match]
    assert len(statements) == len(metadata.tables) + len(indexes)
    assert sorted(set(indexes)) == sorted(indexes)
    if not url.startswith("postgresql"):
        # Keys of up to 3072 bytes need the DYNAMIC row format, whatever a server defaults to.
        assert all("ROW_FORMAT=DYNAMIC" in s for s in statements if not creates_index.match(s))


def test_text_columns_are_bounded_exactly_where_a_server_indexes_them_whole() -> None:
    for table in metadata.sorted_tables:
        indexed_whole = {column.name for column in table.primary_key.columns}
        for constraint in table.constraints:
            if isinstance(constraint, (UniqueConstraint, ForeignKeyConstraint)):
                indexed_whole.update(column.name for column in constraint.columns)
        for index in table.indexes:
            if index._ddl_if is not None and index._ddl_if.dialect == "sqlite":
                continue
            prefix = index.dialect_options["mysql"]["length"] or {}
            indexed_whole.update(
                column.name for column in index.columns if not (isinstance(prefix, int) or column.name in prefix)
            )
        for column in table.columns:
            if isinstance(column.type, (Key, NoCaseKey)):
                assert column.name in indexed_whole, f"{table.name}.{column.name}: a Key no server index needs"
            elif isinstance(column.type, LongText):
                assert column.name not in indexed_whole, f"{table.name}.{column.name}: LongText indexed whole"


@server_only
def test_server_tables_compare_text_byte_for_byte_whatever_the_database_default(empty_database: Database) -> None:
    binary = MARIADB_BINARY_COLLATION if empty_database.dialect_name == "mariadb" else MYSQL_BINARY_COLLATION
    # A server's default collation (and so a new database's) folds case.
    with empty_database.begin(write=True) as conn:
        conn.exec_driver_sql("ALTER DATABASE COLLATE utf8mb4_general_ci")
    try:
        with empty_database.begin(write=True) as conn:
            metadata.create_all(conn)
    finally:
        with empty_database.begin(write=True) as conn:
            conn.exec_driver_sql(f"ALTER DATABASE COLLATE {binary}")

    with empty_database.begin(write=False) as conn:
        tables = conn.exec_driver_sql(
            "SELECT engine, table_collation FROM information_schema.tables WHERE table_schema = DATABASE()"
        ).all()
    assert len(tables) == len(metadata.tables)
    assert set(tables) == {("InnoDB", binary)}


def test_generated_columns_hold_members_of_the_json_document(empty_database: Database) -> None:
    _create(empty_database, models, workflow_library)
    workflow = {"name": "Flow", "description": "", "meta": {"category": "user"}, "tags": "a, b"}

    with empty_database.begin(write=True) as conn:
        conn.execute(insert(models).values(id="m1", config=json.dumps(MODEL_CONFIG)))
        conn.execute(insert(workflow_library).values(workflow_id="w1", workflow=json.dumps(workflow)))

    with empty_database.begin(write=False) as conn:
        model = conn.execute(select(models)).one()._mapping
        flow = conn.execute(select(workflow_library)).one()._mapping

    for member in ("hash", "base", "type", "path", "format", "name", "source", "source_type", "file_size"):
        assert model[member] == MODEL_CONFIG[member], member
    # A JSON null is NULL, not the text 'null'.
    assert model["description"] is None
    assert (flow["name"], flow["description"], flow["category"], flow["tags"]) == ("Flow", "", "user", "a, b")


@pytest.mark.parametrize("member", ["hash", "file_size"])
@pytest.mark.parametrize("missing", ["null", "absent"])
def test_a_model_config_without_a_required_member_is_refused(
    empty_database: Database, member: str, missing: str
) -> None:
    _create(empty_database, models)
    config = {key: value for key, value in MODEL_CONFIG.items() if key != member}
    if missing == "null":
        config[member] = None

    # NOT NULL on the generated column; on MariaDB, which has none for generated columns, a CHECK.
    expected = CheckViolation if empty_database.dialect_name == "mariadb" else NotNullViolation
    with pytest.raises(expected):
        with empty_database.begin(write=True) as conn:
            conn.execute(insert(models).values(id="m1", config=json.dumps(config)))


def test_foreign_keys_cascade_set_null_or_restrict_as_declared(empty_database: Database) -> None:
    _create(empty_database, users, images, boards, board_images, projects)
    with empty_database.begin(write=True) as conn:
        conn.execute(insert(users).values(user_id="u1", email="u1@example.com", password_hash="hash"))
        conn.execute(insert(images).values(_image("i1.png")))
        conn.execute(insert(boards).values(board_id="b1", board_name="Board", cover_image_name="i1.png"))
        conn.execute(insert(boards).values(board_id="b2", board_name="Project board"))
        conn.execute(insert(board_images).values(board_id="b1", image_name="i1.png"))
        conn.execute(insert(projects).values(project_id="p1", user_id="u1", name="Project", data="{}", board_id="b2"))

    with empty_database.begin(write=True) as conn:
        conn.execute(delete(images).where(images.c.image_name == "i1.png"))
    with empty_database.begin(write=False) as conn:
        assert conn.execute(select(boards.c.cover_image_name).where(boards.c.board_id == "b1")).scalar_one() is None
        assert conn.execute(select(func.count()).select_from(board_images)).scalar_one() == 0

    # A project's board cannot be deleted while the project exists.
    with pytest.raises(ForeignKeyViolation):
        with empty_database.begin(write=True) as conn:
            conn.execute(delete(boards).where(boards.c.board_id == "b2"))

    with empty_database.begin(write=True) as conn:
        conn.execute(delete(users).where(users.c.user_id == "u1"))
    with empty_database.begin(write=False) as conn:
        assert conn.execute(select(func.count()).select_from(projects)).scalar_one() == 0


def test_a_move_job_with_items_cannot_be_deleted(empty_database: Database) -> None:
    # The items' foreign key has no ON DELETE action: the job's rows are the audit trail of its moves.
    _create(empty_database, images, image_subfolder_move_jobs, image_subfolder_move_items)
    with empty_database.begin(write=True) as conn:
        conn.execute(insert(images).values(_image("i1.png")))
        job = conn.execute(insert(image_subfolder_move_jobs).values(state="planned")).inserted_primary_key
        assert job is not None
        conn.execute(
            insert(image_subfolder_move_items).values(
                job_id=job.id, image_name="i1.png", old_subfolder="", new_subfolder="a", state="planned"
            )
        )

    with pytest.raises(ForeignKeyViolation):
        with empty_database.begin(write=True) as conn:
            conn.execute(delete(image_subfolder_move_jobs))


def test_vocabulary_terms_are_unique_whatever_their_case(empty_database: Database) -> None:
    _create(empty_database, image_index_vocab_terms)
    with empty_database.begin(write=True) as conn:
        conn.execute(insert(image_index_vocab_terms).values(term="Cat"))
        # Trailing spaces still make a different term, as in SQLite.
        conn.execute(insert(image_index_vocab_terms).values(term="Cat "))

    with pytest.raises(UniqueViolation):
        with empty_database.begin(write=True) as conn:
            conn.execute(insert(image_index_vocab_terms).values(term="cat"))


def test_a_directory_font_path_is_unique_and_uploaded_fonts_have_none(empty_database: Database) -> None:
    # A partial unique index on SQLite, a full one on a server: the same constraint, because only directory
    # fonts have a source path.
    _create(empty_database, users, fonts)
    font = {"filename": "f.ttf", "family": "F", "label": "F", "style": "normal", "weight": 400, "byte_size": 1}
    with empty_database.begin(write=True) as conn:
        conn.execute(insert(users).values(user_id="u1", email="u1@example.com", password_hash="hash"))
        for n in (1, 2):
            uploaded = {"id": f"u{n}", "owner_id": "u1", "scope": "private", "source": "uploaded"}
            conn.execute(
                insert(fonts).values({**font, **uploaded, "storage_path": f"u{n}.ttf", "content_hash": f"{n}" * 64})
            )
        directory = {"scope": "shared", "source": "directory", "source_path": "a/f.ttf"}
        conn.execute(insert(fonts).values({**font, **directory, "id": "d1", "content_hash": "3" * 64}))

    with pytest.raises(UniqueViolation):
        with empty_database.begin(write=True) as conn:
            conn.execute(insert(fonts).values({**font, **directory, "id": "d2", "content_hash": "4" * 64}))


def test_other_names_differ_by_case(empty_database: Database) -> None:
    _create(empty_database, users, wildcards)
    with empty_database.begin(write=True) as conn:
        conn.execute(insert(users).values(user_id="u1", email="u1@example.com", password_hash="hash"))
        conn.execute(insert(wildcards).values(id="w1", name="Animals", user_id="u1"))
        conn.execute(insert(wildcards).values(id="w2", name="animals", user_id="u1"))

    with empty_database.begin(write=False) as conn:
        assert conn.execute(select(func.count()).select_from(wildcards)).scalar_one() == 2


def test_queue_item_ids_are_never_reused(empty_database: Database) -> None:
    _create(empty_database, session_queue)
    item = {"batch_id": "b", "queue_id": "default", "session": "{}"}
    with empty_database.begin(write=True) as conn:
        conn.execute(insert(session_queue).values({**item, "session_id": "s1"}))
        conn.execute(insert(session_queue).values({**item, "session_id": "s2"}))
        conn.execute(delete(session_queue).where(session_queue.c.session_id == "s2"))
        third = conn.execute(insert(session_queue).values({**item, "session_id": "s3"})).inserted_primary_key

    assert third is not None
    assert third.item_id == 3


def test_omitted_columns_take_their_defaults(empty_database: Database) -> None:
    _create(empty_database, images, boards)
    with empty_database.begin(write=True) as conn:
        conn.execute(insert(boards).values(board_id="b1", board_name="Board"))
        conn.execute(insert(images).values(_image("i1.png")))

    with empty_database.begin(write=False) as conn:
        board = conn.execute(select(boards)).one()._mapping
        image = conn.execute(select(images.c.image_subfolder, images.c.is_intermediate, images.c.user_id)).one()

    assert (board["archived"], board["user_id"], board["is_public"], board["board_visibility"]) == (
        False,
        "system",
        False,
        "private",
    )
    for timestamp in (board["created_at"], board["updated_at"]):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}", timestamp)
    assert tuple(image) == ("", False, "system")


def test_an_update_sets_updated_at(empty_database: Database) -> None:
    _create(empty_database, images, boards)
    long_ago = "2020-01-01 00:00:00.000"
    with empty_database.begin(write=True) as conn:
        conn.execute(insert(boards).values(board_id="b1", board_name="Board", created_at=long_ago, updated_at=long_ago))
        conn.execute(update(boards).values(board_name="Renamed"))

    with empty_database.begin(write=False) as conn:
        created_at, updated_at = conn.execute(select(boards.c.created_at, boards.c.updated_at)).one()

    assert created_at == long_ago
    assert isinstance(updated_at, str) and updated_at > long_ago


def test_every_updated_at_column_is_set_by_updates() -> None:
    # No trigger keeps `updated_at` current on any backend: an `update()` of its table must set it. (Fonts set
    # theirs where a directory font changes; the quarantine table of old projects keeps theirs as found.)
    set_elsewhere = {"fonts", "orphaned_projects_2026_08_06"}
    without = [
        name
        for name, table in metadata.tables.items()
        if "updated_at" in table.c and name not in set_elsewhere and table.c.updated_at.onupdate is None
    ]
    assert without == []


def test_columns_hold_long_and_large_values(empty_database: Database) -> None:
    _create(empty_database, models, images, image_embeddings, videos)
    # A model name is indexed by its first characters on a server; a path, which is unique, is limited to
    # what a server's index holds whole.
    config = MODEL_CONFIG | {"name": "n" * 1000, "path": "p" * 768}
    embedding = bytes(range(256)) * 400  # beyond the 64 KiB of a server's BLOB
    data = json.dumps({"layers": "x" * 100_000})  # beyond the 64 KiB of a server's TEXT

    with empty_database.begin(write=True) as conn:
        conn.execute(insert(models).values(id="m1", config=json.dumps(config)))
        conn.execute(insert(images).values({**_image("i1.png"), "metadata": data, "file_size_bytes": 2**40}))
        conn.execute(insert(image_embeddings).values(image_name="i1.png", model_id="clip", dim=1, embedding=embedding))
        conn.execute(
            insert(videos).values(
                video_name="v1.mp4",
                video_origin="internal",
                video_category="general",
                width=8,
                height=8,
                duration=1 / 3,
            )
        )

    with empty_database.begin(write=False) as conn:
        assert tuple(conn.execute(select(models.c.name, models.c.path)).one()) == (config["name"], config["path"])
        assert tuple(conn.execute(select(images.c.metadata, images.c.file_size_bytes)).one()) == (data, 2**40)
        assert conn.execute(select(image_embeddings.c.embedding)).scalar_one() == embedding
        assert conn.execute(select(videos.c.duration)).scalar_one() == 1 / 3
