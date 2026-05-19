"""Benchmark: SQLModel vs raw SQLite implementations.

Compares performance of the old raw-SQL services against the new SQLModel services.
Run with: pytest tests/app/services/test_sqlmodel_services/test_benchmark_sqlmodel_vs_sqlite.py -v -s
"""

import time

from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardRecordOrderBy
from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
from invokeai.app.services.board_records.board_records_sqlmodel import SqlModelBoardRecordStorage
from invokeai.app.services.image_records.image_records_common import ImageCategory, ImageRecordChanges, ResourceOrigin
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
from invokeai.app.services.image_records.image_records_sqlmodel import SqlModelImageRecordStorage
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService
from invokeai.app.services.users.users_sqlmodel import UserServiceSqlModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _time_it(func, iterations=1):
    """Run func `iterations` times and return total seconds."""
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    return time.perf_counter() - start


def _report(name: str, sqlite_time: float, sqlmodel_time: float, iterations: int):
    ratio = sqlmodel_time / sqlite_time if sqlite_time > 0 else float("inf")
    faster = "SQLModel" if ratio < 1 else "SQLite"
    factor = 1 / ratio if ratio < 1 else ratio
    print(f"\n  {name} ({iterations} iterations):")
    print(f"    SQLite:   {sqlite_time * 1000:.1f} ms")
    print(f"    SQLModel: {sqlmodel_time * 1000:.1f} ms")
    print(f"    -> {faster} is {factor:.2f}x faster")
    return sqlite_time, sqlmodel_time


# ---------------------------------------------------------------------------
# Board Records Benchmark
# ---------------------------------------------------------------------------


class TestBoardRecordsBenchmark:
    """Compare board record operations between SQLite and SQLModel."""

    N_BOARDS = 100
    N_READS = 200
    N_QUERIES = 50

    def test_insert_boards(self, db: SqliteDatabase):
        sqlite_storage = SqliteBoardRecordStorage(db=db)
        sqlmodel_storage = SqlModelBoardRecordStorage(db=db)

        def sqlite_insert():
            for i in range(self.N_BOARDS):
                sqlite_storage.save(f"sqlite_board_{i}", "user1")

        def sqlmodel_insert():
            for i in range(self.N_BOARDS):
                sqlmodel_storage.save(f"sqlmodel_board_{i}", "user1")

        t_sqlite = _time_it(sqlite_insert)
        t_sqlmodel = _time_it(sqlmodel_insert)
        _report("INSERT boards", t_sqlite, t_sqlmodel, self.N_BOARDS)

    def test_get_boards(self, db: SqliteDatabase):
        sqlite_storage = SqliteBoardRecordStorage(db=db)
        sqlmodel_storage = SqlModelBoardRecordStorage(db=db)

        # Setup
        board_ids = []
        for i in range(self.N_BOARDS):
            b = sqlite_storage.save(f"board_{i}", "user1")
            board_ids.append(b.board_id)

        def sqlite_get():
            for bid in board_ids:
                sqlite_storage.get(bid)

        def sqlmodel_get():
            for bid in board_ids:
                sqlmodel_storage.get(bid)

        t_sqlite = _time_it(sqlite_get, iterations=3)
        t_sqlmodel = _time_it(sqlmodel_get, iterations=3)
        _report("GET boards (by ID)", t_sqlite, t_sqlmodel, self.N_BOARDS * 3)

    def test_get_many_boards(self, db: SqliteDatabase):
        sqlite_storage = SqliteBoardRecordStorage(db=db)
        sqlmodel_storage = SqlModelBoardRecordStorage(db=db)

        for i in range(self.N_BOARDS):
            sqlite_storage.save(f"board_{i}", "user1")

        def sqlite_query():
            sqlite_storage.get_many(
                user_id="user1",
                is_admin=False,
                order_by=BoardRecordOrderBy.CreatedAt,
                direction=SQLiteDirection.Descending,
                offset=0,
                limit=20,
            )

        def sqlmodel_query():
            sqlmodel_storage.get_many(
                user_id="user1",
                is_admin=False,
                order_by=BoardRecordOrderBy.CreatedAt,
                direction=SQLiteDirection.Descending,
                offset=0,
                limit=20,
            )

        t_sqlite = _time_it(sqlite_query, iterations=self.N_QUERIES)
        t_sqlmodel = _time_it(sqlmodel_query, iterations=self.N_QUERIES)
        _report("GET MANY boards (paginated)", t_sqlite, t_sqlmodel, self.N_QUERIES)

    def test_update_boards(self, db: SqliteDatabase):
        sqlite_storage = SqliteBoardRecordStorage(db=db)
        sqlmodel_storage = SqlModelBoardRecordStorage(db=db)

        board_ids = []
        for i in range(self.N_BOARDS):
            b = sqlite_storage.save(f"board_{i}", "user1")
            board_ids.append(b.board_id)

        def sqlite_update():
            for bid in board_ids:
                sqlite_storage.update(bid, BoardChanges(board_name="updated"))

        def sqlmodel_update():
            for bid in board_ids:
                sqlmodel_storage.update(bid, BoardChanges(board_name="updated"))

        t_sqlite = _time_it(sqlite_update)
        t_sqlmodel = _time_it(sqlmodel_update)
        _report("UPDATE boards", t_sqlite, t_sqlmodel, self.N_BOARDS)


# ---------------------------------------------------------------------------
# Image Records Benchmark
# ---------------------------------------------------------------------------


class TestImageRecordsBenchmark:
    """Compare image record operations between SQLite and SQLModel."""

    N_IMAGES = 200
    N_QUERIES = 50

    def _save_images(self, storage, prefix: str, n: int):
        for i in range(n):
            storage.save(
                image_name=f"{prefix}_{i}",
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                width=512,
                height=512,
                has_workflow=False,
                is_intermediate=(i % 5 == 0),
                starred=(i % 10 == 0),
                user_id="user1",
            )

    def test_insert_images(self, db: SqliteDatabase):
        sqlite_storage = SqliteImageRecordStorage(db=db)
        sqlmodel_storage = SqlModelImageRecordStorage(db=db)

        t_sqlite = _time_it(lambda: self._save_images(sqlite_storage, "sqlite", self.N_IMAGES))
        t_sqlmodel = _time_it(lambda: self._save_images(sqlmodel_storage, "sqlmodel", self.N_IMAGES))
        _report("INSERT images", t_sqlite, t_sqlmodel, self.N_IMAGES)

    def test_get_images(self, db: SqliteDatabase):
        sqlite_storage = SqliteImageRecordStorage(db=db)
        sqlmodel_storage = SqlModelImageRecordStorage(db=db)

        self._save_images(sqlite_storage, "img", self.N_IMAGES)

        names = [f"img_{i}" for i in range(self.N_IMAGES)]

        def sqlite_get():
            for name in names:
                sqlite_storage.get(name)

        def sqlmodel_get():
            for name in names:
                sqlmodel_storage.get(name)

        t_sqlite = _time_it(sqlite_get)
        t_sqlmodel = _time_it(sqlmodel_get)
        _report("GET images (by name)", t_sqlite, t_sqlmodel, self.N_IMAGES)

    def test_get_many_images(self, db: SqliteDatabase):
        sqlite_storage = SqliteImageRecordStorage(db=db)
        sqlmodel_storage = SqlModelImageRecordStorage(db=db)

        self._save_images(sqlite_storage, "img", self.N_IMAGES)

        def sqlite_query():
            sqlite_storage.get_many(
                offset=0,
                limit=20,
                starred_first=True,
                order_dir=SQLiteDirection.Descending,
                categories=[ImageCategory.GENERAL],
            )

        def sqlmodel_query():
            sqlmodel_storage.get_many(
                offset=0,
                limit=20,
                starred_first=True,
                order_dir=SQLiteDirection.Descending,
                categories=[ImageCategory.GENERAL],
            )

        t_sqlite = _time_it(sqlite_query, iterations=self.N_QUERIES)
        t_sqlmodel = _time_it(sqlmodel_query, iterations=self.N_QUERIES)
        _report("GET MANY images (paginated + filtered)", t_sqlite, t_sqlmodel, self.N_QUERIES)

    def test_get_intermediates_count(self, db: SqliteDatabase):
        sqlite_storage = SqliteImageRecordStorage(db=db)
        sqlmodel_storage = SqlModelImageRecordStorage(db=db)

        self._save_images(sqlite_storage, "img", self.N_IMAGES)

        t_sqlite = _time_it(lambda: sqlite_storage.get_intermediates_count(), iterations=self.N_QUERIES)
        t_sqlmodel = _time_it(lambda: sqlmodel_storage.get_intermediates_count(), iterations=self.N_QUERIES)
        _report("COUNT intermediates", t_sqlite, t_sqlmodel, self.N_QUERIES)

    def test_update_images(self, db: SqliteDatabase):
        sqlite_storage = SqliteImageRecordStorage(db=db)
        sqlmodel_storage = SqlModelImageRecordStorage(db=db)

        self._save_images(sqlite_storage, "img", self.N_IMAGES)
        names = [f"img_{i}" for i in range(self.N_IMAGES)]

        def sqlite_update():
            for name in names:
                sqlite_storage.update(name, ImageRecordChanges(starred=True))

        def sqlmodel_update():
            for name in names:
                sqlmodel_storage.update(name, ImageRecordChanges(starred=False))

        t_sqlite = _time_it(sqlite_update)
        t_sqlmodel = _time_it(sqlmodel_update)
        _report("UPDATE images (star)", t_sqlite, t_sqlmodel, self.N_IMAGES)


# ---------------------------------------------------------------------------
# Users Benchmark
# ---------------------------------------------------------------------------


class TestUsersBenchmark:
    """Compare user operations between old and new implementations."""

    N_USERS = 50

    def test_create_users(self, db: SqliteDatabase):
        sqlite_service = UserService(db=db)
        sqlmodel_service = UserServiceSqlModel(db=db)

        def sqlite_create():
            for i in range(self.N_USERS):
                sqlite_service.create(
                    UserCreateRequest(
                        email=f"sqlite{i}@test.com", display_name=f"SQLite {i}", password="TestPassword123"
                    ),
                    strict_password_checking=False,
                )

        def sqlmodel_create():
            for i in range(self.N_USERS):
                sqlmodel_service.create(
                    UserCreateRequest(
                        email=f"sqlmodel{i}@test.com", display_name=f"SQLModel {i}", password="TestPassword123"
                    ),
                    strict_password_checking=False,
                )

        t_sqlite = _time_it(sqlite_create)
        t_sqlmodel = _time_it(sqlmodel_create)
        _report("CREATE users", t_sqlite, t_sqlmodel, self.N_USERS)

    def test_list_users(self, db: SqliteDatabase):
        sqlite_service = UserService(db=db)
        sqlmodel_service = UserServiceSqlModel(db=db)

        for i in range(self.N_USERS):
            sqlite_service.create(
                UserCreateRequest(email=f"user{i}@test.com", display_name=f"User {i}", password="TestPassword123"),
                strict_password_checking=False,
            )

        t_sqlite = _time_it(lambda: sqlite_service.list_users(), iterations=100)
        t_sqlmodel = _time_it(lambda: sqlmodel_service.list_users(), iterations=100)
        _report("LIST users", t_sqlite, t_sqlmodel, 100)

    def test_authenticate_users(self, db: SqliteDatabase):
        sqlite_service = UserService(db=db)
        sqlmodel_service = UserServiceSqlModel(db=db)

        for i in range(10):
            sqlite_service.create(
                UserCreateRequest(email=f"auth{i}@test.com", display_name=f"Auth {i}", password="TestPassword123"),
                strict_password_checking=False,
            )

        def sqlite_auth():
            for i in range(10):
                sqlite_service.authenticate(f"auth{i}@test.com", "TestPassword123")

        def sqlmodel_auth():
            for i in range(10):
                sqlmodel_service.authenticate(f"auth{i}@test.com", "TestPassword123")

        t_sqlite = _time_it(sqlite_auth, iterations=10)
        t_sqlmodel = _time_it(sqlmodel_auth, iterations=10)
        _report("AUTHENTICATE users", t_sqlite, t_sqlmodel, 100)
