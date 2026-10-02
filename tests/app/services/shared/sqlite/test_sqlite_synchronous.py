"""The `synchronous` PRAGMA, and the config field that sets it.

`full` fsyncs on every commit and is what InvokeAI has always run with; `normal` trades the most
recent transactions on a power loss for much shorter commits, and is only safe from corruption
under WAL. The default must not move, because that would change the durability of every existing
install without anyone asking for it.
"""

import logging
import sqlite3
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.backend.util.logging import InvokeAILogger

# What `PRAGMA synchronous` reports back, as documented by SQLite.
PRAGMA_VALUES = {"off": 0, "normal": 1, "full": 2, "extra": 3}


def _synchronous_of(db: SqliteDatabase) -> int:
    return int(db._conn.execute("PRAGMA synchronous;").fetchone()[0])


def _journal_mode_of(db: SqliteDatabase) -> str:
    return str(db._conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower()


def _warnings_about_durability(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Warnings this setting is responsible for, so an unrelated one from elsewhere cannot decide a test."""
    messages = [record.getMessage().lower() for record in caplog.records if record.levelno >= logging.WARNING]
    return [message for message in messages if "journal" in message or "synchronous" in message]


@pytest.fixture
def sqlite_cannot_use_wal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes WAL genuinely unavailable, the way a network filesystem or an `SQLITE_OMIT_WAL` build does.

    SQLite refuses to move a transient database into WAL mode, so the connection opened here answers
    `PRAGMA journal_mode = WAL;` with a rollback-journal mode instead -- a real refusal from real
    SQLite. The interception is at the `sqlite3.connect` boundary; everything `SqliteDatabase` does
    after it is the production path.
    """
    real_connect = sqlite3.connect

    def connect_to_a_database_that_refuses_wal(database: object, **kwargs: object) -> sqlite3.Connection:
        return real_connect("", **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sqlite3, "connect", connect_to_a_database_that_refuses_wal)


class TestTheConfigField:
    def test_the_default_is_full(self):
        # Anything else would silently reduce durability for every existing install on upgrade.
        assert InvokeAIAppConfig().db_synchronous == "full"

    def test_normal_is_accepted(self):
        assert InvokeAIAppConfig(db_synchronous="normal").db_synchronous == "normal"

    @pytest.mark.parametrize("value", ["off", "extra", "NORMAL", "", "1"])
    def test_other_sqlite_values_are_rejected(self, value):
        """`off` and `extra` are real SQLite settings, deliberately not offered: `off` can corrupt
        the database on an OS crash, and `extra` costs more than `full` for a guarantee this
        application does not need."""
        with pytest.raises(ValidationError):
            InvokeAIAppConfig(db_synchronous=value)


class TestTheConstructorValidatesTheSetting:
    """`SqliteDatabase` is a public constructor -- `user_management.py` and tests call it directly --
    and the value ends up interpolated into a PRAGMA statement, which takes no bind parameters. The
    `Literal` annotation is checked by a type checker, not at runtime, so the constructor checks it."""

    @pytest.mark.parametrize("value", ["normal; DROP TABLE x", "off", "NORMAL", "", "1"])
    def test_a_value_outside_the_literal_never_reaches_sql(self, tmp_path, value):
        db_path = tmp_path / "rejected.db"

        with pytest.raises(ValueError, match="synchronous"):
            SqliteDatabase(db_path=db_path, logger=InvokeAILogger.get_logger(), synchronous=value)

        # Rejected before anything was opened, so there is nothing for the statement to have run against.
        assert not db_path.exists()


class TestThePragmaIsApplied:
    def test_the_default_setting_uses_full(self, tmp_path):
        db = SqliteDatabase(db_path=tmp_path / "default.db", logger=InvokeAILogger.get_logger())
        assert _synchronous_of(db) == PRAGMA_VALUES["full"]

    @pytest.mark.parametrize("setting", ["full", "normal"])
    def test_each_setting_reaches_the_connection(self, tmp_path, setting):
        db = SqliteDatabase(
            db_path=tmp_path / f"{setting}.db",
            logger=InvokeAILogger.get_logger(),
            synchronous=setting,
        )
        assert _synchronous_of(db) == PRAGMA_VALUES[setting]

    def test_an_in_memory_database_keeps_normal_despite_having_no_journal(self):
        """An in-memory database reports journal mode `memory`, never `wal`, but it has no durability
        to trade away in the first place -- so the WAL requirement below must not reach it. The
        migrator and several tests run on this path."""
        db = SqliteDatabase(db_path=None, logger=InvokeAILogger.get_logger(), synchronous="normal")

        assert _journal_mode_of(db) != "wal"
        assert _synchronous_of(db) == PRAGMA_VALUES["normal"]

    def test_wal_is_still_on(self, tmp_path):
        """`normal` is only safe against corruption *because* of WAL. If journal mode ever stopped
        being WAL for an ordinary database file, this setting would become a different trade than the
        one documented."""
        db = SqliteDatabase(db_path=tmp_path / "wal.db", logger=InvokeAILogger.get_logger(), synchronous="normal")
        assert _journal_mode_of(db) == "wal"

    def test_normal_is_refused_when_wal_did_not_engage(self, tmp_path, caplog, sqlite_cannot_use_wal):
        """WAL needs shared memory, which NFS/SMB shares, some container volume drivers and
        `SQLITE_OMIT_WAL` builds do not provide; `PRAGMA journal_mode = WAL` then quietly leaves a
        rollback journal in place. `normal` on a rollback journal *can* corrupt the database on power
        loss, which is the opposite of what the setting is documented to do, so it is refused."""
        logger = InvokeAILogger.get_logger()

        with caplog.at_level(logging.WARNING):
            db = SqliteDatabase(db_path=tmp_path / "no-wal.db", logger=logger, synchronous="normal")

        journal_mode = _journal_mode_of(db)
        assert journal_mode != "wal"
        assert _synchronous_of(db) == PRAGMA_VALUES["full"]
        assert any(journal_mode in message and "normal" in message for message in _warnings_about_durability(caplog))

    def test_full_is_applied_without_a_warning_when_wal_did_not_engage(self, tmp_path, caplog, sqlite_cannot_use_wal):
        # Nothing to refuse: `full` is already the stricter setting, so a warning would be noise.
        logger = InvokeAILogger.get_logger()

        with caplog.at_level(logging.WARNING):
            db = SqliteDatabase(db_path=tmp_path / "no-wal-full.db", logger=logger)

        assert _synchronous_of(db) == PRAGMA_VALUES["full"]
        assert _warnings_about_durability(caplog) == []


class TestTheSettingReachesTheDatabase:
    """The gap that a config field and a working PRAGMA still leave open.

    Both halves can be correct while nothing connects them -- and that failure is silent: the app
    boots, every other test passes, and the setting simply does nothing. This happened once during
    development, which is why it is pinned rather than assumed.

    Driven through the real `init_db` against an in-memory database, migrations and all, so what is
    asserted is the connection the app ends up with. Stubbing `SqliteDatabase` here and checking the
    keyword arrived would pin the argument's *name* instead: an implementation that accepts it and
    then applies it conditionally would still pass, which is the regression worth catching.
    """

    @pytest.mark.parametrize("setting", ["full", "normal"])
    def test_init_db_carries_the_configured_value_to_the_connection(self, setting):
        db = init_db(
            config=InvokeAIAppConfig(use_memory_db=True, db_synchronous=setting),
            logger=InvokeAILogger.get_logger(),
            image_files=Mock(),
        )

        assert _synchronous_of(db) == PRAGMA_VALUES[setting]
