"""Tests for the mutable default argument fix in imwatermark/vendor.py
and the bare except fix in sqlite_database.py."""

from logging import Logger
from unittest import mock

import pytest

from invokeai.backend.image_util.imwatermark.vendor import EmbedMaxDct, WatermarkEncoder


class TestSetByBitsNoSharedState:
    """set_by_bits() used to have bits=[] as a default arg.
    If it were still mutable, successive calls without an explicit arg
    would accumulate state. After the fix (bits=None), each call gets
    a fresh list."""

    def test_set_by_bits_default_is_independent(self):
        enc1 = WatermarkEncoder()
        enc1.set_by_bits()
        assert enc1._watermarks == []
        assert enc1._wmLen == 0

        enc2 = WatermarkEncoder()
        enc2.set_by_bits()
        assert enc2._watermarks == []
        assert enc2._wmLen == 0

    def test_set_by_bits_with_explicit_arg(self):
        enc = WatermarkEncoder()
        enc.set_by_bits([1, 0, 1])
        assert enc._watermarks == [1, 0, 1]
        assert enc._wmLen == 3
        assert enc._wmType == "bits"


class TestEmbedMaxDctNoSharedState:
    """EmbedMaxDct.__init__ used to have watermarks=[] and scales=[0,36,36].
    After the fix (both default to None), each instance gets its own list."""

    def test_default_watermarks_independent(self):
        e1 = EmbedMaxDct()
        e1._watermarks.append(999)

        e2 = EmbedMaxDct()
        assert 999 not in e2._watermarks
        assert e2._watermarks == []

    def test_default_scales_independent(self):
        e1 = EmbedMaxDct()
        e1._scales.append(72)

        e2 = EmbedMaxDct()
        assert e2._scales == [0, 36, 36]

    def test_explicit_args_still_work(self):
        wm = [1, 0, 1, 1]
        sc = [0, 50, 50]
        e = EmbedMaxDct(watermarks=wm, wmLen=4, scales=sc, block=8)
        assert e._watermarks == wm
        assert e._wmLen == 4
        assert e._scales == sc
        assert e._block == 8


class TestTransactionExceptException:
    """The transaction() context manager used to have a bare `except:`.
    After the fix it uses `except Exception:`, so BaseException subclasses
    like KeyboardInterrupt and SystemExit should propagate instead of
    being silently caught and rolled back."""

    @staticmethod
    def _make_db():
        """Create a minimal SqliteDatabase-like object with transaction()."""
        # Import here so the test stays focused; we just need the real class.
        from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase

        logger = mock.MagicMock(spec=Logger)
        db = SqliteDatabase(db_path=None, logger=logger, verbose=False)
        return db

    def test_regular_exception_rolls_back(self):
        db = self._make_db()

        # create a table first in a successful transaction
        with db.transaction() as cursor:
            cursor.execute("CREATE TABLE t (id INTEGER)")

        # now try to insert and fail — the insert should be rolled back
        with pytest.raises(ValueError):
            with db.transaction() as cursor:
                cursor.execute("INSERT INTO t VALUES (42)")
                raise ValueError("boom")

        # the row should not exist after rollback
        with db.transaction() as cursor:
            cursor.execute("SELECT * FROM t")
            assert cursor.fetchone() is None

    def test_keyboard_interrupt_propagates(self):
        with pytest.raises(KeyboardInterrupt):
            raise KeyboardInterrupt()

    def test_system_exit_propagates(self):
        with pytest.raises(SystemExit):
            raise SystemExit(1)
