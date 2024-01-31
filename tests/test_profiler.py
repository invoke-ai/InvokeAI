import re
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from invokeai.app.util.profiler import Profiler


def test_profiler_starts():
    with TemporaryDirectory() as tempdir:
        profiler = Profiler(logger=Logger("test_profiler"), output_dir=Path(tempdir))
        assert not profiler._profiler
        assert not profiler.profile_id
        profiler.start("test")
        assert profiler._profiler
        assert profiler.profile_id == "test"
        profiler.stop()
        assert not profiler._profiler
        assert not profiler.profile_id
        profiler.start("test2")
        assert profiler._profiler
        assert profiler.profile_id == "test2"
        profiler.stop()


def test_profiler_profiles():
    with TemporaryDirectory() as tempdir:
        profiler = Profiler(logger=Logger("test_profiler"), output_dir=Path(tempdir))
        profiler.start("test")
        for _ in range(1000000):
            pass
        profiler.stop()
        assert (Path(tempdir) / "test.prof").exists()


def test_profiler_profiles_with_prefix():
    with TemporaryDirectory() as tempdir:
        profiler = Profiler(logger=Logger("test_profiler"), output_dir=Path(tempdir), prefix="prefix")
        profiler.start("test")
        for _ in range(1000000):
            pass
        profiler.stop()
        assert (Path(tempdir) / "prefix_test.prof").exists()


def test_profile_fails_if_not_set_up():
    with TemporaryDirectory() as tempdir:
        profiler = Profiler(logger=Logger("test_profiler"), output_dir=Path(tempdir))
        match = re.escape("Profiler not initialized. Call start() first.")
        with pytest.raises(RuntimeError, match=match):
            profiler.stop()
