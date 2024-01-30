import re
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from invokeai.app.util.profiler import Profiler


def test_profiler_new():
    with TemporaryDirectory() as tempdir:
        profiler = Profiler(logger=Logger("test_profiler"), output_dir=Path(tempdir))
        assert not profiler.profiler
        assert not profiler.profile_id
        profiler.new("test")
        assert profiler.profiler
        assert profiler.profile_id == "test"
        profiler.new("test2")
        assert profiler.profiler
        assert profiler.profile_id == "test2"


def test_profiler_profiles():
    with TemporaryDirectory() as tempdir:
        profiler = Profiler(logger=Logger("test_profiler"), output_dir=Path(tempdir))
        profiler.new("test")
        profiler.enable()
        for _ in range(1000000):
            pass
        profiler.disable()
        profiler.dump()
        assert (Path(tempdir) / "test.prof").exists()


def test_profile_fails_if_not_set_up():
    with TemporaryDirectory() as tempdir:
        profiler = Profiler(logger=Logger("test_profiler"), output_dir=Path(tempdir))
        match = re.escape("Profiler not initialized. Call Profiler.new() first.")
        with pytest.raises(RuntimeError, match=match):
            profiler.enable()
        with pytest.raises(RuntimeError, match=match):
            profiler.disable()
        with pytest.raises(RuntimeError, match=match):
            profiler.dump()
