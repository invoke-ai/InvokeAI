import json
import textwrap
from pathlib import Path

import pytest

from tests import sharding

pytest_plugins = ["pytester"]

FILES = [f"test_{name}.py" for name in "abcdefg"]


@pytest.fixture(autouse=True)
def _plugin_copy(request: pytest.FixtureRequest) -> None:
    # The plugin loads like it does in the repo, through the conftest's `pytest_plugins`, so that xdist workers (fresh
    # interpreters, where the repo's `tests` package is not importable) find it next to the conftest.
    if "pytester" in request.fixturenames:
        pytester = request.getfixturevalue("pytester")
        (pytester.path / "shardplugin.py").write_text(Path(sharding.__file__).read_text(encoding="utf-8"))
        _extend_conftest(pytester, 'pytest_plugins = ["shardplugin"]\n')


def _extend_conftest(pytester: pytest.Pytester, source: str) -> None:
    with (pytester.path / "conftest.py").open("a", encoding="utf-8") as conftest:
        conftest.write(textwrap.dedent(source))


def _write_suite(pytester: pytest.Pytester, files: list[str] = FILES, tests_per_file: int = 3) -> None:
    pytester.makeini("[pytest]\nmarkers =\n    slow: slow\n")
    for name in files:
        body = "".join(f"def test_{i}(): pass\n" for i in range(tests_per_file))
        pytester.makepyfile(**{name.removesuffix(".py"): body})


def _durations(pytester: pytest.Pytester, files: dict[str, float]) -> str:
    path = pytester.path / "durations.json"
    path.write_text(json.dumps({"version": 1, "files": files}))
    return f"--shard-durations={path}"


def _shard_files(pytester: pytest.Pytester, shard: str, *args: str) -> set[str]:
    result = pytester.runpytest("--collect-only", "-q", f"--shard={shard}", *args)
    assert result.ret in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED), result.stderr.str()
    return {line.split("::")[0] for line in result.outlines if "::" in line}


@pytest.mark.parametrize("with_durations", [False, True])
def test_shards_partition_the_files(pytester: pytest.Pytester, with_durations: bool) -> None:
    _write_suite(pytester)
    args = [_durations(pytester, {"test_a.py": 9, "test_b.py": 1, "test_c.py": 4})] if with_durations else []
    shards = [_shard_files(pytester, f"{i}/3", *args) for i in (1, 2, 3)]
    assert sorted(f for shard in shards for f in shard) == FILES
    assert not any(shards[i] & shards[j] for i in range(3) for j in range(i))


def test_partition_ignores_collection_order(pytester: pytest.Pytester) -> None:
    _write_suite(pytester)
    durations = _durations(pytester, {"test_a.py": 9, "test_b.py": 1, "test_c.py": 4})
    forward = [_shard_files(pytester, f"{i}/3", durations) for i in (1, 2, 3)]
    _extend_conftest(pytester, "def pytest_collection_modifyitems(items):\n    items.reverse()\n")
    assert [_shard_files(pytester, f"{i}/3", durations) for i in (1, 2, 3)] == forward


def test_xdist_workers_agree_on_the_partition_without_durations(pytester: pytest.Pytester) -> None:
    # Each xdist worker is its own process with its own hash seed; xdist aborts if their collections differ.
    _write_suite(pytester, [f"test_{i}.py" for i in range(10)], tests_per_file=1)
    result = pytester.runpytest("-n", "2", "--shard=1/2")
    assert result.ret == pytest.ExitCode.OK, result.stdout.str()


def test_known_files_are_balanced_longest_first(pytester: pytest.Pytester) -> None:
    _write_suite(pytester, ["test_a.py", "test_b.py", "test_c.py", "test_d.py"])
    durations = _durations(pytester, {"test_a.py": 10, "test_b.py": 6, "test_c.py": 5, "test_d.py": 5})
    # a, then b, go to the empty shards; c joins the lighter b (11), d joins a (15) rather than b.
    assert _shard_files(pytester, "1/2", durations) == {"test_a.py", "test_d.py"}
    assert _shard_files(pytester, "2/2", durations) == {"test_b.py", "test_c.py"}


@pytest.mark.parametrize("recorded", ["none", "all zero"])
def test_files_without_a_positive_duration_spread_over_the_shards(pytester: pytest.Pytester, recorded: str) -> None:
    names = [f"test_{i}.py" for i in range(8)]
    _write_suite(pytester, names)
    # A 0 s record carries no balance: like an absent one it must not send every file to the lightest shard.
    args = [_durations(pytester, dict.fromkeys(names, 0))] if recorded == "all zero" else []
    shards = [_shard_files(pytester, f"{i}/3", *args) for i in (1, 2, 3)]
    assert sorted(f for shard in shards for f in shard) == sorted(names)
    assert sum(bool(shard) for shard in shards) > 1


def test_marker_deselected_files_do_not_weigh_on_the_balance(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\nmarkers =\n    slow: slow\n")
    pytester.makepyfile(test_slow="import pytest\npytestmark = pytest.mark.slow\ndef test_x(): pass\n")
    pytester.makepyfile(test_b="def test_x(): pass\n", test_c="def test_x(): pass\n")
    durations = _durations(pytester, {"test_slow.py": 100, "test_b.py": 1, "test_c.py": 1})
    # Counted, test_slow would take shard 1 alone and put both others on shard 2.
    first = _shard_files(pytester, "1/2", durations, "-m", "not slow")
    second = _shard_files(pytester, "2/2", durations, "-m", "not slow")
    assert (first, second) == ({"test_b.py"}, {"test_c.py"})


@pytest.mark.parametrize("spec", ["0/3", "4/3", "1/0", "2", "a/b", "1/2/3"])
def test_invalid_shard_is_a_usage_error(pytester: pytest.Pytester, spec: str) -> None:
    _write_suite(pytester)
    result = pytester.runpytest(f"--shard={spec}")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--shard must be I/N*"])


@pytest.mark.parametrize(
    ("content", "shard"),
    [
        ("not json", ["--shard=1/2"]),
        ('{"version": 2, "files": {}}', ["--shard=1/2"]),
        ('{"version": 1, "files": []}', ["--shard=1/2"]),
        ('{"version": 1, "files": {"a.py": true}}', ["--shard=1/2"]),
        ('{"version": 1, "files": {"a.py": NaN}}', ["--shard=1/2"]),
        ('{"version": 1, "files": {"a.py": -1}}', ["--shard=1/2"]),
        # The file is validated whenever it is given, not only for a sharded run.
        ('{"version": 2, "files": {}}', []),
    ],
)
def test_unusable_durations_fail_loudly(pytester: pytest.Pytester, content: str, shard: list[str]) -> None:
    _write_suite(pytester)
    path = pytester.path / "durations.json"
    path.write_text(content)
    result = pytester.runpytest(*shard, f"--shard-durations={path}")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--shard-durations*"])


def test_record_sums_setup_call_and_teardown_per_file_and_only_the_controller_writes(
    pytester: pytest.Pytester,
) -> None:
    # Each test reports 1 s setup + 2 s call + 4 s teardown, so a file's total tells which phases were summed.
    _extend_conftest(
        pytester,
        """
        import pytest
        import shardplugin

        _write = shardplugin._write_durations

        def _logged_write(path, files):
            with open(str(path) + ".writes", "a") as log:
                log.write("write\\n")
            _write(path, files)

        shardplugin._write_durations = _logged_write

        @pytest.hookimpl(wrapper=True)
        def pytest_runtest_makereport(item, call):
            report = yield
            report.duration = {"setup": 1.0, "call": 2.0, "teardown": 4.0}[call.when]
            return report
        """,
    )
    pytester.makepyfile(test_one="def test_0(): pass\n", test_two="def test_0(): pass\ndef test_1(): pass\n")
    record = pytester.path / "record.json"
    result = pytester.runpytest("-n", "2", f"--shard-record={record}")
    result.assert_outcomes(passed=3)
    assert json.loads(record.read_text()) == {"version": 1, "files": {"test_one.py": 7.0, "test_two.py": 14.0}}
    assert (pytester.path / "record.json.writes").read_text().count("write") == 1


def test_record_is_skipped_after_a_failing_session(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_bad="def test_0(): assert False\n")
    record = pytester.path / "record.json"
    result = pytester.runpytest(f"--shard-record={record}")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*shard recording skipped*"])
    assert not record.exists()


def test_main_updates_the_durations_file_from_recordings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("kept.py", "updated.py", "new.py"):
        (tmp_path / name).touch()
    target = tmp_path / "shard_durations.json"
    target.write_text(json.dumps({"version": 1, "files": {"kept.py": 1, "updated.py": 2, "gone.py": 3}}))
    recording = tmp_path / "recording.json"
    recording.write_text(json.dumps({"version": 1, "files": {"updated.py": 5.5, "new.py": 0.5}}))
    monkeypatch.setattr(sharding, "ROOT", tmp_path)
    monkeypatch.setattr(sharding, "DURATIONS_PATH", target)
    sharding.main([str(recording)])
    assert json.loads(target.read_text()) == {
        "version": 1,
        "files": {"kept.py": 1, "new.py": 0.5, "updated.py": 5.5},
    }
    assert "2 updated, 1 kept, 1 dropped" in capsys.readouterr().out


def test_main_rejects_an_unusable_recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sharding, "DURATIONS_PATH", tmp_path / "shard_durations.json")
    with pytest.raises(SystemExit, match="recording"):
        sharding.main([str(tmp_path / "missing.json")])
