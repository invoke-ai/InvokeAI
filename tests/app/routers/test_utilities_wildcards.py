import asyncio
import os

from invokeai.app.api.routers.utilities import parse_dynamicprompts
from invokeai.app.util.wildcards import (
    find_missing_wildcard_references,
    get_wildcard_references,
    get_wildcard_values,
    index_wildcards,
)


class _MockConfiguration:
    def __init__(self, root_path):
        self.root_path = root_path


class _MockServices:
    def __init__(self, root_path):
        self.configuration = _MockConfiguration(root_path)


class _MockInvoker:
    def __init__(self, root_path):
        self.services = _MockServices(root_path)


class _MockApiDependencies:
    def __init__(self, root_path):
        self.invoker = _MockInvoker(root_path)


def test_index_wildcards_supports_txt_json_and_nested_yaml(tmp_path):
    wildcards_dir = tmp_path / "wildcards"
    (wildcards_dir / "camera").mkdir(parents=True)
    (wildcards_dir / "camera" / "lens.txt").write_text(
        "# comment\n\n\ufeff50mm\n\u00ef\u00bb\u00bf85mm\n", encoding="utf-8"
    )
    (wildcards_dir / "styles.json").write_text(
        '{"lighting": ["soft light", "rim light"], "ignored": 123}', encoding="utf-8"
    )
    (wildcards_dir / "packs.yaml").write_text(
        "portrait:\n  mood:\n    - calm\n    - intense\n  ignored: 123\n", encoding="utf-8"
    )

    result = index_wildcards(wildcards_dir)

    assert result.errors == []
    indexed = {wildcard.path: wildcard for wildcard in result.wildcards}
    assert indexed["camera/lens"].token == "__camera/lens__"
    assert indexed["camera/lens"].value_count == 2
    assert indexed["camera/lens"].samples == ["50mm", "85mm"]
    assert indexed["lighting"].file_type == "json"
    assert indexed["portrait/mood"].file_type == "yaml"


def test_get_wildcard_values_resolves_txt_json_and_nested_yaml(tmp_path):
    wildcards_dir = tmp_path / "wildcards"
    (wildcards_dir / "camera").mkdir(parents=True)
    (wildcards_dir / "camera" / "lens.txt").write_text("# comment\n35mm\n50mm\n85mm\n", encoding="utf-8")
    (wildcards_dir / "styles.json").write_text('{"lighting": ["soft light", "rim light"]}', encoding="utf-8")
    (wildcards_dir / "packs.yaml").write_text("portrait:\n  mood:\n    - calm\n    - intense\n", encoding="utf-8")

    txt_values = get_wildcard_values(wildcards_dir, "camera/lens")
    json_values = get_wildcard_values(wildcards_dir, "lighting")
    yaml_values = get_wildcard_values(wildcards_dir, "portrait/mood")

    assert txt_values is not None
    assert txt_values.values == ["35mm", "50mm", "85mm"]
    assert txt_values.path == "camera/lens"
    assert json_values is not None
    assert json_values.values == ["soft light", "rim light"]
    assert yaml_values is not None
    assert yaml_values.values == ["calm", "intense"]


def test_get_wildcard_values_skips_malformed_structured_files_for_unrelated_wildcards(tmp_path):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "bad.json").write_text('{"broken": [', encoding="utf-8")
    (wildcards_dir / "valid.txt").write_text("one\ntwo\n", encoding="utf-8")

    values = get_wildcard_values(wildcards_dir, "valid")

    assert values is not None
    assert values.values == ["one", "two"]


def test_get_wildcard_values_rejects_traversal_and_unknown_paths(tmp_path):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "safe.txt").write_text("safe\n", encoding="utf-8")

    assert get_wildcard_values(wildcards_dir, "../safe") is None
    assert get_wildcard_values(wildcards_dir, "/safe") is None
    assert get_wildcard_values(wildcards_dir, "missing") is None


def test_get_wildcard_values_respects_limit(tmp_path):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "many.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

    values = get_wildcard_values(wildcards_dir, "many", limit=2)

    assert values is not None
    assert values.values == ["one", "two"]
    assert values.value_count == 3
    assert values.truncated is True


def test_index_wildcards_reports_invalid_structured_files(tmp_path):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "bad.yaml").write_text("portrait: [unterminated", encoding="utf-8")
    (wildcards_dir / "bad.json").write_text('{"portrait": [', encoding="utf-8")
    (wildcards_dir / "good.txt").write_text("usable\n", encoding="utf-8")

    result = index_wildcards(wildcards_dir)

    assert [wildcard.path for wildcard in result.wildcards] == ["good"]
    assert {error.path for error in result.errors} == {"bad.json", "bad.yaml"}


def test_index_wildcards_reports_duplicate_structured_paths(tmp_path):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "a.json").write_text('{"k": ["x"]}', encoding="utf-8")
    (wildcards_dir / "b.json").write_text('{"k": ["y"]}', encoding="utf-8")

    result = index_wildcards(wildcards_dir)

    assert [wildcard.path for wildcard in result.wildcards] == ["k"]
    assert len(result.errors) == 1
    assert result.errors[0].path == "b.json"
    assert "Duplicate wildcard path 'k'" in result.errors[0].message


def test_index_wildcards_blocks_symlinks_outside_root(tmp_path):
    wildcards_dir = tmp_path / "wildcards"
    outside_dir = tmp_path / "outside"
    wildcards_dir.mkdir()
    outside_dir.mkdir()
    outside_file = outside_dir / "outside.txt"
    outside_file.write_text("outside\n", encoding="utf-8")
    link = wildcards_dir / "outside.txt"

    try:
        os.symlink(outside_file, link)
    except (OSError, NotImplementedError):
        return

    result = index_wildcards(wildcards_dir)

    assert result.wildcards == []
    assert len(result.errors) == 1
    assert result.errors[0].message in {
        "Wildcard file is outside the wildcards folder",
        "Symlinks are not supported",
    }


def test_get_wildcard_references_normalizes_sampler_and_template_args():
    refs = get_wildcard_references("a __~camera/lens__ and __@lighting*__ with __style(name=test)__")

    assert refs == ["camera/lens", "lighting*", "style"]


def test_find_missing_wildcard_references_supports_globs(tmp_path):
    wildcards_dir = tmp_path / "wildcards"
    (wildcards_dir / "camera").mkdir(parents=True)
    (wildcards_dir / "camera" / "lens.txt").write_text("50mm\n", encoding="utf-8")
    indexed = index_wildcards(wildcards_dir).wildcards

    missing = find_missing_wildcard_references("a __camera/*__ and __missing__", indexed)

    assert missing == ["missing"]


def test_parse_dynamicprompts_random_mode_returns_one_prompt(tmp_path, monkeypatch):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "colors.txt").write_text("red\nblue\n", encoding="utf-8")
    monkeypatch.setattr("invokeai.app.api.routers.utilities.ApiDependencies", _MockApiDependencies(tmp_path))

    result = asyncio.run(parse_dynamicprompts(None, prompt="__colors__", max_prompts=1, combinatorial=False, seed=1))

    assert len(result.prompts) == 1
    assert result.prompts[0] in {"red", "blue"}


def test_parse_dynamicprompts_combinatorial_mode_respects_max_prompts(tmp_path, monkeypatch):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "colors.txt").write_text("\ufeffred\nblue\ngreen\n", encoding="utf-8")
    monkeypatch.setattr("invokeai.app.api.routers.utilities.ApiDependencies", _MockApiDependencies(tmp_path))

    result = asyncio.run(parse_dynamicprompts(None, prompt="__colors__", max_prompts=2, combinatorial=True))

    assert len(result.prompts) == 2
    assert set(result.prompts).issubset({"red", "blue", "green"})
