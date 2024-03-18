from pathlib import Path

import pytest

from invokeai.backend.model_manager.search import ModelSearch


@pytest.fixture
def model_search(tmp_path: Path) -> tuple[ModelSearch, Path]:
    search = ModelSearch()
    return search, tmp_path


def test_model_search_on_search_started(model_search: tuple[ModelSearch, Path]):
    search, tmp_path = model_search
    on_search_started_called_with: Path | None = None

    def on_search_started_callback(path: Path) -> None:
        nonlocal on_search_started_called_with
        on_search_started_called_with = path

    search.on_search_started = on_search_started_callback
    search.search(tmp_path)

    assert on_search_started_called_with == tmp_path


def test_model_search_on_completed(model_search: tuple[ModelSearch, Path]):
    search, tmp_path = model_search
    on_search_completed_called_with: set[Path] | None = None
    file1 = tmp_path / "file1.ckpt"
    with open(file1, "w") as f:
        f.write("")

    def on_search_completed_callback(models: set[Path]) -> None:
        nonlocal on_search_completed_called_with
        on_search_completed_called_with = models

    search.on_search_completed = on_search_completed_callback
    expected = {file1}
    found = search.search(tmp_path)

    assert found == expected
    assert on_search_completed_called_with == expected


def test_model_search_handles_files(model_search: tuple[ModelSearch, Path]):
    search, tmp_path = model_search
    on_model_found_called_with: set[Path] = set()

    file1 = tmp_path / "file1.ckpt"
    file2 = tmp_path / "file2.ckpt"
    file3 = tmp_path / "subfolder" / "file3.ckpt"
    file4 = tmp_path / "subfolder" / "subfolder" / "file4.ckpt"
    file5 = tmp_path / "not_a_model_file.txt"

    file4.parent.mkdir(parents=True)
    for file in [file1, file2, file3, file4, file5]:
        with open(file, "w") as f:
            f.write("")

    def on_model_found_callback(path: Path) -> bool:
        on_model_found_called_with.add(path)
        return True

    search.on_model_found = on_model_found_callback

    expected = {file1, file2, file3, file4}
    found = search.search(tmp_path)

    assert on_model_found_called_with == expected
    assert found == expected
    assert search.stats.models_found == 4
    assert search.stats.models_filtered == 4


def test_model_search_filters_by_on_model_found(model_search: tuple[ModelSearch, Path]):
    search, tmp_path = model_search
    on_model_found_called_with: set[Path] = set()

    file1 = tmp_path / "file1.ckpt"
    file2 = tmp_path / "file2.ckpt"  # explicitly ignored

    for file in [file1, file2]:
        with open(file, "w") as f:
            f.write("")

    def on_model_found_callback(path: Path) -> bool:
        if path == file2:
            return False
        on_model_found_called_with.add(path)
        return True

    search.on_model_found = on_model_found_callback

    expected = {file1}
    found = search.search(tmp_path)

    assert on_model_found_called_with == expected
    assert found == expected
    assert search.stats.models_filtered == 1
    assert search.stats.models_found == 2


def test_model_search_handles_diffusers_model_dirs(model_search: tuple[ModelSearch, Path]):
    search, tmp_path = model_search
    on_model_found_called_with: set[Path] = set()

    diffusers_dir = tmp_path / "diffusers_dir"
    diffusers_dir_entry_point = diffusers_dir / "model_index.json"
    diffusers_dir.mkdir()
    with open(diffusers_dir_entry_point, "w") as f:
        f.write("")

    nested_diffusers_dir = tmp_path / "subfolder" / "nested_diffusers_dir"
    nested_diffusers_dir_entry_point = nested_diffusers_dir / "model_index.json"
    nested_diffusers_dir_ignore_me_file = nested_diffusers_dir / "ignore_me.ckpt"  # totally skipped
    nested_diffusers_dir.mkdir(parents=True)
    with open(nested_diffusers_dir_entry_point, "w") as f:
        f.write("")
    with open(nested_diffusers_dir_ignore_me_file, "w") as f:
        f.write("")

    not_a_diffusers_dir = tmp_path / "not_a_diffusers_dir"
    not_a_diffusers_dir_entry_point = not_a_diffusers_dir / "not_model_index.json"
    not_a_diffusers_dir.mkdir()
    with open(not_a_diffusers_dir_entry_point, "w") as f:
        f.write("")

    def on_model_found_callback(path: Path) -> bool:
        on_model_found_called_with.add(path)
        return True

    search.on_model_found = on_model_found_callback

    expected = {diffusers_dir, nested_diffusers_dir}
    found = search.search(tmp_path)

    assert found == expected
    assert on_model_found_called_with == expected
    assert search.stats.models_found == 2
    assert search.stats.models_filtered == 2
