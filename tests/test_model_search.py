from pathlib import Path

import pytest

from invokeai.backend.model_manager.search import ModelSearch, ModelSearchCancelled


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


def test_model_search_handles_modular_diffusers_model_dirs(model_search: tuple[ModelSearch, Path]):
    """A Modular Diffusers folder (e.g. MiniMax H3) has modular_model_index.json at its root and
    config.json files inside its submodel folders. The walker must stop at the root instead of
    descending and offering each submodel folder as a separate model."""
    search, tmp_path = model_search

    modular_dir = tmp_path / "modular_dir"
    modular_dir.mkdir()
    with open(modular_dir / "modular_model_index.json", "w") as f:
        f.write("")
    submodel_dir = modular_dir / "transformer"
    submodel_dir.mkdir()
    with open(submodel_dir / "config.json", "w") as f:
        f.write("")

    found = search.search(tmp_path)

    assert found == {modular_dir}


def test_model_search_stops_when_asked(tmp_path: Path):
    """A stop request abandons the walk at the next directory instead of finishing the tree."""
    visited: list[Path] = []
    for index in range(4):
        folder = tmp_path / f"folder{index}"
        folder.mkdir()
        (folder / "model.safetensors").write_text("")

    def stop_after_first_model() -> bool:
        return len(visited) > 0

    search = ModelSearch(should_stop=stop_after_first_model, on_model_found=lambda path: visited.append(path) or True)

    with pytest.raises(ModelSearchCancelled):
        search.search(tmp_path)

    assert len(visited) == 1


def test_model_search_skips_a_symlinked_directory(model_search: tuple[ModelSearch, Path]):
    """A folder that is symlinked next to its target must not be walked a second time.

    Stability Matrix and similar launchers point several apps at one model folder, so the same
    model would otherwise show up twice in the scan results.
    """
    search, tmp_path = model_search

    model_dir = tmp_path / "models" / "sd"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "model.safetensors"
    model_file.write_text("")
    (tmp_path / "models" / "sd_link").symlink_to(model_dir, target_is_directory=True)

    found = search.search(tmp_path / "models")

    assert len(found) == 1
    assert {path.resolve() for path in found} == {model_file.resolve()}


def test_model_search_finds_a_model_behind_symlinks(model_search: tuple[ModelSearch, Path]):
    """Following a link is still allowed - only walking the same physical directory twice is not."""
    search, tmp_path = model_search

    model_dir = tmp_path / "real" / "sd"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "model.safetensors"
    model_file.write_text("")
    scan_root = tmp_path / "models"
    scan_root.mkdir()
    (scan_root / "linked").symlink_to(model_dir, target_is_directory=True)

    found = search.search(scan_root)

    assert len(found) == 1
    assert next(iter(found)).resolve() == model_file.resolve()


def test_model_search_survives_a_symlink_cycle(model_search: tuple[ModelSearch, Path]):
    """Two folders linking to each other must not produce the same model over and over."""
    search, tmp_path = model_search

    folder_a = tmp_path / "a"
    folder_b = tmp_path / "b"
    folder_a.mkdir()
    folder_b.mkdir()
    (folder_a / "model.ckpt").write_text("")
    (folder_a / "to_b").symlink_to(folder_b, target_is_directory=True)
    (folder_b / "to_a").symlink_to(folder_a, target_is_directory=True)

    found = search.search(tmp_path)

    assert len(found) == 1
    assert next(iter(found)).name == "model.ckpt"
