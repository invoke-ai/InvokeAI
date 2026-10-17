from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.family_e2e.manifest import ManifestError, load_manifest, plan_steps, resolve_cases

EXAMPLE = Path(__file__).parents[2] / "scripts" / "family_e2e" / "manifest.example.yaml"


def _case(case_id: str, **fields: Any) -> dict[str, Any]:
    return {"id": case_id, "model": "m", "steps": 4, "sizes": [[64, 64]], **fields}


def _write(tmp_path: Path, families: list[dict[str, Any]], inputs: tuple[str, ...] = (), **top: Any) -> Path:
    for name in inputs:
        (tmp_path / "data" / "inputs" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "inputs" / name).write_bytes(b"png")
    manifest = {
        "profile": "box",
        "data_dir": "data",
        "gpu": "RTX",
        "server": {"template_root": "template", "python": "venv/python.exe"},
        "defaults": {"prompt": "a fox", "seed": 7, "warmup": 1, "repeats": 2},
        "families": families,
        **top,
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return path


def test_relative_paths_resolve_against_the_manifest_directory(tmp_path: Path) -> None:
    manifest = load_manifest(_write(tmp_path, [{"family": "f", "cases": [_case("a")]}]))
    assert manifest.data_dir == tmp_path / "data"
    assert manifest.server.template_root == tmp_path / "template"
    assert manifest.server.python == tmp_path / "venv" / "python.exe"


def test_plan_follows_family_case_and_size_order_with_warmups_first(tmp_path: Path) -> None:
    families = [
        {"family": "second", "cases": [_case("b", sizes=[[128, 64], [64, 128]]), _case("c")]},
        {"family": "first", "cases": [_case("a")]},
    ]
    manifest = load_manifest(_write(tmp_path, families))
    steps = plan_steps(resolve_cases(manifest), manifest.defaults.warmup, manifest.defaults.repeats)
    assert [(s.family, s.case_key, s.kind, s.rep) for s in steps] == [
        ("second", "b@128x64", "warmup", 0),
        ("second", "b@128x64", "measure", 0),
        ("second", "b@128x64", "measure", 1),
        ("second", "b@64x128", "warmup", 0),
        ("second", "b@64x128", "measure", 0),
        ("second", "b@64x128", "measure", 1),
        ("second", "c@64x64", "warmup", 0),
        ("second", "c@64x64", "measure", 0),
        ("second", "c@64x64", "measure", 1),
        ("first", "a@64x64", "warmup", 0),
        ("first", "a@64x64", "measure", 0),
        ("first", "a@64x64", "measure", 1),
    ]
    assert {s.arm for s in steps} == {"default"}


def test_filters_select_by_family_and_case_and_reject_unknown_names(tmp_path: Path) -> None:
    families = [{"family": "x", "cases": [_case("a"), _case("b")]}, {"family": "y", "cases": [_case("c")]}]
    manifest = load_manifest(_write(tmp_path, families))
    assert [c.case_key for c in resolve_cases(manifest, families=["y"])] == ["c@64x64"]
    assert [c.case_key for c in resolve_cases(manifest, case_ids=["b", "c"])] == ["b@64x64", "c@64x64"]
    with pytest.raises(ManifestError, match="no case"):
        resolve_cases(manifest, families=["x"], case_ids=["c"])
    with pytest.raises(ManifestError, match="unknown family or case: z"):
        resolve_cases(manifest, families=["z"])


def test_case_values_fall_back_to_the_defaults(tmp_path: Path) -> None:
    families = [{"family": "f", "cases": [_case("a"), _case("b", seed=99, prompt="a cat", psnr_min_db=40)]}]
    a, b = resolve_cases(load_manifest(_write(tmp_path, families)))
    assert (a.seed, a.prompt, a.thresholds.psnr_min_db, a.thresholds.peak_tolerance_mb) == (7, "a fox", 30.0, 256.0)
    assert (b.seed, b.prompt, b.thresholds.psnr_min_db, b.thresholds.time_tolerance) == (99, "a cat", 40.0, 0.10)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        (_case("a", mode="img2img"), "init_image is required"),
        (_case("a", init_image={"path": "i.png", "strength": 0.5}), "init_image is required"),
        (_case("a", mode="controlnet"), "control is required"),
        (_case("a", control=[{"model": "cn", "image": "i.png"}]), "control is required"),
        (_case("a", sizes=[[64, 60]]), "multiples of 8"),
        (_case("a", sizes=[[64, 64], [64, 64]]), "unique"),
        (_case("a.b"), "id"),
        (_case("a", unknown=1), "unknown"),
    ],
)
def test_invalid_cases_are_rejected(tmp_path: Path, case: dict[str, Any], message: str) -> None:
    with pytest.raises(ManifestError, match=message):
        load_manifest(_write(tmp_path, [{"family": "f", "cases": [case]}], inputs=("i.png",)))


def test_case_ids_must_be_unique_across_families(tmp_path: Path) -> None:
    families = [{"family": "x", "cases": [_case("a")]}, {"family": "y", "cases": [_case("a")]}]
    with pytest.raises(ManifestError, match="unique across the manifest: a"):
        load_manifest(_write(tmp_path, families))


def test_inputs_must_exist_under_the_inputs_directory(tmp_path: Path) -> None:
    case = _case("a", mode="img2img", init_image={"path": "missing.png", "strength": 0.5})
    with pytest.raises(ManifestError, match="a: missing.png"):
        load_manifest(_write(tmp_path, [{"family": "f", "cases": [case]}]))

    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "outside.png").write_bytes(b"png")
    escaping = _case("a", reference_images=["../outside.png"])
    with pytest.raises(ManifestError, match="outside"):
        load_manifest(_write(tmp_path, [{"family": "f", "cases": [escaping]}]))

    nested = _case("a", reference_images=["refs/fox.png"])
    manifest = load_manifest(_write(tmp_path, [{"family": "f", "cases": [nested]}], inputs=("refs/fox.png",)))
    assert manifest.input_file("refs/fox.png") == (tmp_path / "data" / "inputs" / "refs" / "fox.png").resolve()


def test_example_manifest_is_valid(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["data_dir"] = str(tmp_path)
    for family in raw["families"]:
        for case in family["cases"]:
            for name in case.get("reference_images", []) + [c["image"] for c in case.get("control", [])]:
                (tmp_path / "inputs").mkdir(exist_ok=True)
                (tmp_path / "inputs" / name).write_bytes(b"png")
            if case.get("init_image"):
                (tmp_path / "inputs").mkdir(exist_ok=True)
                (tmp_path / "inputs" / case["init_image"]["path"]).write_bytes(b"png")
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(raw))
    manifest = load_manifest(path)
    cases = resolve_cases(manifest)
    assert {c.family for c in cases} == {"z-image", "flux2", "krea2", "sdxl"}
    assert {c.spec.mode for c in cases} == {"txt2img", "img2img", "controlnet"}
    assert any(len(c.spec.sizes) > 1 for c in cases)
