"""scripts/calibrate_flux2_denoise_working_memory.py: the forwards it plans, what it hands the transformer and the
estimate for them, and the point files it writes for the data builder.

The measurement needs a CUDA or ROCm device and runs by hand (see the script's docstring). These cover what decides the
points it writes: which forwards a run measures, that a measured forward is the one its features describe, and that a
run's points replace the legacy `denoise.flux2` entry of their platform when the builder reads them. The plumbing it
shares with the other calibration scripts is `scripts/working_memory/fresh_points.py`'s, tested in
`tests/test_fresh_points_working_memory.py`.
"""

import json
import shutil
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch
from diffusers import Flux2Transformer2DModel

from invokeai.backend.flux2.denoise_working_memory import FLUX2_DENOISE_SPEC
from invokeai.backend.model_manager.configs.flux2_variant import flux2_hidden_size, flux2_variant_from_context_dim
from invokeai.backend.working_memory.coefficients import load_data
from invokeai.backend.working_memory.counters import CounterReading
from invokeai.backend.working_memory.measurement import fresh_window
from scripts.working_memory import fresh_points
from scripts.working_memory.build_shipped import DATA_DIR, main, write_data

SCRIPT = Path(__file__).parents[1] / "scripts" / "calibrate_flux2_denoise_working_memory.py"
MIB = 2**20


def _load_script():
    spec = spec_from_file_location("calibrate_flux2_denoise_working_memory", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()
Point = script.Point

GRID = {
    "sizes": [512, 1024, 2048],
    "text_tokens": 512,
    "short_text_tokens": 77,
    "short_text_sizes": [1024],
    "references": [1, 3],
    "batches": [1, 2],
    "blocks": 2,
    "extra_blocks": 0,
}


def test_the_fit_grid_runs_every_size_and_at_1024px_the_short_caption_the_references_and_the_batch_per_width() -> None:
    points = script.plan("fit", widths=[3072, 6144], **(GRID | {"extra_blocks": 4}))

    per_width = [
        ("fit", 512, 512, 0, 1, 2),
        ("fit", 1024, 512, 0, 1, 2),
        ("fit", 2048, 512, 0, 1, 2),
        ("fit", 1024, 77, 0, 1, 2),
        ("fit", 1024, 512, 1, 1, 2),
        ("fit", 1024, 512, 3, 1, 2),
        ("fit", 1024, 512, 0, 2, 2),
        ("blocks", 1024, 512, 0, 1, 4),
    ]
    assert [(p.hidden_size, p.kind, p.px, p.text_tokens, p.references, p.batch, p.blocks) for p in points] == [
        (width, *row) for width in (3072, 6144) for row in per_width
    ]
    assert [point.fit for point in points] == [kind == "fit" for _ in (3072, 6144) for kind, *_ in per_width]
    assert not any(point.force_math for point in points)


def test_the_math_check_pairs_a_forced_math_forward_with_a_default_kernel_one_per_width_out_of_the_fit() -> None:
    points = script.plan("math", widths=[4096, 6144], **GRID)

    assert points == [
        Point("baseline", 4096, 1024, 512),
        Point("math", 4096, 1024, 512, force_math=True),
        Point("baseline", 6144, 1024, 512),
        Point("math", 6144, 1024, 512, force_math=True),
    ]
    assert not any(point.fit for point in points)


@pytest.mark.parametrize("width", sorted(script.CONTEXT_DIMS))
def test_each_widths_caption_width_is_the_model_managers(width: int) -> None:
    assert flux2_hidden_size(flux2_variant_from_context_dim(script.CONTEXT_DIMS[width])) == width


@pytest.mark.parametrize("width", sorted(script.CONTEXT_DIMS))
def test_the_measured_transformer_has_the_geometry_the_loader_builds_at_its_width(width: int) -> None:
    """The loader sizes a FLUX.2 transformer from its checkpoint's `context_embedder` (hidden size by caption width),
    with heads of 128 and as many as the hidden size holds (`model_loaders/flux.py`). A model measured at another inner
    width would put its points on the wrong width's line."""
    model = script._transformer(width, 1)

    assert model.context_embedder.weight.shape == (width, script.CONTEXT_DIMS[width])
    assert model.inner_dim == width
    assert (model.config.attention_head_dim, model.config.num_attention_heads) == (128, width // 128)


def test_a_point_is_priced_on_its_widths_line_over_the_tokens_of_every_sample() -> None:
    """Three 1024px references at a 1024px generation, two samples: the references' tokens are those
    `Flux2RefImageExtension` concatenates, repeated per sample, and the forward fits its width's line."""
    point = Point("fit", 4096, 1024, 512, references=3, batch=2)

    features = script._estimate(point, torch.device("cpu")).features
    line = FLUX2_DENOISE_SPEC.line(features)

    assert (features["image_tokens"], features["reference_tokens"], features["text_tokens"]) == (4096, 12288, 512)
    assert (features["batch_size"], features["hidden_size"]) == (2, 4096)
    assert (line.suffix, line.size, line.fits) == ("denoise_bf16_h4096", 2 * (4096 + 12288 + 512), True)


@pytest.mark.parametrize(
    "references, reference_px, tokens",
    [(1, 1024, 4096), (1, 2024, 16129), (1, 1023, 64 * 64), (3, 1024, 3 * 4096), (3, 1328, 3 * 4096)],
    ids=["one-1024", "one-2024", "odd-side-padded", "three-1024", "three-1328-resized"],
)
def test_reference_tokens_are_what_the_extension_concatenates(references, reference_px, tokens) -> None:
    """One reference keeps up to 2024x2024 pixels, each of several up to 1024x1024 (`Flux2RefImageExtension`), and a
    latent side is padded to whole 2x2 patches: three 1328px references cost what three 1024px ones do."""
    point = Point("fit", 4096, 1024, 512, references=references, reference_px=reference_px)

    assert script.reference_tokens(point) == tokens


def test_the_large_grid_reaches_33025_tokens_and_fits_its_widths_line() -> None:
    """2048px with one 2024px reference, the largest single reference the node keeps. Each forward is a fit point on its
    width's line."""
    points = script.plan("large", widths=[3072, 4096], **GRID)

    assert [(p.hidden_size, p.px, p.references, p.reference_px) for p in points] == [
        (width, px, n, ref_px)
        for width in (3072, 4096)
        for px, n, ref_px in ((2048, 1, 2024), (1536, 1, 2024), (1328, 3, 1328))
    ]
    lines = [FLUX2_DENOISE_SPEC.line(script._estimate(point, torch.device("cpu")).features) for point in points]
    assert [line.size for line in lines] == [33025, 25857, 19689] * 2
    assert all(point.fit for point in points) and all(line.fits for line in lines)


def test_the_measured_forward_takes_the_tokens_its_features_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The node's inputs, run through the transformer class the loader builds, at one head and in float32 so the CPU
    runs it: the image's and the references' tokens on one stream with their position ids, and the caption."""
    point = Point("fit", 3072, 64, 7, references=1, reference_px=32, batch=2)
    features = script._estimate(point, torch.device("cpu")).features
    monkeypatch.setattr(script, "DTYPE", "float32")
    model = Flux2Transformer2DModel(
        in_channels=script.LATENT_CHANNELS,
        out_channels=script.LATENT_CHANNELS,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=script.HEAD_DIM,
        num_attention_heads=1,
        joint_attention_dim=script.CONTEXT_DIMS[3072],
        patch_size=1,
    ).eval()

    inputs = script._inputs(point, torch.device("cpu"))
    with torch.inference_mode():
        (output,) = model(**inputs)

    image_and_references = features["image_tokens"] + features["reference_tokens"]
    assert (image_and_references, features["text_tokens"]) == (16 + 4, 7)
    assert output.shape == (2, image_and_references, script.LATENT_CHANNELS)
    assert inputs["encoder_hidden_states"].shape[:2] == (2, 7)


def _reading(**values) -> CounterReading:
    fields = {
        "reserved": 1000,
        "allocated": 800,
        "peak_reserved": 1000,
        "peak_allocated": 800,
        "num_ooms": 0,
        "num_alloc_retries": 0,
        "driver": 1200,
        "peak_driver": None,
        "paged": 0,
        "device_used": None,
        "method": "windows-d3dkmt",
    }
    return CounterReading(**(fields | values))


SLOPES = {3072: 300_000, 4096: 400_000, 6144: 600_000}
"""Bytes per attended token the synthetic points below lie on, per width, over a 300 MiB base."""


def _record(point) -> dict:
    """The file record of `point` as a measuring process on an RTX 4090 writes it, its demand on its width's synthetic
    line."""
    features = script._estimate(point, torch.device("cpu")).features
    demand = SLOPES[point.hidden_size] * features["tokens"] * features["batch_size"] + 300 * MIB
    window = fresh_window(
        _reading(), _reading(peak_reserved=1000 + demand, driver=1200 + demand), oom=False, spill=False
    )
    common = {
        "estimator": FLUX2_DENOISE_SPEC.id,
        "version": FLUX2_DENOISE_SPEC.version,
        "platform": {"conv_backend": "cudnn", "allocator": "native", "os": "windows"},
        "features": dict(features),
        "device": "NVIDIA GeForce RTX 4090",
    }
    line = fresh_points.point_line(point, common=common, provenance="test", windows=[window, window], raw={})
    record = fresh_points.file_record(line)
    assert record is not None
    return record


LEGACY_VALUES = {
    name: value
    for width, k in ((3072, 330300), (4096, 440401), (6144, 660601))
    for name, value in ((f"k_denoise_bf16_h{width}", k), (f"b_denoise_bf16_h{width}", 2**30))
}
"""The hand-set constants of a `legacy` FLUX.2 denoise entry."""
CUDNN_NATIVE = {"conv_backend": "cudnn", "allocator": "native"}


def _legacy_data_copy(tmp_path: Path) -> Path:
    """The shipped data as it was before the RTX 4090's native-allocator FLUX.2 points shipped: that entry hand-set and
    `legacy` again, without its points."""
    data_dir = tmp_path / "data"
    shutil.copytree(DATA_DIR, data_dir)
    data = load_data(data_dir)["denoise.json"]

    def reverted(entry) -> bool:
        return entry.estimator == FLUX2_DENOISE_SPEC.id and entry.platform == CUDNN_NATIVE

    entries = [
        entry.model_copy(update={"values": LEGACY_VALUES, "legacy": True}) if reverted(entry) else entry
        for entry in data.coefficients
    ]
    points = [point for point in data.points if not reverted(point)]
    write_data(data_dir / "denoise.json", data.model_copy(update={"coefficients": entries, "points": points}))
    return data_dir


def test_a_runs_points_replace_the_legacy_entry_of_their_platform(tmp_path: Path, capsys) -> None:
    """The default grid's fit points span each width's line, reference images and the batch included, so the builder
    fits all three lines and the RTX 4090's native-allocator entry is legacy no more. Its floors for the platforms it
    prices unmeasured stay, and so does every other entry."""
    widths = sorted(SLOPES)
    points = script.plan(
        "fit",
        widths=widths,
        sizes=[512, 768, 1024, 1440, 2048],
        text_tokens=512,
        short_text_tokens=77,
        short_text_sizes=[1024, 1440],
        references=[1, 3],
        batches=[2],
        blocks=2,
        extra_blocks=4,
    )
    records = [_record(point) for point in points]
    point_file = tmp_path / "flux2.jsonl"
    point_file.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    data_dir = _legacy_data_copy(tmp_path)

    assert main(["--points", str(point_file), "--data-dir", str(data_dir), "--write"]) == 0

    assert f"{len(points)} fresh-process points from 1 point files" in capsys.readouterr().out
    entries = {
        ",".join(entry.platform.values()): entry
        for entry in load_data(data_dir)["denoise.json"].coefficients
        if entry.estimator == FLUX2_DENOISE_SPEC.id
    }
    shipped = {
        ",".join(entry.platform.values()): entry
        for entry in load_data(DATA_DIR)["denoise.json"].coefficients
        if entry.estimator == FLUX2_DENOISE_SPEC.id
    }
    fitted, shipped_native = entries.pop("cudnn,native"), shipped.pop("cudnn,native")
    assert not fitted.legacy
    # The synthetic lines plus the shipped fit's 5%, the base rounded up to a whole MiB.
    assert fitted.values == {
        name: value
        for width, slope in SLOPES.items()
        for name, value in ((f"k_denoise_bf16_h{width}", slope * 21 // 20), (f"b_denoise_bf16_h{width}", 315 * MIB))
    }
    assert fitted.unmeasured_floor == shipped_native.unmeasured_floor != {}
    assert entries == shipped
