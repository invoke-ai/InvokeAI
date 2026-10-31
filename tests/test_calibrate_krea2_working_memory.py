"""scripts/calibrate_krea2_working_memory.py: the forwards it plans, what it hands the transformer and the estimate for
them, and the point files it writes for the data builder.

The measurement needs a CUDA or ROCm device and runs by hand (see the script's docstring). These cover what decides the
points it writes: which forwards a run measures, that a measured forward is the one its features describe, and that a
run's points replace the legacy `denoise.krea2` entry of their platform when the builder reads them. The plumbing it
shares with the other calibration scripts is `scripts/working_memory/fresh_points.py`'s, tested in
`tests/test_fresh_points_working_memory.py`.
"""

import json
import shutil
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch
from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel

from invokeai.backend.krea2.denoise_working_memory import KREA2_DENOISE_SPEC
from invokeai.backend.krea2.reference_edit import Krea2ReferenceKVMode
from invokeai.backend.working_memory.coefficients import load_data
from invokeai.backend.working_memory.counters import CounterReading
from invokeai.backend.working_memory.measurement import fresh_window
from scripts.working_memory import fresh_points
from scripts.working_memory.build_shipped import DATA_DIR, main, write_data

SCRIPT = Path(__file__).parents[1] / "scripts" / "calibrate_krea2_working_memory.py"
MIB = 2**20


def _load_script():
    spec = spec_from_file_location("calibrate_krea2_working_memory", SCRIPT)
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
    "references": [0, 1, 3],
    "reference_px": 1024,
    "blocks": 2,
    "extra_blocks": 0,
}
TRANSFORMER = script._transformer(2)


def _features(point):
    return script._estimate(point, TRANSFORMER, torch.device("cpu")).features


def test_the_fit_grid_runs_every_size_the_short_caption_and_the_references_per_dtype() -> None:
    points = script.plan("fit", dtypes=["bfloat16", "float32"], **(GRID | {"extra_blocks": 4}))

    per_dtype = [
        ("fit", 512, 512, 0, 2),
        ("fit", 1024, 512, 0, 2),
        ("fit", 2048, 512, 0, 2),
        ("fit", 1024, 77, 0, 2),
        ("fit", 1024, 512, 1, 2),
        ("fit", 1024, 512, 3, 2),
    ]
    assert [(p.dtype, p.kind, p.px, p.text_tokens, p.references, p.blocks) for p in points] == [
        *((dtype, *row) for dtype in ("bfloat16", "float32") for row in per_dtype),
        ("bfloat16", "blocks", 1024, 512, 0, 4),
    ]
    assert [point.fit for point in points] == [True] * 12 + [False]


def test_a_plain_or_joint_reference_forward_fits_its_dtypes_line_over_every_token() -> None:
    """Three 1024px joint references at a 1024px generation: 4096 tokens each beside the image's 4096 and the caption."""
    point = Point("fit", 1024, 512, "float16", references=3)
    features = _features(point)
    line = KREA2_DENOISE_SPEC.line(features)

    assert (features["image_tokens"], features["reference_tokens"], features["text_tokens"]) == (4096, 3 * 4096, 512)
    assert (line.suffix, line.size, line.fits) == ("denoise_fp16", 4096 + 3 * 4096 + 512, True)


def test_the_large_grid_reaches_29184_tokens_on_the_line() -> None:
    points = script.plan("large", dtypes=["bfloat16"], **GRID)
    lines = [KREA2_DENOISE_SPEC.line(_features(point)) for point in points]

    assert [line.size for line in lines] == [16384 + 4096 + 512, 16384 + 3 * 4096 + 512, 9216 + 3 * 4096 + 512]
    assert all(point.fit for point in points) and all(line.fits for line in lines)


def test_the_caption_checks_run_long_captions_per_dtype_off_the_line() -> None:
    points = script.plan(
        "captions", dtypes=["bfloat16", "float32"], **GRID, long_captions=[1024, 2048], long_caption_sizes=[1024]
    )
    features = [_features(point) for point in points]

    assert [(p.kind, p.dtype, p.px, p.text_tokens, p.fit) for p in points] == [
        ("caption", dtype, 1024, caption, False) for dtype in ("bfloat16", "float32") for caption in (1024, 2048)
    ]
    assert [f["tokens"] for f in features] == [4096 + 1024, 4096 + 2048] * 2
    assert not any(KREA2_DENOISE_SPEC.line(f).fits for f in features)


def test_the_kv_checks_price_the_longer_pass_and_the_cache_off_the_line() -> None:
    """At 1024px three cached references outnumber the step's tokens, one does not; neither fits the line."""
    points = script.plan("kv", dtypes=["bfloat16"], **GRID)
    features = [_features(point) for point in points]

    assert [(p.px, p.references, p.cache_references, p.fit) for p in points] == [
        (1024, 1, True, False),
        (1024, 3, True, False),
        (2048, 1, True, False),
    ]
    assert [f["tokens"] for f in features] == [4096 + 512, 3 * 4096, 16384 + 512]
    assert all(f["reference_kv_bytes"] > 0 and f["reference_tokens"] == 0 for f in features)
    assert not any(KREA2_DENOISE_SPEC.line(f).fits for f in features)


def _tiny_transformer() -> torch.nn.Module:
    """The class the script measures, at one head of 8 and a narrow caption, so the CPU runs it in float32."""
    torch.manual_seed(0)
    return Krea2Transformer2DModel(
        in_channels=script.IN_CHANNELS,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        timestep_embed_dim=8,
        text_hidden_dim=8,
        num_text_layers=12,
        text_num_attention_heads=2,
        text_num_key_value_heads=2,
        text_intermediate_size=16,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(2, 2, 4),
    ).eval()


@pytest.mark.parametrize(
    "point",
    [
        Point("fit", 64, 7, "float32"),
        Point("fit", 64, 7, "float32", references=2, reference_px=32),
        Point("kv", 64, 7, "float32", references=2, reference_px=32),
    ],
    ids=["plain", "joint-references", "kv-cache"],
)
def test_the_measured_operation_runs_the_tokens_its_features_name(point) -> None:
    """The node's inputs through the node's processors: the image's 16 tokens and the caption's 7, the references'
    4 each on the joint sequence or in the cache, which the operation fills for its step and empties after it."""
    features = _features(point)
    model = _tiny_transformer()
    cache = script.install_processors(model, point)
    operation = script.forward(point, model, cache, torch.device("cpu"))
    appended: list[int] = []
    if cache is not None:
        extend = cache.extend

        def recording_extend(block_index, key, value, attention_mask):
            appended.append(cache._cache[block_index][0].shape[2])
            return extend(block_index, key, value, attention_mask)

        cache.extend = recording_extend

    with torch.inference_mode():
        result = operation()
    # The stock forward returns a tuple, `forward_with_references` the tensor.
    output = result[0] if isinstance(result, tuple) else result

    assert (features["image_tokens"], features["text_tokens"]) == (16, 7)
    assert output.shape == (1, 16, script.IN_CHANNELS)
    if point.cache_references:
        assert appended == [features["cached_reference_tokens"]] * 2 == [8, 8]
        assert (len(cache), cache.mode) == (0, Krea2ReferenceKVMode.OFF)
    else:
        assert features["reference_tokens"] == 4 * point.references


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


SLOPES = {"bfloat16": 150_000, "float16": 150_000, "float32": 260_000}
"""Bytes per attended token the synthetic points below lie on, per dtype, over a 300 MiB base."""
SUFFIXES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


def _record(point) -> dict:
    """The file record of `point` as a measuring process on an RTX 4090 writes it, its demand on its dtype's synthetic
    line."""
    features = _features(point)
    demand = SLOPES[point.dtype] * features["tokens"] + 300 * MIB
    window = fresh_window(
        _reading(), _reading(peak_reserved=1000 + demand, driver=1200 + demand), oom=False, spill=False
    )
    common = {
        "estimator": KREA2_DENOISE_SPEC.id,
        "version": KREA2_DENOISE_SPEC.version,
        "platform": {"conv_backend": "cudnn", "allocator": "native", "os": "windows"},
        "features": dict(features),
        "device": "NVIDIA GeForce RTX 4090",
    }
    line = fresh_points.point_line(point, common=common, provenance="test", windows=[window, window], raw={})
    record = fresh_points.file_record(line)
    assert record is not None
    return record


LEGACY_VALUES = {
    f"{part}_denoise_{dtype}": value for dtype in SUFFIXES.values() for part, value in (("k", 2**19), ("b", 3 * 2**29))
}
"""The hand-set constants of a `legacy` Krea-2 denoise entry."""
CUDNN_NATIVE = {"conv_backend": "cudnn", "allocator": "native"}


def _legacy_data_copy(tmp_path: Path) -> Path:
    """The shipped data as it was before the RTX 4090's native-allocator Krea-2 points shipped: that entry hand-set and
    `legacy` again, without its points."""
    data_dir = tmp_path / "data"
    shutil.copytree(DATA_DIR, data_dir)
    data = load_data(data_dir)["denoise.json"]

    def reverted(entry) -> bool:
        return entry.estimator == KREA2_DENOISE_SPEC.id and entry.platform == CUDNN_NATIVE

    entries = [
        entry.model_copy(update={"values": LEGACY_VALUES, "legacy": True}) if reverted(entry) else entry
        for entry in data.coefficients
    ]
    points = [point for point in data.points if not reverted(point)]
    write_data(data_dir / "denoise.json", data.model_copy(update={"coefficients": entries, "points": points}))
    return data_dir


def test_a_runs_points_replace_the_legacy_entry_of_their_platform(tmp_path: Path, capsys) -> None:
    """The default fit grid spans each dtype's line, joint references included, so the builder fits all three lines
    and the RTX 4090's native-allocator entry is legacy no more. Its floors for the platforms it prices unmeasured stay,
    and so does every other entry. The K/V-cache checks join as out-of-fit points their estimate covers."""
    grid = GRID | {"sizes": [512, 768, 1024, 1536, 2048], "short_text_sizes": [1024, 1536], "references": [1, 2, 3]}
    points = [
        *script.plan("fit", dtypes=list(script.DTYPES), **(grid | {"extra_blocks": 4})),
        *script.plan("large", dtypes=["bfloat16"], **grid),
        *script.plan("kv", dtypes=["bfloat16"], **grid),
    ]
    point_file = tmp_path / "krea2.jsonl"
    point_file.write_text("".join(json.dumps(_record(point)) + "\n" for point in points), encoding="utf-8")
    data_dir = _legacy_data_copy(tmp_path)

    assert main(["--points", str(point_file), "--data-dir", str(data_dir), "--write"]) == 0

    assert f"{len(points)} fresh-process points from 1 point files" in capsys.readouterr().out

    def entries(directory: Path) -> dict:
        return {
            ",".join(entry.platform.values()): entry
            for entry in load_data(directory)["denoise.json"].coefficients
            if entry.estimator == KREA2_DENOISE_SPEC.id
        }

    fitted, shipped = entries(data_dir), entries(DATA_DIR)
    native, shipped_native = fitted.pop("cudnn,native"), shipped.pop("cudnn,native")
    assert not native.legacy
    # The synthetic lines plus the shipped fit's 5%, the base rounded up to a whole MiB.
    assert native.values == {
        name: value
        for dtype, slope in SLOPES.items()
        for name, value in (
            (f"k_denoise_{SUFFIXES[dtype]}", slope * 21 // 20),
            (f"b_denoise_{SUFFIXES[dtype]}", 315 * MIB),
        )
    }
    assert native.unmeasured_floor == shipped_native.unmeasured_floor != {}
    assert fitted == shipped
