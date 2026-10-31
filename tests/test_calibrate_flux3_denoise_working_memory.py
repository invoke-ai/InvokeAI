"""scripts/calibrate_flux3_denoise_working_memory.py: the forwards it plans, what it hands the transformer and the
estimate for them, and the point files it writes for the data builder.

The measurement needs a CUDA or ROCm device and runs by hand (see the script's docstring). These cover what decides the
points it writes: which forwards a run measures, that a measured forward is the one its features describe, that the real
model's forwards load through the loader's conversion and price as the shipped 2-block points of their shape, and that a
run's points replace the legacy `denoise.flux3` entry of their platform when the builder reads them. The plumbing it
shares with the other calibration scripts is `scripts/working_memory/fresh_points.py`'s, tested in
`tests/test_fresh_points_working_memory.py`.
"""

import dataclasses
import json
import shutil
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch
from diffusers import Flux3Transformer2DModel
from safetensors.torch import save_file

from invokeai.backend.flux3.conversion import FLUX3_NATIVE_FLOW_MAP_KEY, flux3_transformer_config
from invokeai.backend.flux3.denoise_working_memory import FLUX3_DENOISE_SPEC
from invokeai.backend.working_memory.coefficients import load_data, shipped_coefficients
from invokeai.backend.working_memory.counters import CounterReading
from invokeai.backend.working_memory.measurement import fresh_window
from invokeai.backend.working_memory.platform import Platform
from scripts.working_memory import fresh_points
from scripts.working_memory.build_shipped import DATA_DIR, main, write_data
from tests.backend.flux3.flux3_fixture import forward as tiny_forward
from tests.backend.flux3.flux3_fixture import tiny_flux3_model, to_native
from tests.backend.model_manager.load.state_dicts.flux3_klein_keys import state_dict_keys as KLEIN_KEYS
from tests.backend.model_manager.load.state_dicts.utils import write_header_only_safetensors

SCRIPT = Path(__file__).parents[1] / "scripts" / "calibrate_flux3_denoise_working_memory.py"
MIB = 2**20


def _load_script():
    spec = spec_from_file_location("calibrate_flux3_denoise_working_memory", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()
Point = script.Point
CPU = torch.device("cpu")

GRID = {
    "sizes": [512, 1024, 2048],
    "text_tokens": 560,
    "short_text_tokens": 80,
    "short_text_sizes": [1024],
    "references": [1, 3, 10],
    "batches": [1, 2],
    "blocks": 2,
    "extra_blocks": 0,
}


def _features(point) -> dict:
    return dict(script._estimate(point, CPU, script._transformer(point)).features)


def test_the_fit_grid_runs_every_size_and_at_1024px_the_short_caption_references_batch_and_klein_base_per_dtype():
    """float32 leaves out the ten references, whose activations the legacy price would not keep off a 24 GB card."""
    points = script.plan("fit", dtypes=["bfloat16", "float32"], **(GRID | {"extra_blocks": 4}))

    def rows(dtype, references):
        return [
            (dtype, "fit", 512, 560, 0, 1, 2, True),
            (dtype, "fit", 1024, 560, 0, 1, 2, True),
            (dtype, "fit", 2048, 560, 0, 1, 2, True),
            (dtype, "fit", 1024, 80, 0, 1, 2, True),
            *[(dtype, "fit", 1024, 560, n, 1, 2, True) for n in references],
            (dtype, "fit", 1024, 560, 0, 2, 2, True),
            (dtype, "fit", 1024, 560, 0, 1, 2, False),
        ]

    assert [(p.dtype, p.kind, p.px, p.text_tokens, p.references, p.batch, p.blocks, p.flow_map) for p in points] == [
        *rows("bfloat16", [1, 3, 10]),
        *rows("float32", [1, 3]),
        ("bfloat16", "blocks", 1024, 560, 0, 1, 4, True),
    ]
    assert [point.fit for point in points] == [point.kind == "fit" for point in points]
    assert not any(point.force_math for point in points)


def test_the_math_check_pairs_a_forced_math_forward_with_a_default_kernel_one_per_dtype_out_of_the_fit():
    points = script.plan("math", dtypes=["bfloat16", "float16"], **GRID)

    assert points == [
        Point("baseline", "bfloat16", 1024, 560),
        Point("math", "bfloat16", 1024, 560, force_math=True),
        Point("baseline", "float16", 1024, 560),
        Point("math", "float16", 1024, 560, force_math=True),
    ]
    assert not any(point.fit for point in points)


def test_the_measured_transformer_has_the_geometry_the_loader_reads_off_the_released_checkpoint():
    """`flux3_transformer_config` reads the loader's geometry off the checkpoint's shapes; a model measured at another
    width or caption width would put its points on the wrong line."""

    class _Shape:
        def __init__(self, shape):
            self.shape = tuple(shape)

    released = flux3_transformer_config({key: _Shape(shape) for key, (shape, _) in KLEIN_KEYS.items()})

    for key, value in script.GEOMETRY.items():
        assert released[key] == value, key
    assert set(released) - set(script.GEOMETRY) == {"num_layers", "num_single_layers", "flow_map"}


@pytest.mark.parametrize("flow_map", [True, False], ids=["klein", "klein-base"])
def test_a_point_builds_its_variant_in_its_dtype_with_the_qk_norms_in_float32(flow_map):
    model = script._transformer(Point("fit", "float16", 1024, 560, blocks=2, flow_map=flow_map))

    assert model.config.flow_map is flow_map
    assert (model.config.num_layers, model.config.num_single_layers) == (2, 2)
    dtypes = {name: param.dtype for name, param in model.named_parameters()}
    assert {dtype for name, dtype in dtypes.items() if name.endswith((".norm_q.weight", ".norm_k.weight"))} == {
        torch.float32
    }
    assert {dtype for name, dtype in dtypes.items() if not name.endswith((".norm_q.weight", ".norm_k.weight"))} == {
        torch.float16
    }


def test_a_point_is_priced_on_its_dtypes_line_over_the_tokens_of_every_sample():
    """Three 1024px references at a 1024px generation, two samples, Klein Base."""
    features = _features(Point("fit", "float16", 1024, 560, references=3, batch=2, flow_map=False))
    line = FLUX3_DENOISE_SPEC.line(features)

    assert (features["image_tokens"], features["reference_tokens"], features["text_tokens"]) == (4096, 12288, 560)
    assert (features["batch_size"], features["hidden_size"], features["flow_map"]) == (2, 3072, False)
    assert (line.suffix, line.size, line.fits) == ("denoise_fp16_h3072", 2 * (4096 + 12288 + 560), True)


@pytest.mark.parametrize(
    "references, reference_px, tokens",
    [(1, 1024, 4096), (1, 2048, 16384), (1, 4096, 16384), (1, 1000, 62 * 62), (3, 1024, 3 * 4096), (10, 1024, 40960)],
    ids=["one-1024", "one-2048", "one-4096-to-4mp", "odd-side-cropped", "three-1024", "ten-1024"],
)
def test_reference_tokens_are_what_the_node_encodes(references, reference_px, tokens):
    """One reference keeps up to 4 MP, each of several up to 1 MP (`flux3/ref_images.py`), and a side is cropped to
    whole 16 px cells."""
    point = Point("fit", "bfloat16", 1024, 560, references=references, reference_px=reference_px)

    assert script.reference_tokens(point) == tokens


def test_the_large_grid_reaches_57904_tokens_and_fits_its_line():
    points = script.plan("large", dtypes=["bfloat16"], **GRID)

    assert [(p.px, p.references, p.reference_px) for p in points] == [
        (2048, 1, 2048),
        (1536, 1, 2048),
        (2048, 3, 1024),
        (2048, 10, 1024),
    ]
    lines = [FLUX3_DENOISE_SPEC.line(_features(point)) for point in points]
    assert [line.size for line in lines] == [33328, 26160, 29232, 57904]
    assert all(point.fit for point in points) and all(line.fits for line in lines)


@pytest.mark.parametrize("flow_map", [True, False], ids=["klein", "klein-base"])
def test_the_measured_forward_takes_the_tokens_its_features_name(flow_map: bool) -> None:
    """The node's inputs, run through the transformer class the loader builds, at one head and in float32 so the CPU
    runs it: the image's tokens, a reference on its own stream with its position ids, the caption, and for Klein the
    timestep each step jumps to."""
    point = Point("fit", "float32", 64, 80, references=1, reference_px=32, batch=2, flow_map=flow_map)
    features = _features(point)
    model = Flux3Transformer2DModel(
        **(dict(script.GEOMETRY) | {"num_attention_heads": 1}), num_layers=1, num_single_layers=1, flow_map=flow_map
    ).eval()

    inputs = script._inputs(point, CPU)
    with torch.inference_mode():
        (output,) = model(**inputs)

    # A reference below 64 px a side is scaled up to it: 4 x 4 tokens.
    assert (features["image_tokens"], features["reference_tokens"], features["text_tokens"]) == (16, 16, 80)
    assert inputs["image_cond_hidden_states"].shape[:2] == (2, 16) and inputs["image_cond_ids"].shape == (16, 4)
    assert inputs["encoder_hidden_states"].shape[:2] == (2, 80)
    assert output.shape == (2, 16, script.GEOMETRY["in_channels"])


@pytest.mark.parametrize("flow_map", [True, False], ids=["klein", "klein-base"])
def test_the_real_grid_measures_the_checkpoints_variant_out_of_the_fit(tmp_path: Path, flow_map: bool) -> None:
    """The variant is read off the checkpoint's keys, without torch: only Klein has the flow-map embedder."""
    keys = {key: value for key, value in KLEIN_KEYS.items() if flow_map or not key.startswith("ref_time_in.")}
    path = write_header_only_safetensors(tmp_path / "flux3.safetensors", keys)

    points = script.plan("real", dtypes=["float16"], **GRID, checkpoint=str(path))

    assert [(p.kind, p.dtype, p.px, p.text_tokens, p.references, p.flow_map, p.fit) for p in points] == [
        ("real", "bfloat16", 1024, 560, 0, flow_map, False),
        ("real", "bfloat16", 2048, 560, 0, flow_map, False),
        ("real", "bfloat16", 1024, 560, 1, flow_map, False),
        ("real", "bfloat16", 1024, 560, 3, flow_map, False),
    ]
    assert script.FLOW_MAP_KEY == FLUX3_NATIVE_FLOW_MAP_KEY


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--dtypes", "float16"], "takes no --dtypes"),
        (["--sizes", "1024", "--batches", "3"], "takes no --sizes, --batches"),
        (["--references", "3", "--no-base", "--extra-blocks", "4"], "takes no --references, --no-base, --extra-blocks"),
    ],
    ids=["dtypes", "sizes-batches", "references-base-blocks"],
)
def test_real_mode_refuses_the_options_it_would_ignore(tmp_path: Path, capsys, arguments, message) -> None:
    path = write_header_only_safetensors(tmp_path / "klein.safetensors", KLEIN_KEYS)

    with pytest.raises(SystemExit) as exit_:
        script.main(["--mode", "real", "--checkpoint", str(path), "--dry-run", *arguments])

    assert exit_.value.code == 2 and message in capsys.readouterr().err


def test_real_mode_refuses_a_checkpoint_that_is_not_a_bf16_safetensors_file(tmp_path: Path, capsys) -> None:
    gguf = tmp_path / "klein.gguf"
    gguf.write_bytes(b"GGUF")
    fp32 = write_header_only_safetensors(
        tmp_path / "klein-fp32.safetensors", {key: (shape, "F32") for key, (shape, _) in KLEIN_KEYS.items()}
    )

    for path, message in ((gguf, "is no safetensors single file"), (fp32, "holds F32 tensors")):
        with pytest.raises(SystemExit):
            script.main(["--mode", "real", "--checkpoint", str(path), "--dry-run"])
        assert message in capsys.readouterr().err


def test_a_real_forward_prices_the_features_of_the_2_block_point_of_its_shape(tmp_path: Path) -> None:
    """Real mode sets each forward against the shipped 2-block point of the same features: the geometry the loader
    reads off the released checkpoint's header has to give exactly those."""
    path = write_header_only_safetensors(tmp_path / "klein.safetensors", KLEIN_KEYS)
    real = Point("real", "bfloat16", 1024, 560, references=1, checkpoint=str(path))

    assert _features(real) == _features(Point("fit", "bfloat16", 1024, 560, references=1))


@pytest.mark.parametrize("flow_map", [True, False], ids=["klein", "klein-base"])
def test_a_real_checkpoint_loads_through_the_loaders_conversion(tmp_path: Path, flow_map: bool) -> None:
    """A tiny model in BFL's layout comes back as the model it was written from, in float32 so the CPU runs it, with
    the geometry the meta transformer read off its header."""
    reference = tiny_flux3_model(flow_map=flow_map)
    path = tmp_path / "tiny.safetensors"
    save_file(to_native(reference), str(path))
    point = Point("real", "float32", 64, 80, flow_map=flow_map, checkpoint=str(path))

    loaded = script._load_real(point, CPU)
    meta = script._transformer(point)

    assert {name: p.shape for name, p in meta.named_parameters()} == {
        name: p.shape for name, p in loaded.named_parameters()
    }
    assert torch.equal(tiny_forward(loaded, flow_map), tiny_forward(reference, flow_map))


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


SLOPES = {"bfloat16": 300_000, "float16": 300_000, "float32": 520_000}
"""Bytes per attended token the synthetic points below lie on, per dtype, over a 300 MiB base."""
LINES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


def _record(point, demand: int | None = None, platform: Platform | None = None) -> dict:
    """The file record of `point` as a measuring process on an RTX 4090 writes it, its demand on its dtype's synthetic
    line unless `demand` is given; on `platform` if given, as an RX 9060 XT then."""
    features = _features(point)
    if demand is None:
        demand = SLOPES[point.dtype] * features["tokens"] * features["batch_size"] + 300 * MIB
    window = fresh_window(
        _reading(), _reading(peak_reserved=1000 + demand, driver=1200 + demand), oom=False, spill=False
    )
    common = {
        "estimator": FLUX3_DENOISE_SPEC.id,
        "version": FLUX3_DENOISE_SPEC.version,
        "platform": dataclasses.asdict(platform or Platform("cudnn", "native", "windows")),
        "features": features,
        "device": "AMD Radeon RX 9060 XT (gfx1200)" if platform else "NVIDIA GeForce RTX 4090",
    }
    line = fresh_points.point_line(point, common=common, provenance="test", windows=[window, window], raw={})
    record = fresh_points.file_record(line)
    assert record is not None
    return record


CUDNN_NATIVE = {"conv_backend": "cudnn", "allocator": "native"}
LEGACY_VALUES = {
    f"{part}_denoise_{line}_h3072": 346030 if part == "k" else 2**30 for line in LINES.values() for part in "kb"
}
"""The hand-set constants of a `legacy` FLUX.3 denoise entry."""


def _legacy_data_copy(tmp_path: Path) -> Path:
    """The shipped data as it was before the RTX 4090's native-allocator FLUX.3 points shipped: that entry hand-set and
    `legacy` again, without its points."""
    data_dir = tmp_path / "data"
    shutil.copytree(DATA_DIR, data_dir)
    data = load_data(data_dir)["denoise.json"]

    def reverted(entry) -> bool:
        return entry.estimator == FLUX3_DENOISE_SPEC.id and entry.platform == CUDNN_NATIVE

    entries = [
        entry.model_copy(update={"values": LEGACY_VALUES, "legacy": True}) if reverted(entry) else entry
        for entry in data.coefficients
    ]
    points = [point for point in data.points if not reverted(point)]
    write_data(data_dir / "denoise.json", data.model_copy(update={"coefficients": entries, "points": points}))
    return data_dir


def test_a_runs_points_replace_the_legacy_entry_of_their_platform(tmp_path: Path, capsys) -> None:
    """The default grid's fit points span each dtype's line, reference images, the batch and Klein Base included, so
    the builder fits all three lines and the RTX 4090's native-allocator entry is legacy no more. Its floors for the
    platforms it prices unmeasured stay, and so does every other entry."""
    points = script.plan(
        "fit",
        dtypes=list(script.DTYPES),
        sizes=[512, 768, 1024, 1440, 2048],
        text_tokens=560,
        short_text_tokens=80,
        short_text_sizes=[1024, 1440],
        references=[1, 3, 10],
        batches=[2],
        blocks=2,
        extra_blocks=4,
    )
    point_file = tmp_path / "flux3.jsonl"
    point_file.write_text("".join(json.dumps(_record(point)) + "\n" for point in points), encoding="utf-8")
    data_dir = _legacy_data_copy(tmp_path)

    assert main(["--points", str(point_file), "--data-dir", str(data_dir), "--write"]) == 0

    assert f"{len(points)} fresh-process points from 1 point files" in capsys.readouterr().out

    def entries(directory: Path) -> dict:
        return {
            ",".join(entry.platform.values()): entry
            for entry in load_data(directory)["denoise.json"].coefficients
            if entry.estimator == FLUX3_DENOISE_SPEC.id
        }

    built, shipped = entries(data_dir), entries(DATA_DIR)
    fitted, shipped_native = built.pop("cudnn,native"), shipped.pop("cudnn,native")
    assert not fitted.legacy
    # The synthetic lines plus the shipped fit's 5%, the base rounded up to a whole MiB.
    assert fitted.values == {
        name: value
        for dtype, slope in SLOPES.items()
        for name, value in (
            (f"k_denoise_{LINES[dtype]}_h3072", slope * 21 // 20),
            (f"b_denoise_{LINES[dtype]}_h3072", 315 * MIB),
        )
    }
    assert fitted.unmeasured_floor == shipped_native.unmeasured_floor != {}
    assert built == shipped


ROCM_EXPANDABLE = Platform("miopen", "expandable", "windows")


@pytest.mark.parametrize("margin, built", [(-MIB, True), (MIB, False)], ids=["covered", "under-reserved"])
def test_a_real_forward_ships_as_a_lower_bound_every_refit_must_keep(tmp_path: Path, capsys, margin, built) -> None:
    """Real mode's points are checks: the builder ships them out of the fit, where they bound their estimate from below
    only. One the shipped line covers joins the data and moves no line; one that demands more than the line reserves
    makes the builder refuse, naming it, and write nothing."""
    path = write_header_only_safetensors(tmp_path / "klein.safetensors", KLEIN_KEYS)
    point = Point("real", "bfloat16", 1024, 560, references=3, checkpoint=str(path))
    estimate = FLUX3_DENOISE_SPEC.evaluate(
        _features(point), shipped_coefficients(FLUX3_DENOISE_SPEC, ROCM_EXPANDABLE)
    ).reservation_bytes
    point_file = tmp_path / "real.jsonl"
    point_file.write_text(json.dumps(_record(point, estimate + margin, ROCM_EXPANDABLE)) + "\n", encoding="utf-8")
    data_dir = tmp_path / "data"
    shutil.copytree(DATA_DIR, data_dir)

    status = main(["--points", str(point_file), "--data-dir", str(data_dir), "--write"])

    data = load_data(data_dir)["denoise.json"]
    shipped = load_data(DATA_DIR)["denoise.json"]
    assert data.coefficients == shipped.coefficients
    if built:
        assert status == 0
        (added,) = [p for p in data.points if p.device == "AMD Radeon RX 9060 XT (gfx1200)"]
        assert (added.in_fit, added.measured_bytes) == (False, estimate - MIB)
    else:
        assert status == 1
        assert "reject the result" in capsys.readouterr().err
        assert data.points == shipped.points
