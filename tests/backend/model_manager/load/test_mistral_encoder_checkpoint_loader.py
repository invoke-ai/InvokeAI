"""Loader-level tests for the Mistral encoder's single-file path.

These drive `_load_text_encoder` on a real, tiny safetensors file, because what they pin is order inside the loader:
the nvfp4 layers have to leave the state dict before either scaled-fp8 branch reads the side channel, the config has
to count them although they left it, and the one reservation has to come before anything widens a weight. Both fp8
branches are exercised: the one that keeps fp8 pops every block scale, the one that dequantizes folds them.
"""

import json
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import load_file, save_file

from invokeai.backend.model_manager.configs.mistral_encoder import MistralEncoder_Checkpoint_Config
from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.model_loaders import mistral_encoder
from invokeai.backend.model_manager.load.model_loaders.mistral_encoder import MistralEncoderCheckpointLoader
from invokeai.backend.model_manager.taxonomy import MistralVariantType
from invokeai.backend.quantization import fp8_scaled
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.quantization.nvfp4 import NVFP4Linear, NVFP4Payload, predict_nvfp4_install_host_size
from tests.fixtures.loader_seams import Seam, prepare
from tests.fixtures.quantized_payloads import comfy_quant_marker, quantize_convrot
from tests.fixtures.safetensors_files import record_reads

# The loader derives head counts as projection rows // 128: two query heads over one key/value head, so a config
# that mixes up the two lookups, or falls back to the cow model's, cannot build this model. Dimensions are multiples
# of 128 rows and 64 columns, as nvfp4's tile layout needs. VOCAB too, because the LM head is packed.
HIDDEN = 256
KV_ROWS = 128
INTERMEDIATE = 512
VOCAB = 128
LAYERS = 2
# The file carries 32-bit floats, the compute dtype is bf16 as in production.
COMPUTE_DTYPE = torch.bfloat16

NVFP4_PROJECTIONS = {
    "self_attn.q_proj": (HIDDEN, HIDDEN),
    "self_attn.k_proj": (KV_ROWS, HIDDEN),
    "self_attn.o_proj": (HIDDEN, HIDDEN),
    "mlp.gate_proj": (INTERMEDIATE, HIDDEN),
    "mlp.up_proj": (INTERMEDIATE, HIDDEN),
    "mlp.down_proj": (HIDDEN, INTERMEDIATE),
}
FP8_PROJECTION = "self_attn.v_proj"


def _packed_bytes(rows: int, columns: int) -> int:
    """A layer as stored: two codes per byte, one e4m3 scale per 16 elements, a float32 global scale."""
    return rows * columns // 2 + rows * (columns // 16) + 4


def _write_checkpoint(tmp_path: Path, evidence: str) -> tuple[Path, dict[str, torch.Tensor], dict[int, torch.Tensor]]:
    """A Comfy-style fp4_mixed Mistral: nvfp4 projections and LM head, one scaled-fp8 projection per layer, full
    precision for the rest.

    The nvfp4 layers are named by per-tensor markers, as in Comfy-Org's build, or -- with `evidence="header"` --
    only in the `_quantization_metadata` header, with every key under the `language_model.` wrapper prefix some
    redistributions add. The loader strips that prefix from the keys and has to strip it from the header's
    names as well, or it refuses the layers as unnamed. With `evidence="none"` the projections are plain weights.

    Returns the file, the weights the projections must compute with, and each layer's dequantized fp8
    projection. Codes 2 and 10 decode to +1.0 and -1.0, so no E2M1 table is needed here.
    """
    torch.manual_seed(0)
    prefix = "language_model." if evidence == "header" else ""
    tensors: dict[str, torch.Tensor] = {}
    expected: dict[str, torch.Tensor] = {}
    fp8_dequantized: dict[int, torch.Tensor] = {}
    header: dict[str, dict[str, str]] = {}

    def add_projection(path: str, shape: tuple[int, int]) -> torch.Tensor:
        positive = torch.randint(0, 2, shape, dtype=torch.bool)
        if evidence == "none":
            tensors[f"{path}.weight"] = torch.where(positive, 0.5, -0.5)
            return tensors[f"{path}.weight"]
        codes = torch.where(positive, 2, 10).to(torch.uint8)
        tensors[f"{path}.weight"] = (codes[:, 0::2] << 4) | codes[:, 1::2]
        tensors[f"{path}.weight_scale"] = torch.full((shape[0], shape[1] // 16), 2.0).to(torch.float8_e4m3fn)
        tensors[f"{path}.weight_scale_2"] = torch.tensor(0.25)
        if evidence == "header":
            header[path] = {"format": "nvfp4"}
        else:
            tensors[f"{path}.comfy_quant"] = comfy_quant_marker({"format": "nvfp4"})
        return torch.where(positive, 0.5, -0.5)

    for layer in range(LAYERS):
        for name, shape in NVFP4_PROJECTIONS.items():
            expected[f"layers.{layer}.{name}"] = add_projection(f"{prefix}model.layers.{layer}.{name}", shape)

        fp8_values = torch.randint(-8, 9, (KV_ROWS, HIDDEN)).float()
        v_proj = f"{prefix}model.layers.{layer}.{FP8_PROJECTION}"
        tensors[f"{v_proj}.weight"] = fp8_values.to(torch.float8_e4m3fn)
        tensors[f"{v_proj}.weight_scale"] = torch.tensor(0.5)
        tensors[f"{v_proj}.comfy_quant"] = comfy_quant_marker({"format": "float8_e4m3fn"})
        fp8_dequantized[layer] = fp8_values * 0.5

        for norm in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"{prefix}model.layers.{layer}.{norm}.weight"] = torch.ones(HIDDEN)

    # Bare MistralModel has no LM head; a packed one must be dropped, not refused as naming no module.
    add_projection(f"{prefix}lm_head", (VOCAB, HIDDEN))
    tensors[f"{prefix}model.embed_tokens.weight"] = torch.randn(VOCAB, HIDDEN)
    tensors[f"{prefix}model.norm.weight"] = torch.ones(HIDDEN)
    tensors["tekken_model"] = torch.randint(0, 256, (64,), dtype=torch.uint8)

    checkpoint = tmp_path / "mistral_3_small_flux2_fp4_mixed.safetensors"
    metadata = {"_quantization_metadata": json.dumps({"layers": header})} if header else None
    save_file(tensors, checkpoint, metadata=metadata)
    return checkpoint, expected, fp8_dequantized


SEAM = Seam(
    loader=MistralEncoderCheckpointLoader,
    module=mistral_encoder,
    entry="_load_text_encoder",
    patches_device=True,
    compute_dtype=COMPUTE_DTYPE,
    # The loader takes its dtype from the device and has no FP8 Storage pass, so neither is supplied:
    # an attribute a loader does not read is one a test must not make it look like it reads.
    sets_torch_dtype=False,
    casts_fp8_storage=False,
)

#: The two steps that widen weights. Room has to have been reserved before either runs.
WIDENING = ("_drop_quantization_metadata", "split_fp8_scaled_layers")


def _fp8_matmul(available: bool):
    """The encoder asks the device, not the model config, so this is the only input to the
    keep-or-fold decision -- stubbing the decision itself would let a loader that stopped
    asking stay green."""

    def geometry(patch):
        patch.setattr(load_default, "should_keep_fp8_weights", lambda _device: available)

    return geometry


def _config(checkpoint: Path) -> MistralEncoder_Checkpoint_Config:
    return MistralEncoder_Checkpoint_Config.model_construct(path=str(checkpoint), variant=MistralVariantType.Cow)


def test_reservation_does_not_recast_folded_fp8_weights() -> None:
    """Folded weights and their removed scales are not later cast destinations.

    Real 1M-element payloads keep the arithmetic representative without creating the whole encoder.
    """
    elements = 1 << 20
    layers = 32
    state_dict = {}
    for layer in range(layers):
        state_dict[f"layer.{layer}.weight"] = torch.empty(elements, dtype=torch.float8_e4m3fn)
        state_dict[f"layer.{layer}.weight_scale"] = torch.empty((), dtype=torch.float32)
    folded_keys = set(state_dict)
    folded_output_bytes = layers * elements * COMPUTE_DTYPE.itemsize
    weight = torch.empty(elements, dtype=torch.float8_e4m3fn, device="meta")
    scale = torch.empty((), dtype=torch.float32, device="meta")
    fold_peak = max(
        (layer * (folded_output_bytes // layers - elements) + folded_output_bytes // layers)
        + fp8_scaled.fold_transient_bytes(weight, scale)
        for layer in range(layers)
    )
    calls: list[tuple[int, int]] = []

    def reserve(needed: int, *, host_bytes_needed: int) -> None:
        calls.append((needed, host_bytes_needed))

    needed = reserve_for_load(
        reserve,
        state_dict,
        COMPUTE_DTYPE,
        keep_fp8=False,
        model=None,
        fp8_layers={},
        nvfp4_payloads={},
        folded_keys=folded_keys,
        folded_output_bytes=folded_output_bytes,
        extra_host_bytes=fold_peak,
    )

    assert needed == folded_output_bytes
    assert calls == [(folded_output_bytes, fold_peak)]


@pytest.mark.parametrize("source_tensors_mapped", [False, True], ids=["owned", "mapped"])
def test_fold_net_delta_carries_through_cast_and_install(source_tensors_mapped: bool) -> None:
    """The post-fold phase starts from outputs minus only owned folded sources."""
    elements = 128
    state_dict = {
        "layer.weight": torch.empty(elements, dtype=torch.float8_e4m3fn),
        "layer.weight_scale": torch.empty((), dtype=torch.float32),
        "dense": torch.empty(elements, dtype=COMPUTE_DTYPE),
    }
    folded_keys = {"layer.weight", "layer.weight_scale"}
    folded_output_bytes = elements * COMPUTE_DTYPE.itemsize
    payload = NVFP4Payload(
        weight=torch.empty((128, 32), dtype=torch.uint8),
        weight_scale=torch.empty((128, 4), dtype=torch.float8_e4m3fn),
        weight_scale_2=torch.empty((), dtype=torch.float32),
    )
    model = torch.nn.Module()
    model.add_module("packed", torch.nn.Conv1d(1, 1, 1))
    install_peak = predict_nvfp4_install_host_size(model, {"packed": payload}, COMPUTE_DTYPE, ())
    source_release = 0 if source_tensors_mapped else elements + 4
    post_fold_delta = folded_output_bytes - source_release
    needed = folded_output_bytes + elements * COMPUTE_DTYPE.itemsize + 128 * 64 * COMPUTE_DTYPE.itemsize
    source_bytes = (
        elements * COMPUTE_DTYPE.itemsize if source_tensors_mapped else elements * COMPUTE_DTYPE.itemsize + elements + 4
    )
    expected_host = max(0, needed - source_bytes, post_fold_delta + install_peak)
    calls: list[tuple[int, int]] = []

    def reserve(bytes_needed: int, *, host_bytes_needed: int) -> None:
        calls.append((bytes_needed, host_bytes_needed))

    actual = reserve_for_load(
        reserve,
        state_dict,
        COMPUTE_DTYPE,
        keep_fp8=False,
        model=model,
        fp8_layers={},
        nvfp4_payloads={"packed": payload},
        source_tensors_mapped=source_tensors_mapped,
        folded_keys=folded_keys,
        folded_output_bytes=folded_output_bytes,
    )

    assert actual == needed
    assert calls == [(needed, expected_host)]


def test_mxfp8_decode_headroom_is_checked_before_extraction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint, _, _ = _write_checkpoint(tmp_path, evidence="marker")
    run = prepare(SEAM, monkeypatch, geometry=_fp8_matmul(True))
    make_room = MagicMock(return_value=SimpleNamespace(ram_headroom_satisfied=False))
    run.loader._ram_cache.make_room = make_room
    monkeypatch.setattr(
        "invokeai.backend.quantization.fp8_scaled.mxfp8_scale_decode_host_bytes", lambda _sd, _hints: 1234
    )
    extracted = False
    original_extract = mistral_encoder.extract_fp8_scaled_layers

    def observe_extract(*args, **kwargs):
        nonlocal extracted
        extracted = True
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(mistral_encoder, "extract_fp8_scaled_layers", observe_extract)

    with pytest.raises(MemoryError, match="Insufficient host memory"):
        run.load(_config(checkpoint))

    assert extracted is False
    make_room.assert_called_once_with(0, host_bytes_needed=1234)


@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float8_e4m3fn], ids=["fp32_scale", "fp8_scale"])
def test_mistral_fp8_fallback_folds_in_bounded_slabs(monkeypatch: pytest.MonkeyPatch, scale_dtype: torch.dtype) -> None:
    """The fallback must not hold full FP32 weight/product copies beside the target output."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    class LiveBytes(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.live = 0
            self.peak = 0

        def _free(self, size: int) -> None:
            self.live -= size

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            inputs = {t.untyped_storage().data_ptr() for t in tree_leaves((args, kwargs)) if torch.is_tensor(t)}
            out = func(*args, **kwargs)
            for tensor in tree_leaves(out):
                storage = tensor.untyped_storage() if torch.is_tensor(tensor) else None
                if storage is None or storage.nbytes() == 0 or storage.data_ptr() in inputs:
                    continue
                inputs.add(storage.data_ptr())
                self.live += storage.nbytes()
                self.peak = max(self.peak, self.live)
                weakref.finalize(tensor, self._free, storage.nbytes())
            return out

    slab_elements = 256
    monkeypatch.setattr(fp8_scaled, "_FOLD_SLAB_ELEMENTS", slab_elements)
    weight = torch.arange(128 * 64, dtype=torch.float32).reshape(128, 64).to(torch.float8_e4m3fn)
    scale = torch.linspace(0.5, 1.5, 128).to(scale_dtype)
    expected = (weight.float() * scale.float().reshape(-1, 1)).to(torch.bfloat16)
    sd = {"layer.weight": weight, "layer.weight_scale": scale}

    with LiveBytes() as live:
        mistral_encoder._drop_quantization_metadata(sd, logger=MagicMock(), target_dtype=torch.bfloat16)

    torch.testing.assert_close(sd["layer.weight"], expected, rtol=0, atol=0, equal_nan=True)
    assert sd["layer.weight"].dtype is torch.bfloat16
    assert "layer.weight_scale" not in sd
    target_bytes = weight.numel() * torch.bfloat16.itemsize
    bound = fp8_scaled.fold_transient_bytes(torch.empty(weight.shape, dtype=weight.dtype, device="meta"), scale) + (
        scale.numel() * torch.float32.itemsize if scale_dtype is not torch.float32 else 0
    )
    assert live.peak - target_bytes <= bound


def test_mistral_fp8_fallback_reserves_fold_outputs_through_dense_cast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint, _, _ = _write_checkpoint(tmp_path, evidence="none")
    tensors = load_file(checkpoint)
    tensors["model.embed_tokens.weight"] = torch.zeros((4096, HIDDEN), dtype=torch.float32)
    save_file(tensors, checkpoint)
    run = prepare(SEAM, monkeypatch, geometry=_fp8_matmul(False), observe=WIDENING)
    make_room_calls: list[dict[str, int]] = []

    def record_make_room(_needed: int, **kwargs: int) -> None:
        make_room_calls.append(kwargs)

    run.loader._ram_cache.make_room = record_make_room

    original_load = mistral_encoder.load_safetensors_checkpoint
    last_folded_weight_ref: weakref.ReferenceType[torch.Tensor] | None = None

    def capture_load(path, *, select=None, owned: bool = False):
        nonlocal last_folded_weight_ref
        loaded = original_load(path, select=select, owned=owned)
        last_folded_weight_ref = weakref.ref(loaded[f"model.layers.{LAYERS - 1}.{FP8_PROJECTION}.weight"])
        return loaded

    monkeypatch.setattr(mistral_encoder, "load_safetensors_checkpoint", capture_load)
    original_drop = mistral_encoder._drop_quantization_metadata
    sizing_source_released: list[bool] = []

    def observe_drop(sd, logger, target_dtype=None):
        result = original_drop(sd, logger, target_dtype)
        sizing_source_released.append(last_folded_weight_ref() is None)
        return result

    monkeypatch.setattr(mistral_encoder, "_drop_quantization_metadata", observe_drop)
    model = run.load(_config(checkpoint))

    fold_outputs = sum(
        tensors[f"model.layers.{layer}.{FP8_PROJECTION}.weight"].numel() * COMPUTE_DTYPE.itemsize
        for layer in range(LAYERS)
    )
    embedding_output = tensors["model.embed_tokens.weight"].numel() * COMPUTE_DTYPE.itemsize
    folded_source_bytes = sum(
        tensors[f"model.layers.{layer}.{FP8_PROJECTION}.{suffix}"].nbytes
        for layer in range(LAYERS)
        for suffix in ("weight", "weight_scale", "comfy_quant")
    )
    assert make_room_calls[-1]["host_bytes_needed"] >= fold_outputs - folded_source_bytes + embedding_output
    assert sizing_source_released == [True]
    assert model.get_input_embeddings().weight.dtype is COMPUTE_DTYPE
    assert torch.equal(model.get_input_embeddings().weight, tensors["model.embed_tokens.weight"].to(COMPUTE_DTYPE))


@pytest.mark.parametrize(
    ("keep_fp8", "evidence"),
    [(False, "marker"), (True, "marker"), (False, "header")],
    ids=["fp8_dequantized", "fp8_kept", "named_in_prefixed_header"],
)
def test_an_nvfp4_mixed_checkpoint_keeps_its_nvfp4_layers_packed_under_either_fp8_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keep_fp8: bool, evidence: str
) -> None:
    checkpoint, expected, fp8_dequantized = _write_checkpoint(tmp_path, evidence)
    run = prepare(SEAM, monkeypatch, geometry=_fp8_matmul(keep_fp8), observe=WIDENING)

    model = run.load(_config(checkpoint))

    assert len(model.layers) == LAYERS
    assert (model.config.num_attention_heads, model.config.num_key_value_heads) == (2, 1)
    for path, weight in expected.items():
        module = model.get_submodule(path)
        assert isinstance(module, NVFP4Linear), path
        assert module.weight.dtype is torch.uint8, path
        # Installed after the cast, which would otherwise round the global scale to the compute dtype.
        assert module.weight_scale_2.dtype is torch.float32, path
        # A float32 activation, not the compute dtype: torch 2.7's CPU bf16 GEMM faults with an illegal
        # instruction on part of GitHub's windows runner fleet. These values are exact in either width.
        x = torch.randn(3, module.in_features, dtype=torch.float32)
        torch.testing.assert_close(module(x), x @ weight.T)
    for layer, weight in fp8_dequantized.items():
        v_proj = model.layers[layer].self_attn.v_proj
        if keep_fp8:
            assert v_proj.weight.dtype is torch.float8_e4m3fn
            assert torch.equal(v_proj.weight.float() * v_proj.weight_scale, weight)
        else:
            assert torch.equal(v_proj.weight, weight.to(COMPUTE_DTYPE))
    assert run.order and all(reserved for _step, reserved in run.order), run.order

    # One reservation for what the state dict ends up holding: the nvfp4 layers packed (the LM head not at all), the
    # fp8 weights at one byte where the keep branch holds them and at the compute dtype where they are dequantized,
    # full precision at the compute dtype. Scale scalars and markers aside, which is what the tolerance is for.
    packed = LAYERS * sum(_packed_bytes(*shape) for shape in NVFP4_PROJECTIONS.values())
    fp8_elements = LAYERS * KV_ROWS * HIDDEN
    full_precision_elements = VOCAB * HIDDEN + (2 * LAYERS + 1) * HIDDEN
    expected_bytes = (
        packed
        + fp8_elements * (1 if keep_fp8 else COMPUTE_DTYPE.itemsize)
        + full_precision_elements * COMPUTE_DTYPE.itemsize
        + 64  # tekken_model
    )
    assert len(run.reserved) == 1
    assert abs(run.reserved[0] - expected_bytes) < 1024, (run.reserved, expected_bytes)


@pytest.mark.parametrize("keep_fp8", [False, True], ids=["fp8_folded", "fp8_kept"])
def test_an_int8_convrot_layer_is_refused_rather_than_folded_unrotated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keep_fp8: bool
) -> None:
    """This encoder has no int8 branch, and `int8_tensorwise` shares the fp8 key layout, so the fold
    applied the scale and skipped the inverse rotation -- a weight of the right shape and magnitude
    that bears no relation to the stored one.

    Both branches, because the first version of this guard only covered one. `keep_fp8` is
    unconditionally true on CUDA, and that branch's `extract_fp8_scaled_layers` pops every scale key
    -- discarding the ones whose weight is not float8 -- and deletes the markers with them. A check
    downstream of it is a no-op on the device almost everyone loads on.
    """
    checkpoint, _, _ = _write_checkpoint(tmp_path, evidence="none")
    tensors = load_file(checkpoint)
    target = f"model.layers.0.{FP8_PROJECTION}"
    payload = quantize_convrot(torch.randn(KV_ROWS, HIDDEN), group_size=64)
    tensors[f"{target}.weight"] = payload.codes
    tensors[f"{target}.weight_scale"] = payload.scale
    tensors[f"{target}.comfy_quant"] = comfy_quant_marker(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64}
    )
    save_file(tensors, checkpoint)
    run = prepare(SEAM, monkeypatch, geometry=_fp8_matmul(keep_fp8))

    with pytest.raises(ValueError, match="quantized with convrot"):
        run.load(_config(checkpoint))

    # And before the cache was evicted for a load that cannot finish. The reservation sits fifty
    # lines below the check; a guard placed after it would cost a 16 GiB encoder's worth of room.
    assert run.reserved == []


def test_a_raw_fp8_weight_survives_a_split_that_empties_the_scaled_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The reservation and the cast have to decide with the same value, not the same expression.

    `split_fp8_scaled_layers` rebinds `fp8_layers`, so reading `bool(fp8_layers)` at the cast reads
    the *post-split* mapping while the reservation read the pre-split one. This checkpoint makes the
    two disagree: its only scaled layer carries a block-wise scale that `scaled_mm` cannot apply, so
    the split folds it and hands back nothing -- and a cast that re-derives its flag from that would
    widen the *raw* fp8 weight too, after charging it one byte per element.
    """
    checkpoint, _, _ = _write_checkpoint(tmp_path, evidence="none")
    tensors = load_file(checkpoint)
    for layer in range(LAYERS):
        # A layout the matmul cannot consume, so the split dequantizes this layer and drops it.
        scaled = f"model.layers.{layer}.{FP8_PROJECTION}"
        tensors[f"{scaled}.weight_scale"] = torch.full((KV_ROWS, HIDDEN // 16), 0.5)
        # And one weight that arrives fp8 with no scale at all, which is what `keep_fp8` governs.
        raw = f"model.layers.{layer}.self_attn.o_proj"
        tensors[f"{raw}.weight"] = torch.randint(-8, 9, (HIDDEN, HIDDEN)).float().to(torch.float8_e4m3fn)
    save_file(tensors, checkpoint)
    run = prepare(SEAM, monkeypatch, geometry=_fp8_matmul(True), observe=WIDENING)

    model = run.load(_config(checkpoint))

    for layer in range(LAYERS):
        attn = model.layers[layer].self_attn
        # The premise: the split really did fold the only scaled layer, so `fp8_layers` came back empty.
        assert attn.v_proj.weight.dtype is COMPUTE_DTYPE, "the block-wise layer was expected to be folded"
        # The claim: the raw fp8 weight is still held the way the reservation was told it would be.
        assert attn.o_proj.weight.dtype is torch.float8_e4m3fn
    assert len(run.reserved) == 1


def test_a_checkpoint_without_nvfp4_layers_loads_as_before(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint, expected, _ = _write_checkpoint(tmp_path, evidence="none")
    tensors = load_file(checkpoint)
    for layer in range(LAYERS):
        tensors[f"model.layers.{layer}.{FP8_PROJECTION}.weight_scale"] = torch.full(
            (KV_ROWS,), 0.5, dtype=torch.float8_e4m3fn
        )
    save_file(tensors, checkpoint)
    run = prepare(SEAM, monkeypatch, geometry=_fp8_matmul(False), observe=WIDENING)

    model = run.load(_config(checkpoint))

    for path, weight in expected.items():
        module = model.get_submodule(path)
        assert type(module) is torch.nn.Linear, path
        assert torch.equal(module.weight, weight.to(COMPUTE_DTYPE)), path
    assert len(run.reserved) == 1
    assert run.order and all(reserved for _step, reserved in run.order), run.order


def test_the_config_reads_layer_and_head_counts_through_packed_projections() -> None:
    """A 40-layer encoder whose query and MLP projections left the state dict packed, with widths that differ from
    the cow model's defaults: every count has to come from the payloads or from the dense key projections."""

    def meta(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device="meta")

    def payload(rows: int, columns: int) -> NVFP4Payload:
        return NVFP4Payload(
            weight=meta(rows, columns // 2, dtype=torch.uint8),
            weight_scale=meta(rows, columns // 16, dtype=torch.float8_e4m3fn),
            weight_scale_2=meta(),
        )

    state_dict: dict[str, torch.Tensor] = {"model.embed_tokens.weight": meta(1024, 5120)}
    packed: dict[str, NVFP4Payload] = {}
    for layer in range(40):
        packed[f"model.layers.{layer}.self_attn.q_proj"] = payload(5120, 5120)
        packed[f"model.layers.{layer}.mlp.gate_proj"] = payload(16384, 5120)
        state_dict[f"model.layers.{layer}.self_attn.k_proj.weight"] = meta(2048, 5120, dtype=torch.float8_e4m3fn)

    config = mistral_encoder._build_mistral_config(state_dict, torch.bfloat16, packed_layers=packed)

    assert (config.num_hidden_layers, config.num_attention_heads, config.num_key_value_heads) == (40, 40, 16)
    assert config.intermediate_size == 16384


@pytest.mark.parametrize("keep_fp8", [False, True], ids=["fp8_dequantized", "fp8_kept"])
def test_an_nvfp4_layer_missing_its_global_scale_is_refused_before_the_cache_is_evicted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keep_fp8: bool
) -> None:
    """The degraded half-state, at this seam rather than at the detector.

    A packed uint8 weight with a block-scale grid and no `weight_scale_2` is what a guard keyed on
    `weight_scale_2` -- the key the decode itself keys on -- lets straight through. `find_nvfp4_layers`
    refuses it and `test_nvfp4.py` pins that; what only a seam can answer is whether this loader still
    reaches the detector before it asks the cache for room, and on both fp8 branches. Nothing but the
    order of two statements keeps `pop_nvfp4_layers` above the reservation here.
    """
    checkpoint, _, _ = _write_checkpoint(tmp_path, evidence="none")
    tensors = load_file(checkpoint)
    target = f"model.layers.0.{FP8_PROJECTION}"
    tensors[f"{target}.weight"] = torch.zeros(KV_ROWS, HIDDEN // 2, dtype=torch.uint8)
    tensors[f"{target}.weight_scale"] = torch.zeros(KV_ROWS, HIDDEN // 16).to(torch.float8_e4m3fn)
    save_file(tensors, checkpoint)
    run = prepare(SEAM, monkeypatch, geometry=_fp8_matmul(keep_fp8))

    with pytest.raises(ValueError, match="with a weight_scale but no weight_scale_2"):
        run.load(_config(checkpoint))

    assert run.reserved == []


def test_the_vision_tower_and_projector_of_a_multimodal_file_are_never_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Comfy-Org's Ministral file bundles a Pixtral tower and projector the encoder has no slot for. Read through
    ``pread`` they would cost their size in process memory before being dropped."""
    checkpoint, _, _ = _write_checkpoint(tmp_path, evidence="none")
    language_model = load_file(checkpoint)
    vision = {
        "vision_tower.transformer.layers.0.attention.q_proj.weight": torch.ones(8, 8),
        "vision_tower.patch_conv.weight": torch.ones(8, 3, 2, 2),
        "multi_modal_projector.linear_1.weight": torch.ones(8, 8),
    }
    save_file({**language_model, **vision}, checkpoint)
    log = record_reads(monkeypatch)
    run = prepare(SEAM, monkeypatch, geometry=_fp8_matmul(False))

    run.load(_config(checkpoint))

    assert set(log.tensors) == set(language_model)
