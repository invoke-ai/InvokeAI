"""Single-file Qwen3 encoders (Z-Image, FLUX.2 Klein, Anima) in ComfyUI's scaled-fp8 layout.

A tiny one-layer Qwen3 stands in for the 4B/8B encoders: the loader reads the architecture from tensor shapes, so the
same code builds both. The file is written the way ComfyUI writes `qwen_3_*_fp8_scaled`: every projection in float8
beside its `weight_scale` (a scalar for some, one value per output row for others), plus a scaled token table and LM
head, and dense norms.

What these pin is the storage decision and its consequences: on a device that can hold or multiply float8 the
projections stay float8 with the file's own scale on their module and encode like the weights the file decodes to; the
token table (not a Linear) is always folded; elsewhere everything is folded exactly as it decodes. Forwards run in
float32: CPU bf16 matmuls fault on part of the hosted Windows fleet.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from invokeai.backend.model_manager.configs.qwen3_encoder import Qwen3Encoder_Checkpoint_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules import custom_linear
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import z_image
from invokeai.backend.model_manager.load.model_loaders.z_image import Qwen3EncoderCheckpointLoader
from invokeai.backend.model_manager.taxonomy import Qwen3VariantType
from invokeai.backend.quantization.block_scale_tiles import block_scale_grid
from tests.fixtures.quantized_payloads import comfy_quant_marker

FP8 = torch.float8_e4m3fn
HIDDEN = 128
INTERMEDIATE = 256
VOCAB = 128
LAYER = "model.layers.0"

# (shape, whether the scale is one value per output row)
PROJECTIONS = {
    "self_attn.q_proj": ((HIDDEN, HIDDEN), False),
    "self_attn.k_proj": ((HIDDEN, HIDDEN), True),
    "self_attn.v_proj": ((HIDDEN, HIDDEN), True),
    "self_attn.o_proj": ((HIDDEN, HIDDEN), False),
    "mlp.gate_proj": ((INTERMEDIATE, HIDDEN), True),
    "mlp.up_proj": ((INTERMEDIATE, HIDDEN), False),
    "mlp.down_proj": ((HIDDEN, INTERMEDIATE), True),
}
MX_MODULE = f"{LAYER}.self_attn.q_proj"  # 128x128: the smallest weight an MXFP8 scale grid can describe


class _Checkpoint:
    def __init__(self, path: Path, dense: dict[str, torch.Tensor], scales: dict[str, torch.Tensor]) -> None:
        self.path = path
        # What the file decodes to, computed from the codes and scales written, independently of the loader.
        self.dense = dense
        # The scale written for each scaled layer's module path.
        self.scales = scales


def _write(
    tmp_path: Path,
    evidence: str = "marker",
    extra: dict[str, torch.Tensor] | None = None,
    *,
    scaled_fp8: bool = True,
    mx_module: str | None = None,
) -> _Checkpoint:
    """`evidence` is how the file says its layers are float8: per-layer `comfy_quant` markers, the safetensors header,
    or nothing (the scales alone). `mx_module` carries an MXFP8 scale grid, named by `evidence` alone; `scaled_fp8=False`
    writes raw float8 weights with no scale."""
    torch.manual_seed(0)
    tensors: dict[str, torch.Tensor] = {}
    dense: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    layers: dict[str, dict] = {}

    def scaled(path: str, shape: tuple[int, int], per_row: bool) -> None:
        codes = torch.randint(-12, 13, shape).float()
        scale = torch.linspace(0.25, 1.5, shape[0]) if per_row else torch.tensor(0.5)
        tensors[f"{path}.weight"] = codes.to(FP8)
        if not scaled_fp8:
            dense[f"{path}.weight"] = codes
            return
        if path == mx_module:
            exponents = torch.randint(125, 130, (shape[0], shape[1] // 32))
            tensors[f"{path}.weight_scale"] = block_scale_grid(exponents.to(torch.uint8))
            dense[f"{path}.weight"] = codes * torch.exp2(exponents.float() - 127).repeat_interleave(32, dim=1)
            layers[path] = {"format": "mxfp8"}
            return
        scale = torch.linspace(0.25, 1.5, shape[0]) if per_row else torch.tensor(0.5)
        tensors[f"{path}.weight_scale"] = scale
        dense[f"{path}.weight"] = codes * (scale[:, None] if per_row else scale)
        scales[path] = scale
        layers[path] = {"format": "float8_e4m3fn"}

    for name, (shape, per_row) in PROJECTIONS.items():
        scaled(f"{LAYER}.{name}", shape, per_row)
    scaled("model.embed_tokens", (VOCAB, HIDDEN), False)
    scaled("lm_head", (VOCAB, HIDDEN), False)
    for key, tensor in {
        f"{LAYER}.input_layernorm.weight": torch.rand(HIDDEN) + 0.5,
        f"{LAYER}.post_attention_layernorm.weight": torch.rand(HIDDEN) + 0.5,
        f"{LAYER}.self_attn.q_norm.weight": torch.rand(128) + 0.5,
        f"{LAYER}.self_attn.k_norm.weight": torch.rand(128) + 0.5,
        "model.norm.weight": torch.rand(HIDDEN) + 0.5,
    }.items():
        tensors[key] = tensor
        dense[key] = tensor
    tensors.update(extra or {})
    if mx_module is not None:
        layers = {mx_module: layers[mx_module]}

    metadata = None
    if evidence == "marker":
        for path, marker in layers.items():
            tensors[f"{path}.comfy_quant"] = comfy_quant_marker(marker)
    elif evidence == "header":
        metadata = {"_quantization_metadata": json.dumps({"layers": layers})}
    checkpoint = tmp_path / "qwen_3_4b_fp8_scaled.safetensors"
    save_file(tensors, checkpoint, metadata=metadata)
    return _Checkpoint(checkpoint, dense, scales)


def _reference(dense: dict[str, torch.Tensor]) -> Qwen3ForCausalLM:
    """A plain Qwen3 holding the decoded weights, configured the way the loader configures it."""
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=40960,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        use_sliding_window=False,
        attention_bias=False,
        attention_dropout=0.0,
    )
    model = Qwen3ForCausalLM(config)
    model.load_state_dict({k: v for k, v in dense.items() if k != "lm_head.weight"}, strict=False)
    model.tie_weights()
    return model.eval()


def _encode(model: torch.nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=torch.tensor([[1, 5, 7, 2, 30]]), output_hidden_states=True).hidden_states[-1]


def _decide(
    monkeypatch: pytest.MonkeyPatch, *, matmul: bool = False, storage: bool = False, device: str = "cuda:0"
) -> None:
    """The session runs on `device`; only a CUDA device passes the gates that are switched on."""
    monkeypatch.setattr(z_image.TorchDevice, "choose_torch_device", lambda: torch.device(device))
    monkeypatch.setattr(z_image.TorchDevice, "choose_bfloat16_safe_dtype", lambda _device: torch.float32)
    monkeypatch.setattr(z_image, "should_keep_fp8_weights", lambda d: matmul and d.type == "cuda")
    monkeypatch.setattr(z_image, "_device_supports_fp8_storage", lambda d, _logger=None: storage and d.type == "cuda")


def _loader() -> Qwen3EncoderCheckpointLoader:
    loader = object.__new__(Qwen3EncoderCheckpointLoader)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    return loader


def _config(checkpoint: _Checkpoint, cpu_only: bool | None = None) -> Qwen3Encoder_Checkpoint_Config:
    return Qwen3Encoder_Checkpoint_Config.model_construct(
        path=str(checkpoint.path), variant=Qwen3VariantType.Qwen3_4B, cpu_only=cpu_only
    )


@pytest.mark.parametrize("evidence", ["marker", "header", "none"])
@pytest.mark.parametrize("reason", ["fp8_compute", "fp8_storage"])
def test_a_device_that_holds_float8_keeps_the_projections_scaled_and_encodes_like_the_decoded_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence: str, reason: str
) -> None:
    checkpoint = _write(tmp_path, evidence)
    _decide(monkeypatch, matmul=reason == "fp8_compute", storage=reason == "fp8_storage")
    attached: list[tuple[int, int]] = []
    warn = z_image.warn_on_unattached_scales
    monkeypatch.setattr(
        z_image,
        "warn_on_unattached_scales",
        lambda logger, what, count, layers: (attached.append((count, len(layers))), warn(logger, what, count, layers)),
    )

    model = _loader()._load_from_singlefile(_config(checkpoint))

    layer = model.model.layers[0]
    for name in PROJECTIONS:
        module = layer.get_submodule(name)
        assert module.weight.dtype is FP8, name
        # The file's own scale, on the module the weight sits in.
        expected_scale = checkpoint.scales[f"{LAYER}.{name}"]
        assert torch.equal(module.weight_scale.reshape(-1), expected_scale.reshape(-1)), name
        # Persisted would make a re-save apply it twice.
        assert "weight_scale" not in module.state_dict(), name
    # Storage only, whatever the producer said: the fp8 matmul would quantize the activations too.
    assert all(layer.get_submodule(name)._fp8_full_precision_matmul is True for name in PROJECTIONS)
    # A token table is not a Linear: kept float8 it would feed fp8 activations to the first layer. The LM head is the
    # tied table.
    assert model.model.embed_tokens.weight.dtype is torch.float32
    assert torch.equal(model.model.embed_tokens.weight, checkpoint.dense["model.embed_tokens.weight"])
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert not hasattr(model.model.embed_tokens, "weight_scale")
    for key, param in model.named_parameters():
        assert (param.dtype is FP8) == (key.endswith(tuple(f"{n}.weight" for n in PROJECTIONS))), key
    # Every kept scale found its module (the LM head's was dropped with its weight, not left dangling).
    assert attached == [(len(PROJECTIONS), len(PROJECTIONS))]

    apply_custom_layers_to_model(model)
    torch.testing.assert_close(_encode(model), _encode(_reference(checkpoint.dense)), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("evidence", ["marker", "header", "none"])
def test_a_device_without_float8_folds_every_scaled_weight_to_what_the_file_decodes_to(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence: str
) -> None:
    checkpoint = _write(tmp_path, evidence)
    _decide(monkeypatch)

    model = _loader()._load_from_singlefile(_config(checkpoint))

    assert all(param.dtype is torch.float32 for param in model.parameters())
    assert not any(hasattr(module, "weight_scale") for module in model.modules())
    for key, param in model.state_dict().items():
        if key in checkpoint.dense and key != "lm_head.weight":  # the head is the tied table, not its own weight
            # Bitwise: a per-row scale folded along the wrong axis would differ here, on a square weight too.
            assert torch.equal(param, checkpoint.dense[key]), key
    torch.testing.assert_close(_encode(model), _encode(_reference(checkpoint.dense)), rtol=1e-4, atol=1e-5)


def test_a_cpu_only_encoder_is_folded_whatever_device_the_session_runs_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _write(tmp_path)
    _decide(monkeypatch, matmul=True)

    cpu = _loader()._load_from_singlefile(_config(checkpoint, cpu_only=True))
    gpu = _loader()._load_from_singlefile(_config(checkpoint, cpu_only=None))

    assert all(param.dtype is torch.float32 for param in cpu.parameters())
    assert gpu.model.layers[0].self_attn.q_proj.weight.dtype is FP8


def test_the_reservation_counts_float8_projections_one_byte_and_everything_else_at_the_compute_dtype(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _write(tmp_path)
    projection_elements = sum(shape[0] * shape[1] for shape, _ in PROJECTIONS.values())
    # The folded token table, the norms, and the scales recovered out of the file (the LM head's went with its
    # weight). The markers are consumed before anything is reserved.
    widened = VOCAB * HIDDEN + 5 * HIDDEN  # the five norms
    scale_bytes = 4 * (
        sum(shape[0] if per_row else 1 for shape, per_row in PROJECTIONS.values()) + 1  # the token table's
    )

    reserved = {}
    for keep in (True, False):
        _decide(monkeypatch, matmul=keep)
        loader = _loader()
        loader._load_from_singlefile(_config(checkpoint))
        loader._ram_cache.make_room.assert_called_once()
        reserved[keep] = loader._ram_cache.make_room.call_args.args[0]

    assert reserved[True] == projection_elements + 4 * widened + scale_bytes
    assert reserved[False] == 4 * (projection_elements + widened) + scale_bytes


@pytest.mark.parametrize("keep", [True, False])
def test_a_scale_spelling_the_loader_does_not_read_is_refused_rather_than_cast_unscaled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keep: bool
) -> None:
    """`weight_scale_inv` beside a float8 weight: cast to the compute dtype it would load as raw codes, off by the whole
    scale, and encode noise."""
    inverse = {f"{LAYER}.self_attn.q_proj.weight_scale_inv": torch.tensor(2.0)}
    checkpoint = _write(tmp_path, extra=inverse)
    _decide(monkeypatch, matmul=keep)
    loader = _loader()

    with pytest.raises(NotImplementedError, match="weight_scale_inv"):
        loader._load_from_singlefile(_config(checkpoint))

    loader._ram_cache.make_room.assert_not_called()


def test_kept_layers_are_dequantized_per_forward_even_where_the_fp8_matmul_is_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Quantizing the activations as well (down-projection outliers) would change the conditioning for a speedup one
    encoder pass per prompt does not need."""
    checkpoint = _write(tmp_path)
    _decide(monkeypatch, matmul=True)
    model = _loader()._load_from_singlefile(_config(checkpoint))
    apply_custom_layers_to_model(model)

    def refuse(*_args, **_kwargs):
        raise AssertionError("an encoder layer ran on the fp8 matmul")

    monkeypatch.setattr(custom_linear, "is_fp8_matmul_enabled", lambda: True)
    monkeypatch.setattr(custom_linear, "device_supports_fp8_matmul", lambda _device: True)
    monkeypatch.setattr(custom_linear, "scaled_mm_linear", refuse)

    torch.testing.assert_close(_encode(model), _encode(_reference(checkpoint.dense)), rtol=1e-4, atol=1e-5)


def test_an_unscaled_fp8_file_stays_fp8_where_float8_is_held_and_decodes_elsewhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _write(tmp_path, "none", scaled_fp8=False)

    _decide(monkeypatch, storage=True)
    kept = _loader()._load_from_singlefile(_config(checkpoint))
    for name in PROJECTIONS:
        module = kept.model.layers[0].get_submodule(name)
        assert module.weight.dtype is FP8 and not hasattr(module, "weight_scale"), name
        assert module._fp8_full_precision_matmul is True, name
    apply_custom_layers_to_model(kept)
    torch.testing.assert_close(_encode(kept), _encode(_reference(checkpoint.dense)), rtol=1e-4, atol=1e-5)

    _decide(monkeypatch, storage=True, device="cpu")
    folded = _loader()._load_from_singlefile(_config(checkpoint))
    assert all(param.dtype is torch.float32 for param in folded.parameters())
    torch.testing.assert_close(_encode(folded), _encode(_reference(checkpoint.dense)), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("evidence", ["marker", "header"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_a_mxfp8_layer_named_by_its_marker_or_the_header_is_decoded_and_the_rest_keeps_its_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence: str, device: str
) -> None:
    checkpoint = _write(tmp_path, evidence, mx_module=MX_MODULE)
    _decide(monkeypatch, storage=True, device=device)

    model = _loader()._load_from_singlefile(_config(checkpoint))

    # Block-wise scales never reach the matmul, so this layer is folded even where the others stay float8.
    q_proj = model.model.layers[0].self_attn.q_proj
    assert q_proj.weight.dtype is torch.float32
    assert torch.equal(q_proj.weight, checkpoint.dense[f"{MX_MODULE}.weight"])
    assert model.model.layers[0].self_attn.k_proj.weight.dtype is (FP8 if device == "cuda:0" else torch.float32)
    apply_custom_layers_to_model(model)
    torch.testing.assert_close(_encode(model), _encode(_reference(checkpoint.dense)), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_an_mxfp8_layer_nothing_names_is_refused_before_the_cache_is_evicted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: str
) -> None:
    checkpoint = _write(tmp_path, "none", mx_module=MX_MODULE)  # the grid is there; no marker, no header
    _decide(monkeypatch, storage=True, device=device)
    loader = _loader()

    with pytest.raises(NotImplementedError, match="MXFP8"):
        loader._load_from_singlefile(_config(checkpoint))

    loader._ram_cache.make_room.assert_not_called()
