"""Loading a llama.cpp Mistral encoder GGUF (FLUX.2 [dev]).

This loader had its own copy of the llama.cpp key converter until the three copies were folded into
``util/llamacpp_keys``. A differential run over the two implementations says the fold is
output-identical for any real GGUF, but nothing exercised this loader end to end — so a rename in
the shared table, or a change to the prefix handling that follows it, would have surfaced at a
FLUX.2 [dev] user's first prompt instead of here.

llama.cpp's converter also regroups each head's query and key rows for its own RoPE, and the loader
has to put them back. Nothing fails when it does not: queries and keys carry the same reordering, so
only RoPE pairs the wrong dimensions, and the first position is unaffected. The fixture for that is
built so it cannot hide: several positions, grouped-query attention (keys regrouped by the query head
count land wrong), and a small RoPE base. Pair 0 turns a radian per position at any base; a small one
makes the other pairs rotate too, where a large one leaves them nearly still.

The GGUFs are real and their matrices genuinely Q8_0, so the weights arrive as the ``GGMLTensor``
wrappers the loader has to cope with.
"""

import gc
from pathlib import Path
from unittest.mock import MagicMock, call, create_autospec

import gguf
import numpy as np
import pytest
import torch
from gguf.quants import quantize
from transformers import MistralConfig, MistralModel

from invokeai.backend.model_manager.configs.mistral_encoder import MistralEncoder_GGUF_Config
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders.mistral_encoder import MistralEncoderGGUFLoader
from invokeai.backend.model_manager.taxonomy import MistralVariantType
from invokeai.backend.quantization.gguf.ggml_tensor import HOST_DEQUANT_PEAK_COPIES as PEAK_COPIES
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

_Q8_0 = gguf.GGMLQuantizationType.Q8_0

# `_build_mistral_config` derives the head counts by dividing the projection rows by a fixed head
# dim of 128, so the attention shapes have to be multiples of it however small the rest is.
_HEAD_DIM = 128
_VOCAB, _HIDDEN, _FFN, _LAYERS = 64, _HEAD_DIM, 256, 2
_FILL = 0.25

# The permutation fixture. Q8_0 blocks are 32 wide, so every input dimension is a multiple of 32.
_N_HEAD, _N_HEAD_KV = 2, 1
_P_VOCAB, _P_HIDDEN, _P_FFN, _P_LAYERS = 32, 64, 64, 2
_ROPE_THETA = 10000.0
_CONTEXT = 64
_TOKEN_IDS = torch.tensor([[3, 17, 5, 29, 11, 8]])

# llama.cpp component -> transformers module path within a layer, as `MistralModel` names it.
_LAYER_COMPONENTS = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
}


@pytest.fixture(autouse=True)
def _collect_only_this_tests_objects():
    """Closing a GGUF reader runs gc.collect(), which walks the worker's whole heap. Freezing what existed
    before the test leaves each collection to the test's own objects."""
    gc.freeze()
    yield
    gc.unfreeze()


def _llamacpp_permute(weights: np.ndarray, n_head: int, n_head_kv: int | None) -> np.ndarray:
    """llama.cpp's ``LlamaModel.permute``, verbatim; its converter runs it on ``q_proj`` with
    ``(n_head, n_head)`` and on ``k_proj`` with ``(n_head, n_head_kv)``.

    https://github.com/ggml-org/llama.cpp/blob/339f3242c761f16c27e5ffcde1d35dcc31933a55/conversion/llama.py#L176-L182
    """
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (
        weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
        .swapaxes(1, 2)
        .reshape(weights.shape)
    )


def _write_gguf(path: Path, architecture: str, tensors: dict[str, np.ndarray], metadata: dict[str, int | float | str]):
    """Matrices as Q8_0, vectors F32 — how llama.cpp quantizes a real encoder."""
    writer = gguf.GGUFWriter(str(path), architecture)
    for key, value in metadata.items():
        if isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        else:
            writer.add_uint32(key, value)
    for name, data in tensors.items():
        if data.ndim == 2:
            packed = quantize(data, _Q8_0)
            writer.add_tensor(name, packed, raw_shape=packed.shape, raw_dtype=_Q8_0)
        else:
            writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _write_uniform_gguf(path: Path) -> Path:
    """The llama.cpp tensor set for a tiny Mistral decoder, every value `_FILL`.

    No `attn_q_norm` / `attn_k_norm`: `MistralModel` has no QK norm, and a real Mistral GGUF carries
    none either.
    """
    shapes: dict[str, tuple[int, ...]] = {"token_embd.weight": (_VOCAB, _HIDDEN), "output_norm.weight": (_HIDDEN,)}
    for layer in range(_LAYERS):
        shapes.update(
            {
                f"blk.{layer}.attn_q.weight": (_HEAD_DIM, _HIDDEN),
                f"blk.{layer}.attn_k.weight": (_HEAD_DIM, _HIDDEN),
                f"blk.{layer}.attn_v.weight": (_HEAD_DIM, _HIDDEN),
                f"blk.{layer}.attn_output.weight": (_HIDDEN, _HEAD_DIM),
                f"blk.{layer}.ffn_gate.weight": (_FFN, _HIDDEN),
                f"blk.{layer}.ffn_up.weight": (_FFN, _HIDDEN),
                f"blk.{layer}.ffn_down.weight": (_HIDDEN, _FFN),
                f"blk.{layer}.attn_norm.weight": (_HIDDEN,),
                f"blk.{layer}.ffn_norm.weight": (_HIDDEN,),
            }
        )
    tensors = {name: np.full(shape, _FILL, dtype=np.float32) for name, shape in shapes.items()}
    metadata = {"llama.block_count": _LAYERS, "llama.attention.head_count": 1, "llama.attention.head_count_kv": 1}
    return _write_gguf(path, "llama", tensors, metadata)


def _raw_weights() -> dict[str, np.ndarray]:
    """Random weights keyed as bare ``MistralModel`` names them, every row distinct.

    Matrices sit on a power-of-two grid with a full-scale value in every 32-wide block, which Q8_0
    stores exactly, so a load can be compared against them exactly, whatever precision the
    dequantization runs at.
    """
    rng = np.random.default_rng(0)
    q_rows, kv_rows = _N_HEAD * _HEAD_DIM, _N_HEAD_KV * _HEAD_DIM
    shapes: dict[str, tuple[int, ...]] = {"embed_tokens.weight": (_P_VOCAB, _P_HIDDEN), "norm.weight": (_P_HIDDEN,)}
    for layer in range(_P_LAYERS):
        prefix = f"layers.{layer}"
        shapes.update(
            {
                f"{prefix}.self_attn.q_proj.weight": (q_rows, _P_HIDDEN),
                f"{prefix}.self_attn.k_proj.weight": (kv_rows, _P_HIDDEN),
                f"{prefix}.self_attn.v_proj.weight": (kv_rows, _P_HIDDEN),
                f"{prefix}.self_attn.o_proj.weight": (_P_HIDDEN, q_rows),
                f"{prefix}.mlp.gate_proj.weight": (_P_FFN, _P_HIDDEN),
                f"{prefix}.mlp.up_proj.weight": (_P_FFN, _P_HIDDEN),
                f"{prefix}.mlp.down_proj.weight": (_P_HIDDEN, _P_FFN),
                f"{prefix}.input_layernorm.weight": (_P_HIDDEN,),
                f"{prefix}.post_attention_layernorm.weight": (_P_HIDDEN,),
            }
        )
    weights = {}
    for name, shape in shapes.items():
        if len(shape) == 1:
            weights[name] = (1.0 + 0.1 * rng.standard_normal(shape)).astype(np.float32)
            continue
        codes = rng.integers(-127, 128, size=shape)
        codes[:, ::32] = np.where(rng.random(codes[:, ::32].shape) < 0.5, -127, 127)
        weights[name] = (codes * 2.0**-9).astype(np.float32)
    return weights


def _as_tensors(raw: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {name: torch.from_numpy(value) for name, value in raw.items()}


def _llamacpp_metadata(
    architecture: str = "llama", **overrides: int | float | str | None
) -> dict[str, int | float | str]:
    fields: dict[str, int | float | str | None] = {
        "block_count": _P_LAYERS,
        "context_length": _CONTEXT,
        "attention.head_count": _N_HEAD,
        "attention.head_count_kv": _N_HEAD_KV,
        "attention.key_length": _HEAD_DIM,
        "rope.freq_base": _ROPE_THETA,
        **overrides,
    }
    return {f"{architecture}.{key}": value for key, value in fields.items() if value is not None}


def _write_llamacpp_gguf(
    path: Path,
    raw: dict[str, np.ndarray],
    architecture: str = "llama",
    unprefixed: dict[str, int] | None = None,
    **metadata,
) -> Path:
    """`raw` as llama.cpp's converter writes it: renamed by role, query and key rows permuted.

    `unprefixed` adds metadata keys as given, outside the architecture's prefix."""
    tensors = {"token_embd.weight": raw["embed_tokens.weight"], "output_norm.weight": raw["norm.weight"]}
    for layer in range(_P_LAYERS):
        for component, module in _LAYER_COMPONENTS.items():
            value = raw[f"layers.{layer}.{module}.weight"]
            if component == "attn_q":
                value = _llamacpp_permute(value, _N_HEAD, _N_HEAD)
            elif component == "attn_k":
                value = _llamacpp_permute(value, _N_HEAD, _N_HEAD_KV)
            tensors[f"blk.{layer}.{component}.weight"] = value
    return _write_gguf(
        path, architecture, tensors, {**_llamacpp_metadata(architecture, **metadata), **(unprefixed or {})}
    )


def _load(tmp_path: Path, monkeypatch, gguf_path: Path, variant=MistralVariantType.Cow, ram_cache=None):
    config = MistralEncoder_GGUF_Config.model_construct(path=str(gguf_path), variant=variant, name="tiny")
    loader = object.__new__(MistralEncoderGGUFLoader)
    loader._logger = MagicMock()
    loader._ram_cache = ram_cache if ram_cache is not None else create_autospec(ModelCache, instance=True)
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.mistral_encoder.TorchDevice.choose_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.mistral_encoder.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )
    return loader._load_from_gguf(config)


def test_the_shared_converter_lands_every_weight_in_the_mistral_module_tree(tmp_path, monkeypatch) -> None:
    """The fold's actual risk: this loader builds a *bare* `MistralModel`, so the shared converter's
    `model.`-prefixed output has to survive `_strip_known_prefixes` and
    `_convert_for_bare_mistral_model` before it fits.

    Asserting over *every* parameter is what makes this catch a mis-mapped key. A "no meta tensors"
    check would not: `_materialize_remaining_meta_tensors` fills whatever the load missed with zeros
    (ones for norms) and only logs, so a key mapped to a module that does not exist leaves a
    plausible-looking, silently wrong model rather than a meta tensor. The fixture writes every
    tensor as the same non-default value, so a defaulted parameter is exactly what fails here.

    What this cannot catch is a transposition *within* a same-shaped pair -- `attn_q`/`attn_k`/
    `attn_v`, `ffn_gate`/`ffn_up`, the two norms. `test_llamacpp_keys.py` owns that case, where each
    component carries a distinct sentinel.
    """
    model = _load(tmp_path, monkeypatch, _write_uniform_gguf(tmp_path / "mistral_tiny.gguf"))

    assert len(model.layers) == _LAYERS
    for name, parameter in model.named_parameters():
        actual = parameter.get_dequantized_tensor() if isinstance(parameter, GGMLTensor) else parameter
        assert torch.allclose(actual, torch.full(parameter.shape, _FILL), atol=1e-3), name

    for layer in model.layers:
        for projection in (layer.self_attn.q_proj, layer.mlp.gate_proj, layer.mlp.down_proj):
            # Still packed: the reason to load a GGUF at all.
            assert isinstance(projection.weight, GGMLTensor)
            assert projection.weight.shape == (projection.out_features, projection.in_features)


def test_the_embedding_is_materialized_and_its_copy_reserved(tmp_path, monkeypatch) -> None:
    """The site this change added a reservation to. `nn.Embedding` indexes its weight directly, so
    the dequantized copy is unavoidable — and the framework reserved only the file size.
    """
    ram_cache = create_autospec(ModelCache, instance=True)

    model = _load(tmp_path, monkeypatch, _write_uniform_gguf(tmp_path / "mistral_tiny.gguf"), ram_cache=ram_cache)

    assert not isinstance(model.embed_tokens.weight, GGMLTensor)
    assert model.embed_tokens(torch.tensor([[0, 1]])).shape == (1, 2, _HIDDEN)
    assert ram_cache.make_room.call_args_list == [call(PEAK_COPIES * _VOCAB * _HIDDEN * torch.float32.itemsize)]


@pytest.mark.parametrize(
    "architecture, metadata",
    # A declared "none" scaling is plain RoPE, not a reason to refuse.
    [("llama", {}), ("mistral3", {}), ("mistral3", {"rope.scaling.type": "none"})],
    ids=["llama", "mistral3", "mistral3-none-scaling"],
)
def test_llamacpp_query_and_key_rows_load_in_transformers_order(
    tmp_path, monkeypatch, architecture: str, metadata: dict
) -> None:
    """Both names llama.cpp writes Mistral Small 3 under, with the metadata under each one's prefix.

    Values rows too: llama.cpp leaves them in order, and regrouping them with the others would be
    the same silent corruption. Queries and keys stay packed -- the un-permute gathers quantized rows.
    """
    raw = _raw_weights()
    expected = _as_tensors(raw)
    gguf_path = _write_llamacpp_gguf(tmp_path / "mistral_llamacpp.gguf", raw, architecture, **metadata)

    model = _load(tmp_path, monkeypatch, gguf_path, variant=MistralVariantType.Mistral24B)

    for index, layer in enumerate(model.layers):
        for projection in ("q_proj", "k_proj", "v_proj"):
            weight = getattr(layer.self_attn, projection).weight
            assert isinstance(weight, GGMLTensor)
            assert weight._ggml_quantization_type is _Q8_0
            assert torch.equal(
                weight.get_dequantized_tensor(), expected[f"layers.{index}.self_attn.{projection}.weight"]
            )


def test_a_llamacpp_gguf_repeating_the_header_as_metadata_loads(tmp_path, monkeypatch) -> None:
    """Install reads such a file with the header-tolerant reader; the loader's metadata read must too,
    or the architecture reads as None and a file install accepted is refused at load."""
    raw = _raw_weights()
    expected = _as_tensors(raw)
    gguf_path = _write_llamacpp_gguf(tmp_path / "mistral_llamacpp.gguf", raw, unprefixed={"GGUF.kv_count": 3})

    model = _load(tmp_path, monkeypatch, gguf_path, variant=MistralVariantType.Mistral24B)

    for index, layer in enumerate(model.layers):
        for projection in ("q_proj", "k_proj"):
            weight = getattr(layer.self_attn, projection).weight
            assert torch.equal(
                weight.get_dequantized_tensor(), expected[f"layers.{index}.self_attn.{projection}.weight"]
            )


def test_a_llamacpp_encoder_computes_what_the_reference_model_computes(tmp_path, monkeypatch) -> None:
    """The loaded GGUF against a dense `MistralModel` holding the same weights in transformers order.

    The comparison that matters to a FLUX.2 [dev] user: hidden states over several positions, where
    RoPE is what a wrong row order changes. Run as the model cache runs it, through the custom layers
    that dequantize GGML weights.
    """
    raw = _raw_weights()
    model = _load(
        tmp_path,
        monkeypatch,
        _write_llamacpp_gguf(tmp_path / "mistral_llamacpp.gguf", raw),
        variant=MistralVariantType.Mistral24B,
    )
    apply_custom_layers_to_model(model)
    reference = MistralModel(
        MistralConfig(
            vocab_size=_P_VOCAB,
            hidden_size=_P_HIDDEN,
            intermediate_size=_P_FFN,
            num_hidden_layers=_P_LAYERS,
            num_attention_heads=_N_HEAD,
            num_key_value_heads=_N_HEAD_KV,
            head_dim=_HEAD_DIM,
            max_position_embeddings=_CONTEXT,
            rms_norm_eps=1e-5,
            rope_theta=_ROPE_THETA,
            tie_word_embeddings=False,
        )
    ).eval()
    reference.load_state_dict(_as_tensors(raw), strict=True)

    with torch.no_grad():
        actual = model(input_ids=_TOKEN_IDS).last_hidden_state
        expected = reference(input_ids=_TOKEN_IDS).last_hidden_state

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_an_hf_named_gguf_is_loaded_in_the_order_it_is_stored(tmp_path, monkeypatch) -> None:
    """gguf-org's cow builds -- every Mistral GGUF starter -- keep transformers' names and order and
    carry no head counts. Regrouping their rows would corrupt every starter install."""
    raw = _raw_weights()
    expected = _as_tensors(raw)
    hf_named = {f"model.{name}": value for name, value in raw.items()}
    gguf_path = _write_gguf(tmp_path / "cow.gguf", "cow", hf_named, {})

    model = _load(tmp_path, monkeypatch, gguf_path)

    for index, layer in enumerate(model.layers):
        for projection in ("q_proj", "k_proj"):
            weight = getattr(layer.self_attn, projection).weight
            assert torch.equal(
                weight.get_dequantized_tensor(), expected[f"layers.{index}.self_attn.{projection}.weight"]
            )


@pytest.mark.parametrize(
    "architecture, metadata, match",
    [
        # A Qwen3 converter never permutes; un-permuting one would scramble it.
        ("qwen3", {}, "architecture 'qwen3'"),
        ("llama", {"attention.head_count_kv": None}, "without llama.attention.head_count_kv,"),
        # The rows are 2 x 128 query and 1 x 128 key; the metadata has to agree, or the regrouping is wrong.
        ("llama", {"attention.head_count": 4}, "4 query and 1 key/value heads"),
        # The rows agree with 4 x 64 and 2 x 64, but the model builds 128-wide heads, whose RoPE would
        # pair dimensions of different heads.
        ("llama", {"attention.head_count": 4, "attention.head_count_kv": 2, "attention.key_length": 64}, "of width 64"),
        # Devstral Small 2's shape: Mistral Small 3's geometry, but YaRN, which a plain MistralModel ignores.
        ("mistral3", {"rope.scaling.type": "yarn", "rope.scaling.factor": 16.0}, "rope.scaling.type='yarn'"),
        # A YaRN parameter without a declared scaling type is refused just the same.
        ("llama", {"rope.scaling.yarn_log_multiplier": 1.0}, "llama.rope.scaling.yarn_log_multiplier=1.0"),
        # Llama 4-style attention temperature, also numeric only.
        ("mistral3", {"attention.temperature_scale": 0.5}, "mistral3.attention.temperature_scale=0.5"),
    ],
)
def test_a_llamacpp_gguf_the_rows_cannot_be_trusted_for_is_refused(
    tmp_path, monkeypatch, architecture: str, metadata: dict, match: str
) -> None:
    """For records installed before identification refused such files: the alternative to refusing
    is an encoder that loads cleanly and conditions wrongly."""
    gguf_path = _write_llamacpp_gguf(tmp_path / "mistral_llamacpp.gguf", _raw_weights(), architecture, **metadata)

    with pytest.raises(ValueError, match=match):
        _load(tmp_path, monkeypatch, gguf_path, variant=MistralVariantType.Mistral24B)


@pytest.mark.parametrize("layout", ["llama.cpp", "hf_named"])
def test_a_5120_wide_encoder_without_mistral_small_3s_mlp_is_refused(tmp_path, monkeypatch, layout: str) -> None:
    """Records of Mistral NeMo 12B -- Mistral Small 3's width, depth, heads and vocabulary, a 14336-wide
    MLP -- made before identification refused it, and HF-named files of other 5120-wide models.

    Only what the guard reads is written, at the real width; the MLP is narrower still, which the guard
    treats as it treats NeMo's.
    """
    hidden, mlp_width = 5120, 64
    shapes = {
        "embed": (_P_VOCAB, hidden),
        "q": (_N_HEAD * _HEAD_DIM, hidden),
        "k": (_N_HEAD_KV * _HEAD_DIM, hidden),
        "gate": (mlp_width, hidden),
    }
    if layout == "llama.cpp":
        names = {"embed": "token_embd.weight", "q": "blk.0.attn_q.weight", "k": "blk.0.attn_k.weight"}
        names["gate"] = "blk.0.ffn_gate.weight"
        architecture, metadata = "llama", _llamacpp_metadata()
    else:
        names = {"embed": "model.embed_tokens.weight", "q": "model.layers.0.self_attn.q_proj.weight"}
        names |= {"k": "model.layers.0.self_attn.k_proj.weight", "gate": "model.layers.0.mlp.gate_proj.weight"}
        architecture, metadata = "cow", {}
    tensors = {names[role]: np.full(shape, _FILL, dtype=np.float32) for role, shape in shapes.items()}
    gguf_path = _write_gguf(tmp_path / "nemo-like.gguf", architecture, tensors, metadata)

    with pytest.raises(ValueError, match="nemo-like.gguf is 5120 wide with an MLP of width 64"):
        _load(tmp_path, monkeypatch, gguf_path, variant=MistralVariantType.Mistral24B)
