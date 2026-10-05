"""Loading a llama.cpp Mistral encoder GGUF (FLUX.2 [dev]).

This loader had its own copy of the llama.cpp key converter until the three copies were folded into
``util/llamacpp_keys``. A differential run over the two implementations says the fold is
output-identical for any real GGUF, but nothing exercised this loader end to end — so a rename in
the shared table, or a change to the prefix handling that follows it, would have surfaced at a
FLUX.2 [dev] user's first prompt instead of here.

The GGUF is real and its matrices are genuinely Q8_0, so the weights arrive as the ``GGMLTensor``
wrappers the loader has to cope with.
"""

from pathlib import Path
from unittest.mock import MagicMock, call, create_autospec

import numpy as np
import torch

from invokeai.backend.model_manager.configs.mistral_encoder import MistralEncoder_GGUF_Config
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_loaders.mistral_encoder import MistralEncoderGGUFLoader
from invokeai.backend.model_manager.load.quantized_embedding import _PEAK_COPIES as PEAK_COPIES
from invokeai.backend.model_manager.taxonomy import MistralVariantType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

# `_build_mistral_config` derives the head counts by dividing the projection rows by a fixed head
# dim of 128, so the attention shapes have to be multiples of it however small the rest is.
_HEAD_DIM = 128
_VOCAB, _HIDDEN, _FFN, _LAYERS = 64, _HEAD_DIM, 256, 2
_FILL = 0.25


def _write_mistral_gguf(path: Path) -> None:
    """The llama.cpp tensor set for a tiny Mistral decoder.

    No `attn_q_norm` / `attn_k_norm`: `MistralModel` has no QK norm, and a real Mistral GGUF carries
    none either. Matrices are Q8_0, norms stay F32 — how llama.cpp quantizes a real encoder.
    """
    import gguf
    from gguf.quants import quantize

    tensors: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (_VOCAB, _HIDDEN),
        "output_norm.weight": (_HIDDEN,),
    }
    for layer in range(_LAYERS):
        tensors.update(
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

    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_uint32("llama.block_count", _LAYERS)
    for name, shape in tensors.items():
        data = np.full(shape, _FILL, dtype=np.float32)
        if len(shape) == 2:
            packed = quantize(data, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(name, packed, raw_shape=packed.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _load(tmp_path: Path, monkeypatch, ram_cache=None):
    gguf_path = tmp_path / "mistral_tiny.gguf"
    _write_mistral_gguf(gguf_path)
    config = MistralEncoder_GGUF_Config.model_construct(
        path=str(gguf_path), variant=MistralVariantType.Cow, name="tiny"
    )
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
    model = _load(tmp_path, monkeypatch)

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

    model = _load(tmp_path, monkeypatch, ram_cache=ram_cache)

    assert not isinstance(model.embed_tokens.weight, GGMLTensor)
    assert model.embed_tokens(torch.tensor([[0, 1]])).shape == (1, 2, _HIDDEN)
    assert ram_cache.make_room.call_args_list == [call(PEAK_COPIES * _VOCAB * _HIDDEN * torch.float32.itemsize)]
