"""Loading a llama.cpp Qwen3-VL encoder GGUF (Krea-2 / Ideogram 4).

Nothing here mocks the GGUF layer: each test writes a real, tiny GGUF with genuinely Q8_0-quantized
matrices and the production ``gguf_sd_loader`` reads it, so the tensors arrive as the ``GGMLTensor``
wrappers the loader has to cope with. See ``qwen3vl_gguf_fixture`` for what is and is not simulated.
"""

from unittest.mock import call, create_autospec

import pytest
import torch
import transformers.configuration_utils

import invokeai.backend.model_manager.load.model_loaders.krea2 as krea2_loaders
from invokeai.backend.model_manager.configs.qwen3_vl_encoder import Qwen3VLEncoder_GGUF_Config
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders.krea2 import Qwen3VLEncoderGGUFLoader
from invokeai.backend.model_manager.load.quantized_embedding import _PEAK_COPIES as PEAK_COPIES
from invokeai.backend.model_manager.taxonomy import Qwen3VLVariantType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.backend.model_manager.load.qwen3vl_gguf_fixture import (
    FILL_VALUE,
    HIDDEN_SIZE,
    QUANT_TOLERANCE,
    load_tiny_gguf_encoder,
)


def _dequantized(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, GGMLTensor):
        return tensor.get_dequantized_tensor().to(torch.float32)
    return tensor.detach().to(torch.float32)


@pytest.fixture
def loaded_encoder(monkeypatch, tmp_path):
    return load_tiny_gguf_encoder(monkeypatch, tmp_path)


def test_every_language_model_parameter_is_filled(loaded_encoder) -> None:
    """The load tolerates missing keys, so a mis-mapped llama.cpp name leaves its parameter on the
    meta device instead of failing at the mapping. The expectation comes from the module tree
    transformers builds, not from a transcribed key list, so a rename of the language tower fails
    here rather than at a user's 2.7 GB load.
    """
    assert [name for name, p in loaded_encoder.named_parameters() if p.is_meta] == []

    unfilled = [
        name
        for name, p in loaded_encoder.language_model.named_parameters()
        if not torch.allclose(_dequantized(p), torch.full(p.shape, FILL_VALUE), atol=QUANT_TOLERANCE)
    ]
    assert unfilled == []


def test_the_projections_stay_quantized(loaded_encoder) -> None:
    """The feature's whole point: a Q4_K 4B encoder is ~2.5 GB against the safetensors ~9 GB, and it
    only stays that way if the matrices reach the cache packed. Materializing them at load — a
    dequantization sweep beside the embedding's, say — would pass every other test here while
    quietly costing the memory this exists to save.
    """
    for layer in loaded_encoder.language_model.layers:
        for projection in (layer.self_attn.q_proj, layer.mlp.gate_proj, layer.mlp.down_proj):
            weight = projection.weight
            assert isinstance(weight, GGMLTensor)
            assert weight.dtype is torch.uint8
            # The dequantized shape, not the packed byte count: that is what makes the packed weight
            # usable in place of a dense one.
            assert weight.shape == (projection.out_features, projection.in_features)


def test_the_visual_tower_is_dropped(loaded_encoder) -> None:
    """Its weights live in the companion mmproj file, so there is nothing to load into it. Keeping
    the module would leave it on the meta device and the completeness sweep would reject every GGUF.
    Neither consumer reaches it: `Qwen3VLModel.forward` touches `visual` only for `pixel_values`.
    """
    assert isinstance(loaded_encoder.visual, torch.nn.Identity)
    assert list(loaded_encoder.visual.parameters()) == []


def test_the_embedding_weight_is_materialized(loaded_encoder) -> None:
    """`nn.Embedding` indexes its weight directly, which a quantized GGMLTensor cannot serve. Were
    this to regress, the failure would be a runtime error inside the lookup on the user's first
    prompt rather than at load.
    """
    assert not isinstance(loaded_encoder.language_model.embed_tokens.weight, GGMLTensor)

    loaded_encoder.language_model.embed_tokens(torch.tensor([[0, 1, 2]]))


def test_a_krea2_shaped_forward_produces_hidden_states(loaded_encoder) -> None:
    """Every other test here stops at "the weights are in the right modules". This one runs the call
    the Krea-2 node makes, through the cache's GGUF-aware layers, over the packed weights.

    It is what would catch the two assumptions the loader rests on and cannot check at load time: a
    transformers version that reaches `self.visual` outside the `pixel_values` branch (the module is
    an `Identity` here), and mRoPE positions — the (3, B, L) `position_ids` that only a real
    Qwen3-VL text tower accepts, and the reason this loader builds one instead of a plain Qwen3.
    """
    apply_custom_layers_to_model(loaded_encoder)

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0).unsqueeze(0).expand(3, -1, -1)

    outputs = loaded_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    hidden_states = outputs.hidden_states
    assert len(hidden_states) == loaded_encoder.config.text_config.num_hidden_layers + 1
    assert all(state.shape == (1, input_ids.shape[1], HIDDEN_SIZE) for state in hidden_states)
    assert all(torch.isfinite(state).all() for state in hidden_states)


def test_the_embedding_materialization_is_reserved(monkeypatch, tmp_path) -> None:
    """The framework reserves the GGUF's file size, which does not cover unpacking a tensor on top
    of it — 0.78 GB for the 4B, 1.24 GB for the 8B, and several times that at the peak. Unreserved,
    that is what pushes a tight machine into swap.

    The expected figure is computed from the *dequantized* shape on purpose: `Tensor.numel()` is not
    overridden on a GGMLTensor and reports the packed element count instead — a different number
    either way, and for a 4-bit quant roughly half of what is about to be allocated. The multiplier
    is imported rather than restated: what this test owns is that the loader reserves against the
    right *tensor*, while `test_quantized_embedding.py` owns the multiplier's value.
    """
    ram_cache = create_autospec(ModelCache, instance=True)

    encoder = load_tiny_gguf_encoder(monkeypatch, tmp_path, ram_cache=ram_cache)

    embed_weight = encoder.language_model.embed_tokens.weight
    expected = PEAK_COPIES * embed_weight.shape.numel() * torch.float32.itemsize
    assert ram_cache.make_room.call_args_list == [call(expected)]


def test_the_architecture_config_comes_from_the_bundle_and_never_the_network(monkeypatch, tmp_path) -> None:
    """A single-file encoder carries no config, so the loader supplies one from the vendored copy.

    Asserted through `AutoConfig.from_pretrained`, which used to serve this call: reaching the hub
    here is what an offline or proxied install cannot do, and what a re-upload could change under
    otherwise identical weights.
    """
    loader = object.__new__(Qwen3VLEncoderGGUFLoader)
    config = Qwen3VLEncoder_GGUF_Config.model_construct(
        path=str(tmp_path / "unused.gguf"), variant=Qwen3VLVariantType.Qwen3VL_4B, name="tiny"
    )

    def refuse(*args, **kwargs):
        raise AssertionError("the loader reached HuggingFace for a config it bundles")

    # Patched where a config actually resolves a repo file, not at AutoConfig: the loader builds
    # the config with Qwen3VLConfig.from_dict now, so trapping AutoConfig alone would stay green if
    # someone reintroduced Qwen3VLConfig.from_pretrained (verified by mutating the loader to do
    # exactly that). `cached_file` is the single door every from_pretrained goes through, cache hit
    # included, so this catches a reintroduced fetch even on a machine that has the repo cached.
    monkeypatch.setattr(transformers.configuration_utils, "cached_file", refuse)
    monkeypatch.setattr(krea2_loaders.AutoConfig, "from_pretrained", refuse)

    text_config = loader._load_te_config(config).text_config

    # The constants that decide the module tree the weights are folded into. rope_theta in
    # particular: 1e6 instead of Qwen3-VL's 5e6 costs relative L2 0.1008 against identical weights.
    # Read through both spellings because transformers 5.x folds the scalar into rope_parameters.
    rope_theta = getattr(text_config, "rope_theta", None) or text_config.rope_parameters["rope_theta"]

    assert rope_theta == 5000000
    assert text_config.hidden_size == 2560
