"""Layer hints must survive each loader's own key rewrites.

`_quantization_metadata` names its layers in the checkpoint's scheme, while the scales are read
after the loader has stripped whatever prefix that checkpoint carries. A hint whose name still
carries the prefix matches no layer, so `full_precision_matrix_mult` is silently ignored and the
producer's "do not multiply this one in fp8" instruction is disregarded -- the exact failure the
hint plumbing exists to prevent. It has now been found twice (FLUX/Krea-2/Z-Image/Anima in round 2,
Mistral in round 3), so each loader's own prefix list gets a pin.
"""

import torch

from invokeai.backend.model_manager.load.model_loaders.mistral_encoder import (
    MISTRAL_KEY_PREFIXES,
    _strip_known_prefixes,
)
from invokeai.backend.quantization.fp8_scaled import (
    FP8_DTYPE,
    TRANSFORMER_KEY_PREFIXES,
    extract_fp8_scaled_layers,
    strip_layer_path_prefix,
)


def _quantized(path: str) -> dict[str, torch.Tensor]:
    return {
        f"{path}.weight": torch.zeros(16, 16, dtype=torch.float32).to(FP8_DTYPE),
        f"{path}.weight_scale": torch.tensor(2.0),
    }


class TestMistralHintPrefixes:
    """The Mistral loader strips wrapper prefixes the generic tuple does not know about."""

    def test_its_own_prefixes_are_not_covered_by_the_generic_tuple(self) -> None:
        """If this ever becomes false the extra argument at the call site is dead weight."""
        assert not set(MISTRAL_KEY_PREFIXES) & set(TRANSFORMER_KEY_PREFIXES)

    def test_a_language_model_prefixed_hint_reaches_its_layer(self) -> None:
        """Multimodal Mistral3 redistributions prefix every key with `language_model.`.

        The sd loses that prefix in `_strip_known_prefixes`; the header does not. Passing only the
        generic tuple leaves the hint keyed on `language_model.model.layers.0.…` while the layer is
        `model.layers.0.…`, and the flag is dropped in silence.
        """
        sd = _strip_known_prefixes(_quantized("language_model.model.layers.0.self_attn.q_proj"))
        header = {"language_model.model.layers.0.self_attn.q_proj": {"full_precision_matrix_mult": True}}

        hints = strip_layer_path_prefix(header, prefixes=(*MISTRAL_KEY_PREFIXES, *TRANSFORMER_KEY_PREFIXES))
        layers = extract_fp8_scaled_layers(sd, layer_hints=hints)

        assert set(layers) == {"model.layers.0.self_attn.q_proj"}
        assert layers["model.layers.0.self_attn.q_proj"].full_precision_matmul is True

    def test_the_generic_tuple_alone_would_have_lost_the_flag(self) -> None:
        """Pins the failure mode itself, so the fix cannot be reverted unnoticed."""
        sd = _strip_known_prefixes(_quantized("language_model.model.layers.0.self_attn.q_proj"))
        header = {"language_model.model.layers.0.self_attn.q_proj": {"full_precision_matrix_mult": True}}

        hints = strip_layer_path_prefix(header)
        layers = extract_fp8_scaled_layers(sd, layer_hints=hints)

        assert layers["model.layers.0.self_attn.q_proj"].full_precision_matmul is False


class TestFluxHintPrefixes:
    """FLUX.1's hint plumbing had no test at all.

    ComfyUI FLUX.1 redistributions prefix their keys with `model.diffusion_model.`, which the
    bundle conversion strips before the scales are read.
    """

    def test_a_prefixed_hint_lands_on_the_renamed_layer(self) -> None:
        sd = {
            k[len("model.diffusion_model.") :]: v
            for k, v in _quantized("model.diffusion_model.double_blocks.0.img_attn.qkv").items()
        }
        header = {"model.diffusion_model.double_blocks.0.img_attn.qkv": {"full_precision_matrix_mult": True}}

        layers = extract_fp8_scaled_layers(sd, layer_hints=strip_layer_path_prefix(header))

        assert set(layers) == {"double_blocks.0.img_attn.qkv"}
        assert layers["double_blocks.0.img_attn.qkv"].full_precision_matmul is True

    def test_an_unprefixed_hint_is_passed_through_unchanged(self) -> None:
        """A partially-prefixed header must not have its plain names dropped."""
        sd = _quantized("double_blocks.0.img_attn.qkv")
        header = {"double_blocks.0.img_attn.qkv": {"full_precision_matrix_mult": True}}

        layers = extract_fp8_scaled_layers(sd, layer_hints=strip_layer_path_prefix(header))

        assert layers["double_blocks.0.img_attn.qkv"].full_precision_matmul is True
