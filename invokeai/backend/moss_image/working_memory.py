"""The working memory one MOSS-Image transformer forward needs on its compute device.

The fitted part is the activations of one sample's single-stream forward, which scale with the tokens it attends over:
the image's (one per 16x16 pixels) and the prompt's. The coefficient and the points it was measured on ship in
`invokeai/backend/working_memory/data/moss_image.json`.
"""

from collections.abc import Mapping, Sequence
from typing import Optional, cast

import torch

from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageTransformer
from invokeai.backend.patches.lora_conversions.moss_image_lora_constants import MOSS_IMAGE_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.util.attention import sdpa_score_matrix_bytes
from invokeai.backend.working_memory.estimate import FeatureValue, WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.fit import Line
from invokeai.backend.working_memory.spec import EstimatorSpec, register

# One transformer token per 2x2 patch of the FLUX.2 VAE's 8x latent.
MOSS_IMAGE_TOKEN_SCALE = 16


def _tokens(features: Mapping[str, FeatureValue]) -> int:
    image_tokens = (cast(int, features["height"]) // MOSS_IMAGE_TOKEN_SCALE) * (
        cast(int, features["width"]) // MOSS_IMAGE_TOKEN_SCALE
    )
    return image_tokens + cast(int, features["text_tokens"])


def _line(features: Mapping[str, FeatureValue]) -> Optional[Line]:
    """The forward's line, `token` over the attended tokens. Only the plain forward fits it: a fused SDPA kernel and a
    dense build, every analytic term zero. The label names the size, the prompt's tokens and each analytic term that
    is not zero, in bytes.

    The shipped entry carries the slope alone (`k_token`), no base, so the line is not one it carries and the shipped
    fit leaves it out (`fit.hand_set_values`): the slope is hand-set until a fitted base ships with it."""
    if not all(isinstance(features.get(name), int) for name in ("height", "width", "text_tokens")):
        return None
    on_top = [
        f"{name}={features[name]}"
        for name in ("score_matrix_bytes", "dequant_bytes", "lora_sidecar_bytes")
        if features.get(name)
    ]
    shape = f"{features['height']}x{features['width']}-text{features['text_tokens']}"
    return Line("token", _tokens(features), not on_top, "-".join((shape, *on_top)))


def _denoise_terms(features: Mapping[str, FeatureValue], values: Mapping[str, float]) -> tuple[WorkingMemoryTerm, ...]:
    tokens = _tokens(features)
    return (
        WorkingMemoryTerm.fitted("activations", "k_token", tokens, values),
        # Analytic: zero on a fused SDPA kernel, the materialized score matrices where the build has none.
        WorkingMemoryTerm("attention_scores", cast(int, features["score_matrix_bytes"]), True),
        # Analytic: the dequantized copy of the largest packed Linear of a quantized build, zero for a dense one.
        WorkingMemoryTerm("dequant_transient", cast(int, features["dequant_bytes"]), True),
        # Analytic: the LoRAs' tensors and the update their sidecars form per forward. Absent from the points measured
        # before LoRAs ran on MOSS-Image, all of them forwards without one.
        WorkingMemoryTerm("lora_sidecar", cast(int, features.get("lora_sidecar_bytes", 0)), True),
    )


MOSS_IMAGE_DENOISE_SPEC = register(
    EstimatorSpec(
        id="moss_image.denoise",
        version=1,
        coefficients=frozenset({"k_token"}),
        # Not keyed by platform: the fitted term is the activations of GEMMs and fused attention, not convolution
        # workspaces, so the convolution backend does not move it. A backend without a fused SDPA kernel for these
        # shapes is what does, and the analytic score-matrix term prices that.
        platform_keys=(),
        terms=_denoise_terms,
        line=_line,
    )
)


def estimate_moss_image_denoise_working_memory(
    transformer: MOSSImageTransformer,
    *,
    height: int,
    width: int,
    text_tokens: int,
    loras: Sequence[ModelPatchRaw] = (),
    model_format: ModelFormat,
    device: torch.device,
    dtype: torch.dtype,
) -> WorkingMemoryEstimate:
    """What one forward of `transformer` at `height` x `width` over `text_tokens` prompt tokens needs on `device`.

    The positive and negative passes run one after the other, so `text_tokens` is the longer prompt's length, not the
    sum. `transformer` is read unlocked, before the lock this reservation applies to. `loras` are the LoRA patches the
    forward applies, with the transformer's `model_format`, which decides whether they run as sidecars.
    """
    num_heads = int(transformer.config.num_heads)
    image_tokens = (height // MOSS_IMAGE_TOKEN_SCALE) * (width // MOSS_IMAGE_TOKEN_SCALE)
    features: dict[str, FeatureValue] = {
        "height": height,
        "width": width,
        "text_tokens": text_tokens,
        "score_matrix_bytes": sdpa_score_matrix_bytes(
            device=device,
            dtype=dtype,
            num_heads=num_heads,
            head_dim=int(transformer.config.dim) // num_heads,
            seq_len=image_tokens + text_tokens,
        ),
        "dequant_bytes": peak_dequant_transient_bytes(transformer, dtype),
        "lora_sidecar_bytes": _lora_sidecar_bytes(transformer, loras, model_format, image_tokens + text_tokens, dtype),
    }
    return MOSS_IMAGE_DENOISE_SPEC.estimate(features, device)


# The Linears that read the timestep embedding: one row per sample, not one per token.
_TIMESTEP_MODULES = ("time_embedding.0", "time_embedding.2", "shared_block_modulation", "video_head_modulation")


def _lora_sidecar_bytes(
    transformer: MOSSImageTransformer,
    loras: Sequence[ModelPatchRaw],
    model_format: ModelFormat,
    tokens: int,
    dtype: torch.dtype,
) -> int:
    if not loras:
        return 0
    # Imported here: the patch layers import the model manager's loaders, and `import_estimator_modules` imports this
    # module on its own.
    from invokeai.backend.patches.sidecar_working_memory import lora_sidecar_bytes

    return lora_sidecar_bytes(
        loras,
        transformer,
        model_format,
        prefix=MOSS_IMAGE_LORA_TRANSFORMER_PREFIX,
        rows=tokens,
        per_sample_rows=dict.fromkeys(_TIMESTEP_MODULES, 1),
        dtype=dtype,
    )
