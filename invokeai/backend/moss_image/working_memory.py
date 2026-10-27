"""The working memory one MOSS-Image transformer forward needs on its compute device.

The fitted part is the activations of one sample's single-stream forward, which scale with the tokens it attends over:
the image's (one per 16x16 pixels) and the prompt's. The coefficient and the points it was measured on ship in
`invokeai/backend/working_memory/data/moss_image.json`.
"""

from collections.abc import Mapping
from typing import Optional, cast

import torch

from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageTransformer
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
    fit leaves it out (`fit.line_values`): the slope is hand-set until a fitted base ships with it."""
    if not all(isinstance(features.get(name), int) for name in ("height", "width", "text_tokens")):
        return None
    on_top = [f"{name}={features[name]}" for name in ("score_matrix_bytes", "dequant_bytes") if features.get(name)]
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
    device: torch.device,
    dtype: torch.dtype,
) -> WorkingMemoryEstimate:
    """What one forward of `transformer` at `height` x `width` over `text_tokens` prompt tokens needs on `device`.

    The positive and negative passes run one after the other, so `text_tokens` is the longer prompt's length, not the
    sum. `transformer` is read unlocked, before the lock this reservation applies to.
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
    }
    return MOSS_IMAGE_DENOISE_SPEC.estimate(features, device)
