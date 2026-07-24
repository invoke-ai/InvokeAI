"""Canonical FLUX.2 transformer dimensions per variant, and reverse lookups.

Single source of truth for the geometry that distinguishes the FLUX.2 variants, so the
identification code in ``main.py`` (checkpoint + diffusers) and ``lora.py`` cannot drift
apart. Previously the same 7680/12288/15360 (and 2560/4096/5120, 3072/4096/6144) literals
were hand-maintained in four places.

Only the three *distilled* variants are represented here. Base variants (Klein4BBase /
Klein9BBase) share architecture with their distilled counterparts and are indistinguishable
from geometry — callers detect them via a filename heuristic and upgrade the returned
distilled variant themselves.

Dimensions:
- ``context_in_dim`` = ``joint_attention_dim`` = 3 × text-encoder hidden_size (context embedder)
- ``vec_in_dim``     = text-encoder hidden_size (vector embedder)
- ``hidden_size``    = transformer hidden size (attention projections)
"""

from invokeai.backend.model_manager.taxonomy import Flux2VariantType

# context_in_dim (= joint_attention_dim) per distilled variant.
_CONTEXT_IN_DIM: dict[Flux2VariantType, int] = {
    Flux2VariantType.Klein4B: 7680,  # 3 × Qwen3-4B 2560
    Flux2VariantType.Klein9B: 12288,  # 3 × Qwen3-8B 4096
    Flux2VariantType.Dev: 15360,  # 3 × Mistral Small 3.1 5120
}

# vec_in_dim (text-encoder hidden_size) per distilled variant.
_VEC_IN_DIM: dict[Flux2VariantType, int] = {
    Flux2VariantType.Klein4B: 2560,
    Flux2VariantType.Klein9B: 4096,
    Flux2VariantType.Dev: 5120,
}

# transformer hidden_size per distilled variant.
_HIDDEN_SIZE: dict[Flux2VariantType, int] = {
    Flux2VariantType.Klein4B: 3072,
    Flux2VariantType.Klein9B: 4096,
    Flux2VariantType.Dev: 6144,  # 48 heads × 128 head_dim
}

# All recognized FLUX.2 context_in_dim values (used as a cheap "is this FLUX.2?" check).
FLUX2_CONTEXT_IN_DIMS: frozenset[int] = frozenset(_CONTEXT_IN_DIM.values())

_CONTEXT_IN_DIM_TO_VARIANT: dict[int, Flux2VariantType] = {dim: v for v, dim in _CONTEXT_IN_DIM.items()}
_VEC_IN_DIM_TO_VARIANT: dict[int, Flux2VariantType] = {dim: v for v, dim in _VEC_IN_DIM.items()}
_HIDDEN_SIZE_TO_VARIANT: dict[int, Flux2VariantType] = {dim: v for v, dim in _HIDDEN_SIZE.items()}


def flux2_variant_from_context_dim(dim: int) -> Flux2VariantType | None:
    """Return the distilled FLUX.2 variant for a context_in_dim, or ``None`` if unrecognized."""
    return _CONTEXT_IN_DIM_TO_VARIANT.get(dim)


def flux2_variant_from_vec_dim(dim: int) -> Flux2VariantType | None:
    """Return the distilled FLUX.2 variant for a vec_in_dim, or ``None`` if unrecognized."""
    return _VEC_IN_DIM_TO_VARIANT.get(dim)


def flux2_variant_from_hidden_size(dim: int) -> Flux2VariantType | None:
    """Return the distilled FLUX.2 variant for a transformer hidden_size, or ``None`` if unrecognized."""
    return _HIDDEN_SIZE_TO_VARIANT.get(dim)
