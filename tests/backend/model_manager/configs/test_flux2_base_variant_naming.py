"""Which FLUX.2 Klein variant a file is identified as when only its name can say.

Klein 9B and Klein 9B Base ship identical architectures and identical keys, so the state dict
cannot tell them apart and the name is the whole decision. It is not a cosmetic one: the two carry
different default step counts (4 against 28), so identifying a distilled model as Base generates at
seven times the cost, and the reverse generates at four steps the model was never trained for.
"""

from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Flux2_Config
from invokeai.backend.model_manager.taxonomy import Flux2VariantType

# `context_in_dim` is what names the size: 3 x Qwen3-8B's hidden 4096 for the 9B, 3 x Qwen3-4B's
# 2560 for the 4B. Nothing in either file says which of the two 9B releases it is.
_CONTEXT_DIM = {Flux2VariantType.Klein9B: 12288, Flux2VariantType.Klein4B: 7680}


def _variant_of(name: str, size: Flux2VariantType) -> Flux2VariantType:
    mod = SimpleNamespace(
        name=name,
        load_state_dict=lambda: {"txt_in.weight": torch.zeros(1, _CONTEXT_DIM[size])},
    )
    return Main_Checkpoint_Flux2_Config._get_variant_or_raise(mod)


@pytest.mark.parametrize(
    "name",
    [
        "flux-2-klein-9b-base",
        "FLUX.2-klein-base-9B",
        "flux2_klein_9b_BASE_fp8",
        # A digit is not a word boundary the way a letter is: this is still the word "base".
        "flux2-klein-base9b",
        # The claim spelled out. It contains "distilled", so reading that word first gets it backwards.
        "flux-2-klein-9b-undistilled",
        "flux-2-klein-9b-base-non-distilled",
    ],
)
def test_a_name_that_says_base_identifies_the_undistilled_release(name: str) -> None:
    assert _variant_of(name, Flux2VariantType.Klein9B) is Flux2VariantType.Klein9BBase


@pytest.mark.parametrize(
    "name",
    [
        # The case that reached users. Installed by path this repack is
        # `flux-2-klein-9b-int8-convrot` and identified correctly; installed by repo id the name
        # becomes the repo's, which says what the weights were distilled *from*.
        "Klein9b-Distilled-Base-INT8-Convrot",
        "flux-2-klein-9b-distilled",
        # A substring is not a word. Neither of these says anything about the variant.
        "klein-9b-database-export",
        "klein-9b-basement-run",
        # Names its method, not its variant.
        "klein-9b-distillation-run",
    ],
)
def test_a_name_that_does_not_claim_the_base_variant_stays_distilled(name: str) -> None:
    assert _variant_of(name, Flux2VariantType.Klein9B) is Flux2VariantType.Klein9B


def test_the_4b_pair_is_decided_the_same_way() -> None:
    """Both sizes have a Base release and both read the same name, so a fix to one that misses the
    other would leave half the problem in place."""
    assert _variant_of("flux-2-klein-4b-base", Flux2VariantType.Klein4B) is Flux2VariantType.Klein4BBase
    assert _variant_of("Klein4b-Distilled-Base", Flux2VariantType.Klein4B) is Flux2VariantType.Klein4B
