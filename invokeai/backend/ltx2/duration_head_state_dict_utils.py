"""Recognize an LTX-2 duration head from its state dict.

The head is tiny (19 tensors, ~3.6 MB) and ships in diffusers layout, so there is no key
conversion -- only the question of whether a given file *is* one. What identifies it is the pair of
modality input projections feeding a shared attention pooler: the head is the only LTX-2 component
that projects both connector widths (video 4096, audio 2048) into one pooled space and regresses a
single scalar from it.
"""

from collections.abc import Mapping
from typing import Any

# Every tensor the head carries, and nothing else. Matched as an exact set rather than as a marker
# or a subset because the loader that follows is strict in both directions: a file with a tensor
# this architecture does not have would identify, install, and then fail at the first generation
# with "Unexpected keys loading LTX-2 duration head" -- after the user has waited for a download and
# picked the model in a panel. Demanding the whole set turns that into a refusal at install time,
# which is where a "this is not a head this version can read" answer belongs. A missing tensor is
# refused for the opposite reason: it would load with the remainder left uninitialised and answer
# every prompt with a confident, meaningless duration.
_EXPECTED_KEYS = frozenset(
    {
        "video_input_proj.weight",
        "video_input_proj.bias",
        "video_modality_emb",
        "audio_input_proj.weight",
        "audio_input_proj.bias",
        "audio_modality_emb",
        "attention_pooler.query_tokens",
        "attention_pooler.to_q.weight",
        "attention_pooler.to_q.bias",
        "attention_pooler.to_k.weight",
        "attention_pooler.to_k.bias",
        "attention_pooler.to_v.weight",
        "attention_pooler.to_v.bias",
        "attention_pooler.to_out.weight",
        "attention_pooler.to_out.bias",
        "mlp_hidden.weight",
        "mlp_hidden.bias",
        "mlp_out.weight",
        "mlp_out.bias",
    }
)


# The widths the loader builds the head at (diffusers' `LTX2DurationHead` defaults: pooled 256,
# video connector 4096, audio connector 2048). The file carries no config, so a head published at
# other widths has the same keys and would only fail at the first generation's strict load; checking
# the shapes that fix those widths moves that refusal to install time too. (The pooler's head count
# shows in no tensor shape and cannot be checked here.)
_EXPECTED_SHAPES = {
    "video_input_proj.weight": (256, 4096),
    "audio_input_proj.weight": (256, 2048),
    "mlp_out.weight": (1, 256),
}


def is_state_dict_likely_ltx2_duration_head(state_dict: Mapping[str, Any]) -> bool:
    """True if this state dict is an LTX-2 duration head in its published (diffusers) layout."""
    return set(state_dict.keys()) == _EXPECTED_KEYS and all(
        tuple(state_dict[key].shape) == shape for key, shape in _EXPECTED_SHAPES.items()
    )
