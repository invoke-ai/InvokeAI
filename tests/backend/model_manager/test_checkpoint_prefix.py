"""The shared wrapper-prefix strip, which nine places across six modules had a copy of.

What is worth pinning here is not that a prefix comes off -- it is the three decisions the copies
made silently and differently: which prefix wins when two match, what happens to keys that were not
under it, and whether a bare checkpoint may have keys dropped at all.
"""

import pytest
import torch

from invokeai.backend.model_manager.checkpoint_prefix import COMFYUI_KEY_PREFIXES, CheckpointPrefix

#: Spelled out rather than taken from `COMFYUI_KEY_PREFIXES`: a cell parametrized over the constant
#: it is meant to pin loses the case along with the entry, and passes.
BOTH_COMFYUI_SPELLINGS = ("model.diffusion_model.", "diffusion_model.")


@pytest.mark.parametrize("prefix", BOTH_COMFYUI_SPELLINGS)
def test_every_spelling_the_config_probes_accept_is_actually_stripped(prefix: str) -> None:
    """Both reach real files: the config probes accept either, so a file carrying the bare
    `diffusion_model.` form installs and is routed to a loader. Were it missing from the constant,
    that loader would hand the model a state dict with every key still wrapped."""
    sd = {f"{prefix}blocks.0.attn.weight": torch.zeros(1)}

    detected = CheckpointPrefix.detect(sd, COMFYUI_KEY_PREFIXES)

    assert detected.prefix == prefix
    assert list(detected.strip(sd)) == ["blocks.0.attn.weight"]


def test_detection_takes_the_first_prefix_in_list_order() -> None:
    """The two ComfyUI spellings are mutually exclusive as prefixes -- `model.diffusion_model.x` does
    not start with `diffusion_model.` -- so their order in the constant cannot be observed, and a
    cell using them would pin nothing. What *is* worth pinning is the rule for a list where one entry
    is a prefix of another, because that is the day the order starts to matter: the first match in
    list order wins, not the longest.
    """
    sd = {"a.x": torch.zeros(1), "a.b.y": torch.zeros(1)}

    assert CheckpointPrefix.detect(sd, ("a.b.", "a.")).prefix == "a.b."
    assert CheckpointPrefix.detect(sd, ("a.", "a.b.")).prefix == "a."


def test_a_bare_checkpoint_is_returned_as_it_came() -> None:
    sd = {"blocks.0.attn.weight": torch.zeros(1)}

    assert CheckpointPrefix.detect(sd, COMFYUI_KEY_PREFIXES).strip(sd) is sd


def test_keys_outside_the_prefix_are_kept_by_default() -> None:
    """Eleven of the twelve call sites keep them, and filter later by rules of their own if at
    all. Only Anima asks for them to be dropped."""
    sd = {"net.blocks.0.weight": torch.zeros(1), "first_stage_model.encoder.weight": torch.zeros(1)}

    stripped = CheckpointPrefix.detect(sd, ("net.",)).strip(sd)

    assert sorted(stripped) == ["blocks.0.weight", "first_stage_model.encoder.weight"]


def test_an_all_in_one_export_can_drop_what_is_not_its_model() -> None:
    """The one caller that asks for it (Anima) reads bundles where the VAE and text encoder sit
    beside the transformer under namespaces of their own. Those are not its model's weights."""
    sd = {"net.blocks.0.weight": torch.zeros(1), "first_stage_model.encoder.weight": torch.zeros(1)}

    stripped = CheckpointPrefix.detect(sd, ("net.",)).strip(sd, drop_foreign=True)

    assert sorted(stripped) == ["blocks.0.weight"]


def test_a_bare_checkpoint_keeps_its_stray_keys_even_when_foreign_ones_may_be_dropped() -> None:
    """With no prefix there is nothing to tell a foreign key from an own one, so dropping would be
    guessing -- and it would hide the stray key from the unexpected-key checks that report it."""
    sd = {"blocks.0.weight": torch.zeros(1), "discriminator.weight": torch.zeros(1)}

    stripped = CheckpointPrefix.detect(sd, ("net.",)).strip(sd, drop_foreign=True)

    assert sorted(stripped) == ["blocks.0.weight", "discriminator.weight"]


def test_a_non_string_key_survives_detection_and_the_strip() -> None:
    """Checkpoints have arrived with non-string keys; every copy of this code guarded for it, so the
    guard is load-bearing rather than defensive."""
    sd = {7: torch.zeros(1), "model.diffusion_model.blocks.0.weight": torch.zeros(1)}

    stripped = CheckpointPrefix.detect(sd, COMFYUI_KEY_PREFIXES).strip(sd)

    assert sorted(map(str, stripped)) == ["7", "blocks.0.weight"]
