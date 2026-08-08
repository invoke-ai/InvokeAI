"""Regression tests for the PiD key-space normalisation shared by identification and loading.

Identification and the loader used to carry a copy each, and the copies had diverged: only the
loader dropped the distill-only submodules. That drift is silent in the dangerous direction —
identification accepting a checkpoint the loader then refuses — so these tests pin that both
consumers see exactly the same keys.
"""

import torch

from invokeai.backend.pid.state_dict_utils import has_net_prefix, pid_net_keys, strip_net_prefix


class TestStripNetPrefix:
    def test_student_prefix_is_removed(self) -> None:
        sd = {"net.lq_proj.latent_proj.0.weight": torch.zeros(1), "net.blocks.0.weight": torch.zeros(1)}
        assert set(strip_net_prefix(sd)) == {"lq_proj.latent_proj.0.weight", "blocks.0.weight"}

    def test_distill_only_submodules_are_dropped(self) -> None:
        """`net_ema.*` shadows PidNet's own parameter names, so it must be dropped, not renamed."""
        sd = {
            "net.lq_proj.a": torch.zeros(1),
            "net_ema.lq_proj.a": torch.zeros(1),
            "fake_score.x": torch.zeros(1),
            "discriminator.y": torch.zeros(1),
        }
        assert set(strip_net_prefix(sd)) == {"lq_proj.a"}

    def test_a_bare_pid_net_checkpoint_is_untouched(self) -> None:
        """Without the prefix there is no evidence this is a distill serialisation, so nothing is
        filtered — a stray key should reach the loader's "unexpected keys" check, not vanish."""
        sd = {"lq_proj.a": torch.zeros(1), "discriminator.y": torch.zeros(1)}
        assert strip_net_prefix(sd) is sd

    def test_values_survive_the_rename(self) -> None:
        tensor = torch.arange(3.0)
        assert torch.equal(strip_net_prefix({"net.lq_proj.a": tensor})["lq_proj.a"], tensor)

    def test_non_string_keys_are_dropped_rather_than_crashing(self) -> None:
        # `mod.load_state_dict()` is typed `dict[str | int, Any]`; a torch checkpoint can carry
        # non-string keys, and neither consumer should raise on one.
        assert set(strip_net_prefix({"net.lq_proj.a": torch.zeros(1), 0: torch.zeros(1)})) == {"lq_proj.a"}


class TestPidNetKeys:
    def test_matches_strip_net_prefix_exactly(self) -> None:
        """The whole point of the shared module: identification's view and the loader's view of a
        checkpoint's key space cannot differ."""
        sd = {
            "net.lq_proj.a": torch.zeros(1),
            "net_ema.lq_proj.a": torch.zeros(1),
            "discriminator.y": torch.zeros(1),
            "blocks.0.weight": torch.zeros(1),
        }
        assert pid_net_keys(sd) == set(strip_net_prefix(sd))


class TestHasNetPrefix:
    def test_detects_the_distill_serialisation(self) -> None:
        assert has_net_prefix({"net.lq_proj.a": torch.zeros(1)})
        assert not has_net_prefix({"lq_proj.a": torch.zeros(1)})
        # `net_ema.` is not the student prefix — the dot matters.
        assert not has_net_prefix({"net_ema.lq_proj.a": torch.zeros(1)})
