"""Regression tests for the PiD key-space normalisation shared by identification and loading.

Identification and the loader used to carry a copy each, and the copies had diverged: only the
loader dropped the distill-only submodules. That drift is silent in the dangerous direction —
identification accepting a checkpoint the loader then refuses — so these tests pin that both
consumers see exactly the same keys.
"""

import torch

from invokeai.backend.pid.state_dict_utils import has_net_prefix, pid_net_shapes, strip_net_prefix


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

    def test_a_bare_checkpoints_non_string_keys_survive(self) -> None:
        """The other half of the pass-through, and the reason the result is not `dict[str, T]`.

        A prefixed checkpoint has its non-string keys filtered out by the rename above; a bare one is
        returned as-is, so callers get whatever the `.pth` was pickled with. Dropping them would hide
        a malformed file from the checks meant to catch it — but it does put the burden on both
        consumers, neither of which may assume the key type: identification sorts its key reports with
        `key=str`, and `load_pid_decoder` rejects non-strings before torch can trip over them.
        """
        sd = {"lq_proj.a": torch.zeros(1), 0: torch.zeros(1)}
        assert set(strip_net_prefix(sd)) == {"lq_proj.a", 0}
        assert set(pid_net_shapes(sd)) == {"lq_proj.a", 0}


class TestPidNetShapes:
    def test_matches_strip_net_prefix_exactly(self) -> None:
        """The whole point of the shared module: identification's view and the loader's view of a
        checkpoint's key space cannot differ."""
        sd = {
            "net.lq_proj.a": torch.zeros(1),
            "net_ema.lq_proj.a": torch.zeros(1),
            "discriminator.y": torch.zeros(1),
            "blocks.0.weight": torch.zeros(1),
        }
        assert pid_net_shapes(sd).keys() == strip_net_prefix(sd).keys()

    def test_carries_shapes(self) -> None:
        assert pid_net_shapes({"net.lq_proj.latent_proj.0.weight": torch.zeros(512, 16, 3, 3)}) == {
            "lq_proj.latent_proj.0.weight": (512, 16, 3, 3)
        }
        # 0-dim parameters are real: PidNet's gate modules carry scalar `log_alpha`s.
        assert pid_net_shapes({"net.lq_proj.gate_modules.0.log_alpha": torch.zeros(())}) == {
            "lq_proj.gate_modules.0.log_alpha": ()
        }

    def test_a_value_with_no_shape_is_reported_as_none(self) -> None:
        """A checkpoint can hold arbitrary objects. One sitting under a PidNet parameter name is
        malformed, which identification reports — not an AttributeError mid-probe."""
        assert pid_net_shapes({"net.lq_proj.a": "not a tensor"}) == {"lq_proj.a": None}

    def test_the_ema_copy_cannot_be_read_as_the_students_weight(self) -> None:
        """Identification used to find the discriminator weight by *suffix* over the raw dict, so a
        checkpoint carrying only the EMA copy had `net_ema.lq_proj.latent_proj.0.weight` read as the
        student's. Looking it up by name in the normalised map cannot do that."""
        sd = {"net.blocks.0.weight": torch.zeros(1), "net_ema.lq_proj.latent_proj.0.weight": torch.zeros(512, 16, 3, 3)}
        assert "lq_proj.latent_proj.0.weight" not in pid_net_shapes(sd)


class TestHasNetPrefix:
    def test_detects_the_distill_serialisation(self) -> None:
        assert has_net_prefix({"net.lq_proj.a": torch.zeros(1)})
        assert not has_net_prefix({"lq_proj.a": torch.zeros(1)})
        # `net_ema.` is not the student prefix — the dot matters.
        assert not has_net_prefix({"net_ema.lq_proj.a": torch.zeros(1)})
