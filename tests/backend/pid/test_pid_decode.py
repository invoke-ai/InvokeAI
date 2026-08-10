"""Regression tests for the PiD distill schedule, decoder/base validation and checkpoint completeness."""

import math

import pytest
import torch

from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.pid.decode import (
    _PER_BACKBONE,
    BACKBONE_DISCRIMINATOR_KEY,
    _get_t_list,
    assert_pid_decoder_matches_base,
    load_pid_decoder,
    required_pid_net_shapes,
)

_CPU = torch.device("cpu")


@pytest.mark.parametrize("num_steps", [1, 2, 3, 4])
def test_student_schedule_is_strictly_decreasing(num_steps: int) -> None:
    """Every permitted step count yields a strictly decreasing schedule with no duplicate timesteps.

    The student schedule has only four transitions; sub-sampling to >4 steps rounds distinct indices
    onto the same point and produces duplicates (e.g. 5 steps → [.999, .866, .634, .634, .342, 0]),
    which is why the public range is capped at 4.
    """
    t = _get_t_list(_CPU, num_steps=num_steps).tolist()
    assert len(t) == num_steps + 1
    assert t[-1] == pytest.approx(0.0, abs=1e-6)
    assert all(a > b for a, b in zip(t[:-1], t[1:], strict=True)), t
    assert len(set(t)) == len(t)


def test_default_schedule_matches_four_steps() -> None:
    assert _get_t_list(_CPU).tolist() == _get_t_list(_CPU, num_steps=4).tolist()


def test_out_of_range_step_count_trips_the_safety_net() -> None:
    """If an invalid count ever bypassed the field cap, the guard raises. Uses ValueError (not assert)
    so it still fires under `python -O`, where assertions are stripped."""
    with pytest.raises(ValueError, match="strictly decreasing"):
        _get_t_list(_CPU, num_steps=5)


def test_matching_decoder_base_is_accepted() -> None:
    assert_pid_decoder_matches_base(BaseModelType.Flux, BaseModelType.Flux, node_title="FLUX PiD Decode")


def test_mismatched_decoder_base_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires a"):
        assert_pid_decoder_matches_base(
            BaseModelType.StableDiffusion3, BaseModelType.Flux, node_title="FLUX PiD Decode"
        )


def test_z_image_node_accepts_flux_decoder() -> None:
    """Z-Image reuses the FLUX decoder, so its node passes node_base=FLUX and accepts a FLUX decoder."""
    assert_pid_decoder_matches_base(BaseModelType.Flux, BaseModelType.Flux, node_title="Z-Image PiD Decode")


class TestRequiredPidNetShapes:
    """The contract model identification holds a checkpoint to, probed from a meta `PidNet`."""

    def test_covers_the_whole_network(self) -> None:
        shapes = required_pid_net_shapes()
        lq = {k for k in shapes if k.startswith("lq_proj.")}
        # The LQ projection is a small fraction of the net: checking only it (as identification once
        # did) leaves the 385 backbone weights to be discovered missing at load time.
        assert len(lq) == 71
        assert len(shapes) - len(lq) == 385
        # Latent projection conv, one output head + one gate per injection point (patch_depth 14 /
        # lq_interval 2 = 7). The image branch is disabled (lq_in_channels=0), so it must not appear.
        assert shapes[BACKBONE_DISCRIMINATOR_KEY] == (512, 16, 3, 3)
        assert sum(k.endswith(".weight") and ".output_heads." in k for k in lq) == 7
        assert sum(k.endswith(".log_alpha") for k in lq) == 7
        assert not any(".image_conv." in k for k in lq)

    def test_one_contract_holds_for_every_backbone(self) -> None:
        """Identification validates a checkpoint *before* it can know the backbone, so it probes one
        canonical backbone and applies that contract to all five. This pins what makes that sound:
        the key names never vary, and `BACKBONE_DISCRIMINATOR_KEY` is the only shape that does. A
        backbone that ever varies anything else has to fail here, while it is being added, rather
        than silently weakening the install-time check."""
        canonical = required_pid_net_shapes()
        for backbone in _PER_BACKBONE:
            per_backbone = required_pid_net_shapes(backbone)
            assert per_backbone.keys() == canonical.keys(), backbone
            differing = {k for k in canonical if per_backbone[k] != canonical[k]}
            assert differing <= {BACKBONE_DISCRIMINATOR_KEY}, (backbone, differing)
            # And it varies in exactly the dimension identification reads the backbone from.
            assert per_backbone[BACKBONE_DISCRIMINATOR_KEY][1] == _PER_BACKBONE[backbone]["lq_latent_channels"]

    def test_unsupported_backbone_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            required_pid_net_shapes(BaseModelType.StableDiffusion1)

    def test_probing_does_not_consume_the_global_rng(self) -> None:
        """The probe builds a real PidNet only to read parameter names and shapes. Constructing one
        normally runs every `reset_parameters()`, which draws from the global CPU RNG — during
        *model identification*. Later unseeded randomness would then depend on how many candidate
        files were probed, i.e. on install order."""
        torch.manual_seed(0)
        control = torch.rand(4)

        torch.manual_seed(0)
        required_pid_net_shapes.cache_clear()
        required_pid_net_shapes()
        after = torch.rand(4)

        assert torch.equal(control, after), "identification must not advance the global RNG"

    def test_the_contract_is_too_large_to_build_for_real(self) -> None:
        """Why the probe must stay on the meta device: the network it describes is ~5.5 GB in
        float32. Model identification builds one for every PiD-looking file it is handed."""
        elements = sum(math.prod(shape) for shape in required_pid_net_shapes().values())
        assert elements * 4 > 5 * 1024**3


class TestLoadPidDecoderRejectsPartialCheckpoints:
    """Loaders run under `skip_torch_weight_init()`, so any key the checkpoint omits stays uninitialised
    memory. `load_pid_decoder` must therefore refuse a partial state dict instead of decoding garbage."""

    class _TinyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lq_proj = torch.nn.Linear(2, 2)
            self.blocks = torch.nn.Linear(2, 2)

    @pytest.fixture
    def tiny_net(self, monkeypatch: pytest.MonkeyPatch) -> "TestLoadPidDecoderRejectsPartialCheckpoints._TinyNet":
        """Stand in for the (multi-GB) real PidNet — only load_state_dict's bookkeeping is under test."""
        net = self._TinyNet()
        monkeypatch.setattr("invokeai.backend.pid.decode.build_pid_net", lambda backbone: net)
        return net

    def test_complete_state_dict_loads(self, tiny_net: torch.nn.Module) -> None:
        assert load_pid_decoder(dict(tiny_net.state_dict()), BaseModelType.Flux) is tiny_net

    def test_missing_lq_keys_are_rejected(self, tiny_net: torch.nn.Module) -> None:
        sd = {k: v for k, v in tiny_net.state_dict().items() if not k.startswith("lq_proj.")}
        with pytest.raises(RuntimeError, match="LQ projection is incomplete"):
            load_pid_decoder(sd, BaseModelType.Flux)

    def test_missing_backbone_keys_are_rejected(self, tiny_net: torch.nn.Module) -> None:
        sd = {k: v for k, v in tiny_net.state_dict().items() if k != "blocks.weight"}
        with pytest.raises(RuntimeError, match="missing 1 keys"):
            load_pid_decoder(sd, BaseModelType.Flux)

    def test_unexpected_keys_are_rejected(self, tiny_net: torch.nn.Module) -> None:
        sd = dict(tiny_net.state_dict()) | {"not_a_pid_key": torch.zeros(1)}
        with pytest.raises(RuntimeError, match="unexpected keys"):
            load_pid_decoder(sd, BaseModelType.Flux)
