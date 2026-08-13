"""Regression tests for the PiD distill schedule, decoder/base validation and checkpoint completeness."""

import math
from typing import Any
from unittest.mock import patch

import pytest
import torch

from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.pid import decode as pid_decode_module
from invokeai.backend.pid._src.networks.pid_net import PidNet
from invokeai.backend.pid._src.networks.pixeldit_official import PiTBlock
from invokeai.backend.pid.decode import (
    _PER_BACKBONE,
    BACKBONE_DISCRIMINATOR_KEY,
    PiDDecodeConfig,
    PiDDecoder,
    _get_t_list,
    _student_sample_loop,
    _velocity_to_x0,
    assert_pid_decoder_matches_base,
    estimate_pid_decode_working_memory,
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


@pytest.mark.parametrize(
    ("x_dtype", "net_output_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_velocity_to_x0_uses_float32_math_and_preserves_input_dtype(
    x_dtype: torch.dtype, net_output_dtype: torch.dtype
) -> None:
    x_t = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]], dtype=x_dtype)
    net_output = torch.tensor([[[[0.5, -0.25], [0.125, -0.0625]]]], dtype=net_output_dtype)
    timestep = torch.tensor([0.634], dtype=torch.float32)
    expected = (x_t.float() - timestep.view(1, 1, 1, 1) * net_output.float()).to(x_dtype)

    # A float64 intermediate doubles memory for each full-resolution sampler tensor. PiD already
    # predicts under bf16 autocast, so perform this update in float32 without calling Tensor.double().
    with patch.object(torch.Tensor, "double", side_effect=AssertionError("unexpected float64 conversion")):
        actual = _velocity_to_x0(x_t, net_output, timestep, pid_memory_optimization=True)

    assert actual.dtype == x_dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_velocity_to_x0_uses_original_float64_math_when_optimization_is_disabled() -> None:
    x_t = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]], dtype=torch.bfloat16)
    net_output = torch.tensor([[[[0.5, -0.25], [0.125, -0.0625]]]], dtype=torch.bfloat16)
    timestep = torch.tensor([0.634], dtype=torch.float32)
    expected = (x_t.double() - timestep.double().view(1, 1, 1, 1) * net_output.double()).to(x_t.dtype)

    with patch("torch.addcmul", side_effect=AssertionError("unexpected optimized path")):
        actual = _velocity_to_x0(x_t, net_output, timestep, pid_memory_optimization=False)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("pid_memory_optimization", "expected_chunk_size"), [(False, None), (True, 1024)])
def test_student_sample_loop_passes_per_call_activation_chunk_size(
    pid_memory_optimization: bool, expected_chunk_size: int | None
) -> None:
    activation_chunk_sizes: list[int | None] = []

    def net(x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        activation_chunk_sizes.append(kwargs.get("activation_chunk_size"))  # type: ignore[arg-type]
        return torch.zeros_like(x)

    _student_sample_loop(
        net,  # type: ignore[arg-type]
        noise=torch.zeros(1, 3, 2, 2),
        t_list=torch.tensor([0.999, 0.0]),
        caption_embs=torch.zeros(1, 1, 1),
        caption_mask=None,
        lq_latent=None,
        degrade_sigma=torch.zeros(1),
        pid_memory_optimization=pid_memory_optimization,
    )

    assert activation_chunk_sizes == [expected_chunk_size]


def test_pid_decoder_decode_reaches_chunked_pixel_path(monkeypatch: pytest.MonkeyPatch) -> None:
    net = PidNet(
        in_channels=3,
        num_groups=2,
        hidden_size=8,
        pixel_hidden_size=4,
        pixel_attn_hidden_size=8,
        pixel_num_groups=2,
        patch_depth=1,
        pixel_depth=1,
        num_text_blocks=1,
        patch_size=2,
        txt_embed_dim=6,
        txt_max_length=4,
        rope_mode="original",
        rope_ref_h=4,
        rope_ref_w=4,
        lq_in_channels=0,
        lq_latent_channels=2,
        lq_hidden_dim=4,
        lq_num_res_blocks=1,
        lq_interval=1,
        sr_scale=1,
        latent_spatial_down_factor=8,
    ).eval()
    decoder = PiDDecoder(net, backbone=BaseModelType.Flux)
    calls: list[int] = []
    original_forward_chunked = PiTBlock._forward_chunked

    def spy_forward_chunked(self: PiTBlock, *args: object, **kwargs: object) -> torch.Tensor:
        calls.append(1)
        return original_forward_chunked(self, *args, **kwargs)

    monkeypatch.setattr(PiTBlock, "_forward_chunked", spy_forward_chunked)
    monkeypatch.setattr(pid_decode_module, "_PID_ACTIVATION_CHUNK_SIZE", 1)

    output = decoder.decode(
        latent=torch.randn(1, 2, 2, 2),
        caption_embs=torch.randn(1, 4, 6),
        config=PiDDecodeConfig(num_inference_steps=1, pid_memory_optimization=True, seed=0),
    )

    assert output.shape == (1, 3, 16, 16)
    assert calls


def test_pid_memory_optimization_defaults_to_disabled() -> None:
    assert PiDDecodeConfig().pid_memory_optimization is False


def test_working_memory_estimate_shrinks_when_the_optimization_is_enabled() -> None:
    """The estimates contain only calibrated activation/workspace terms."""
    latent = torch.zeros(1, 16, 64, 64)  # FLUX: 64 * 4 * 8 = 2048px output

    unoptimized = estimate_pid_decode_working_memory(latent, BaseModelType.Flux)
    optimized = estimate_pid_decode_working_memory(latent, BaseModelType.Flux, True)

    output_bytes = 2048 * 2048 * 4
    # Exact activation-only formulas reject reintroducing a fixed model/cache term.
    assert unoptimized == 260 * output_bytes
    assert optimized == 120 * output_bytes + 224 * 2**20


def test_working_memory_estimate_keeps_a_fixed_term_for_the_chunk_working_set() -> None:
    """The optimized peak is not a pure multiple of the output size.

    Chunking bounds the per-block activations to a fixed working set, so halving the output area does
    not halve the peak (measured: 509 MiB at 1024px vs 1533 MiB at 2048px — a factor of 3.0, not 4).
    A pure scaling constant would therefore under-reserve at small sizes or over-reserve at large ones.
    """
    small = estimate_pid_decode_working_memory(torch.zeros(1, 16, 32, 32), BaseModelType.Flux, True)
    large = estimate_pid_decode_working_memory(torch.zeros(1, 16, 64, 64), BaseModelType.Flux, True)

    assert large < 4 * small, "the estimate scales purely with area — the fixed chunk term is missing"
    assert large > 2 * small, "the per-pixel term has been lost"


def test_working_memory_estimate_accounts_for_every_image_in_the_batch() -> None:
    """The decoder accepts batched latents, so reserving only one image's activations can OOM.

    The optimized estimate has one fixed chunk working set plus a per-output-pixel term. Doubling the
    batch must therefore increase the estimate, but by less than 2x because the fixed term is shared.
    """
    single = estimate_pid_decode_working_memory(torch.zeros(1, 16, 64, 64), BaseModelType.Flux, True)
    batched = estimate_pid_decode_working_memory(torch.zeros(2, 16, 64, 64), BaseModelType.Flux, True)
    single_unoptimized = estimate_pid_decode_working_memory(torch.zeros(1, 16, 64, 64), BaseModelType.Flux)
    batched_unoptimized = estimate_pid_decode_working_memory(torch.zeros(2, 16, 64, 64), BaseModelType.Flux)

    assert single < batched < 2 * single
    assert batched_unoptimized == 2 * single_unoptimized


def test_working_memory_estimate_never_exceeds_the_unoptimized_one() -> None:
    """Below the chunk size the pixel blocks run unchunked, so the fixed chunk-working-set term must
    not be charged: a small output would otherwise reserve *more* with the optimization enabled than
    without it.
    """
    # 8 * 4 * 8 = 256px output -> 256 patch tokens, well under the 1024-token chunk size.
    small_latent = torch.zeros(1, 16, 8, 8)

    optimized = estimate_pid_decode_working_memory(small_latent, BaseModelType.Flux, True)
    unoptimized = estimate_pid_decode_working_memory(small_latent, BaseModelType.Flux)

    assert optimized <= unoptimized


def test_working_memory_estimate_still_returns_zero_for_unsupported_backbones() -> None:
    latent = torch.zeros(1, 16, 64, 64)
    for optimized in (False, True):
        assert estimate_pid_decode_working_memory(latent, BaseModelType.StableDiffusion1, optimized) == 0


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

    def test_non_string_keys_are_rejected_before_torch_sees_them(self, tiny_net: torch.nn.Module) -> None:
        """A bare checkpoint keeps whatever keys the `.pth` was pickled with (see `strip_net_prefix`),
        so one need not be a string. `nn.Module.load_state_dict` calls `.startswith()` on every key and
        raises `AttributeError` on a non-string one — from inside torch, before it reports anything —
        so `load_pid_decoder` has to catch it first and say what is actually wrong with the file."""
        sd: dict[Any, torch.Tensor] = dict(tiny_net.state_dict()) | {1: torch.zeros(1)}
        with pytest.raises(RuntimeError, match="1 keys that are not strings"):
            load_pid_decoder(sd, BaseModelType.Flux)
