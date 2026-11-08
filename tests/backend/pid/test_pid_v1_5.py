"""PiD v1.5: the network paths NVIDIA's second decoder generation uses, and how a checkpoint selects them.

v1.5 keeps the v1 backbone and changes the LQ projection around it: a 1024-wide projection, a per-token scalar gate
instead of a per-token per-channel one, replicate padding, LQ injection into the PiT pixel blocks, and for FLUX.2
an unpatchify of the 128-channel latent to 32 channels. The expected shapes below were read off the header of
Comfy-Org's `PixelDiT/diffusion_models/pid_1.5_qwenimage_1024_to_4096_4step_bf16.safetensors` (and its FLUX.2
sibling), and the settings no weight shape reveals from upstream's `PID_SR4X_V1PT5` net config — not derived from
the code under test.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.pid import decode as pid_decode_module
from invokeai.backend.pid._src.networks.lq_projection_2d import LQProjection2D, _build_gate
from invokeai.backend.pid._src.networks.pid_net import PidNet
from invokeai.backend.pid.decode import (
    BACKBONE_DISCRIMINATOR_KEY,
    build_pid_net,
    load_pid_decoder,
    required_pid_net_shapes,
)
from invokeai.backend.pid.state_dict_utils import PiDVersion

# Only in the v1.5 header, and 57 keys whose shape differs from the v1 (`pid_qwenimage_…_4step_bf16`) header.
_V1_5_ONLY_KEYS = {
    "lq_proj.pit_head.weight",
    "lq_proj.pit_head.bias",
    "pit_lq_gate.content_proj.weight",
    "pit_lq_gate.content_proj.bias",
    "pit_lq_gate.log_alpha",
}
_SHAPES_THAT_DIFFER_FROM_V1 = 57


class TestV1_5Contract:
    def test_matches_the_released_checkpoint(self) -> None:
        v1 = required_pid_net_shapes()
        v1_5 = required_pid_net_shapes(version=PiDVersion.V1_5)

        assert v1_5.keys() - v1.keys() == _V1_5_ONLY_KEYS
        assert v1.keys() <= v1_5.keys()
        assert v1_5[BACKBONE_DISCRIMINATOR_KEY] == (1024, 16, 3, 3)
        assert v1_5["lq_proj.latent_proj.2.weight"] == (1024, 1024, 3, 3)
        assert v1_5["lq_proj.output_heads.0.weight"] == (1536, 1024)
        assert v1_5["lq_proj.pit_head.weight"] == (1536, 1024)
        assert v1_5["lq_proj.gate_modules.6.content_proj.weight"] == (1, 3072)
        assert v1_5["pit_lq_gate.content_proj.weight"] == (1, 3072)
        assert v1_5["pit_lq_gate.log_alpha"] == ()
        differing = {k for k in v1 if v1[k] != v1_5[k]}
        assert len(differing) == _SHAPES_THAT_DIFFER_FROM_V1
        assert all(k.startswith("lq_proj.") for k in differing), "v1.5 must leave the backbone untouched"

    @pytest.mark.parametrize(
        ("backbone", "latent_channels"),
        [(BaseModelType.Flux, 16), (BaseModelType.QwenImage, 16), (BaseModelType.Flux2, 32)],
    )
    def test_one_contract_holds_for_every_backbone(self, backbone: BaseModelType, latent_channels: int) -> None:
        """FLUX.2's 128 channels are unpatchified before the projection, so its discriminator reads 32."""
        canonical = required_pid_net_shapes(version=PiDVersion.V1_5)
        per_backbone = required_pid_net_shapes(backbone, PiDVersion.V1_5)

        assert per_backbone.keys() == canonical.keys()
        assert {k for k in canonical if per_backbone[k] != canonical[k]} <= {BACKBONE_DISCRIMINATOR_KEY}
        assert per_backbone[BACKBONE_DISCRIMINATOR_KEY] == (1024, latent_channels, 3, 3)

    @pytest.mark.parametrize("backbone", [BaseModelType.StableDiffusion3, BaseModelType.StableDiffusionXL])
    def test_exists_only_for_the_backbones_nvidia_released(self, backbone: BaseModelType) -> None:
        with pytest.raises(ValueError, match="v1.5 decoder backbone"):
            required_pid_net_shapes(backbone, PiDVersion.V1_5)


@pytest.mark.parametrize(
    ("version", "padding_mode", "rope_ref"),
    [(PiDVersion.V1, "zeros", 1024), (PiDVersion.V1_5, "replicate", 2048)],
)
def test_the_released_settings_no_weight_shape_reveals(version: PiDVersion, padding_mode: str, rope_ref: int) -> None:
    """Padding and the RoPE reference size change no parameter shape, so no contract or load check can see them."""
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        net = build_pid_net(BaseModelType.QwenImage, version)

    padded_convs = [m for m in net.lq_proj.modules() if isinstance(m, torch.nn.Conv2d) and m.kernel_size != (1, 1)]
    assert padded_convs
    assert {conv.padding_mode for conv in padded_convs} == {padding_mode}
    assert (net.rope_ref_grid_h, net.rope_ref_grid_w) == (rope_ref // 16, rope_ref // 16)


def test_loading_builds_the_generation_the_latent_projection_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown width is built as v1, so the file still gets the loader's key and shape reports."""
    built: list[PiDVersion] = []

    class _Built(Exception):
        pass

    def build(backbone: BaseModelType, version: PiDVersion) -> PidNet:
        built.append(version)
        raise _Built

    monkeypatch.setattr(pid_decode_module, "build_pid_net", build)
    for width in (512, 1024, 768):
        with pytest.raises(_Built):
            load_pid_decoder(
                {BACKBONE_DISCRIMINATOR_KEY: torch.empty(width, 16, 3, 3, device="meta")}, BaseModelType.Flux
            )
    with pytest.raises(_Built):
        load_pid_decoder({}, BaseModelType.Flux)

    assert built == [PiDVersion.V1, PiDVersion.V1_5, PiDVersion.V1, PiDVersion.V1]


def test_the_per_token_gate_is_one_scalar_per_token_decaying_with_sigma() -> None:
    """content_proj zeroed and log_alpha 0: the gate is sigmoid(-sigma), 1/2 at sigma 0 and 1/4 at sigma ln 3."""
    gate = _build_gate("sigma_aware_per_token", dim=4)
    with torch.no_grad():
        gate.content_proj.weight.zero_()
        gate.content_proj.bias.zero_()
        gate.log_alpha.zero_()
    x = torch.zeros(2, 3, 4)
    lq = torch.ones(2, 3, 4)

    out = gate(x, lq, sigma=torch.tensor([0.0, math.log(3.0)]))

    assert gate.content_proj.out_features == 1
    torch.testing.assert_close(out[0], torch.full((3, 4), 0.5))
    torch.testing.assert_close(out[1], torch.full((3, 4), 0.25))


def test_unpatchify_is_the_pixel_shuffle_of_the_latent() -> None:
    """FLUX.2 patchifies 2x2 latent pixels into channels exactly as `pixel_unshuffle` does, so undoing it must equal
    `pixel_shuffle`. Unit patch ratio and an identity projection leave the unpatchified latent observable."""
    proj = LQProjection2D(
        in_channels=0,
        latent_channels=8,
        hidden_dim=4,
        patch_size=8,
        sr_scale=1,
        latent_spatial_down_factor=16,
        latent_unpatchify_factor=2,
    )
    proj.latent_proj = torch.nn.Identity()
    latent = torch.arange(8 * 2 * 2, dtype=torch.float32).reshape(1, 8, 2, 2)

    aligned = proj._align_latent_to_patch_grid(latent, 4, 4)

    torch.testing.assert_close(aligned, F.pixel_shuffle(latent, 2))


@pytest.mark.parametrize(("padding_mode", "uniform"), [("replicate", True), ("zeros", False)])
def test_replicate_padding_keeps_a_constant_latent_constant_at_the_border(padding_mode: str, uniform: bool) -> None:
    """Every layer of the projection maps a spatially constant input to a spatially constant output, except a conv
    that pads with zeros: its border tokens see the padding. One conv left on zeros breaks the uniformity."""
    torch.manual_seed(0)
    proj = LQProjection2D(
        in_channels=0,
        latent_channels=2,
        hidden_dim=4,
        out_dim=4,
        patch_size=2,
        sr_scale=1,
        latent_spatial_down_factor=2,
        num_res_blocks=1,
        conv_padding_mode=padding_mode,
    )
    latent = torch.full((1, 2, 4, 4), 0.7)

    with torch.no_grad():
        tokens = proj(lq_latent=latent, target_pH=4, target_pW=4)[0]

    spread = (tokens - tokens[:, :1]).abs().max().item()
    assert (spread < 1e-5) is uniform, spread


def _tiny_v1_5_net() -> PidNet:
    net = PidNet(
        in_channels=3,
        num_groups=2,
        hidden_size=8,
        pixel_hidden_size=4,
        pixel_attn_hidden_size=8,
        pixel_num_groups=2,
        patch_depth=2,
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
        lq_conv_padding_mode="replicate",
        lq_gate_type="sigma_aware_per_token",
        lq_interval=1,
        sr_scale=1,
        latent_spatial_down_factor=2,
        pit_lq_inject=True,
    ).eval()
    with torch.no_grad():
        # The backbone zero-inits its final layer, which would make every output zero. The patch-block heads are
        # zeroed instead, so the LQ latent reaches the pixels through the PiT injection alone.
        torch.nn.init.normal_(net.final_layer.linear.weight)
        for head in net.lq_proj.output_heads:
            head.weight.zero_()
            head.bias.zero_()
    return net


class TestPiTInjection:
    """The PiT gate broadcasts sigma per sample over [B, L, D] tokens; the PiT head's output rides last."""

    @pytest.fixture
    def inputs(self) -> dict[str, torch.Tensor]:
        torch.manual_seed(1)
        return {
            "x": torch.randn(2, 3, 4, 4),
            "t": torch.tensor([500.0, 250.0]),
            "y": torch.randn(2, 4, 6),
            # On the 2x2 patch grid itself: a unit patch ratio aligns nothing.
            "lq_latent": torch.randn(2, 2, 2, 2),
            "degrade_sigma": torch.tensor([0.0, 0.7]),
        }

    @staticmethod
    def _forward(net: PidNet, inputs: dict[str, torch.Tensor], **overrides: torch.Tensor) -> torch.Tensor:
        args = inputs | overrides
        with torch.no_grad():
            return net(
                args["x"], args["t"], args["y"], lq_latent=args["lq_latent"], degrade_sigma=args["degrade_sigma"]
            )

    def test_a_sample_decodes_the_same_alone_as_inside_a_batch(self, inputs: dict[str, torch.Tensor]) -> None:
        """Fed flattened [B*L, D] tokens the gate mixed every token of the batch with every sample's sigma."""
        torch.manual_seed(0)
        net = _tiny_v1_5_net()

        batched = self._forward(net, inputs)
        alone = self._forward(net, {k: v[1:] for k, v in inputs.items()})

        assert batched.abs().max() > 0
        torch.testing.assert_close(batched[1:], alone)

    def test_the_lq_latent_reaches_the_pixels_through_the_pit_head(self, inputs: dict[str, torch.Tensor]) -> None:
        torch.manual_seed(0)
        net = _tiny_v1_5_net()
        with_pit = self._forward(net, inputs)
        with torch.no_grad():
            net.lq_proj.pit_head.weight.zero_()
            net.lq_proj.pit_head.bias.zero_()

        assert not torch.allclose(self._forward(net, inputs), with_pit)

    def test_each_sample_is_gated_by_its_own_sigma(self, inputs: dict[str, torch.Tensor]) -> None:
        torch.manual_seed(0)
        net = _tiny_v1_5_net()

        straight = self._forward(net, inputs)
        swapped = self._forward(net, inputs, degrade_sigma=inputs["degrade_sigma"].flip(0))

        assert not torch.allclose(straight[1], swapped[1])
