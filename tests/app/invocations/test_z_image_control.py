"""What the Z-Image denoise does with its control field: refuses a run without the VAE or with another kind of model,
locks the control with the transformer's reservation and hooks it inside the exit stack, gates it once per step for
both guidance passes and both of Heun's forwards, and builds the packed control context VideoX-Fun builds for control
alone.

The denoise runs its transformer in bfloat16, which the CPU test fleet cannot matmul, so every run here stops at the
LoRA patcher or runs the loop with the transformer's forward stood in. The forward with the hooks is covered in
`tests/backend/fun_controlnet/test_z_image_fun_control.py`; the reservation and the encode's tiling in
`test_z_image_denoise_working_memory.py`.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from PIL import Image

from invokeai.app.invocations.fields import ZImageConditioningField
from invokeai.app.invocations.model import ModelIdentifierField, VAEField
from invokeai.app.invocations.z_image import z_image_denoise
from invokeai.app.invocations.z_image.z_image_control import ZImageControlField
from invokeai.app.invocations.z_image.z_image_denoise import ZImageDenoiseInvocation
from invokeai.backend.fun_controlnet.hooks import FunControlHooks
from invokeai.backend.fun_controlnet.z_image import Z_IMAGE_FUN_CONTROL_2_1_LITE, ZImageFunControl
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    ZImageConditioningInfo,
)
from invokeai.backend.util.devices import TorchDevice

SIDE = 64  # an 8x8 latent: 16 image tokens
LATENT_SIDE = SIDE // 8
CAPTION_TOKENS = 4
CAPTION_DIM = 32
HIDDEN = 64
HEADS = 2
# The Lite layout's hints land after base layers 0, 10 and 20, which a 21-block transformer is just deep enough for.
LAYERS = 21


class _Stop(Exception):
    """Raised where the node has made the decision a test looks at."""


@pytest.fixture(autouse=True)
def cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", lambda _device=None: torch.bfloat16)


def _model(type_: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(key=type_.value, hash="hash", name=type_.value, base=BaseModelType.ZImage, type=type_)


_TRANSFORMER_CONFIG = {
    "all_patch_size": (2,),
    "all_f_patch_size": (1,),
    "in_channels": 16,
    "dim": HIDDEN,
    "n_layers": LAYERS,
    "n_refiner_layers": 2,
    "n_heads": HEADS,
    "n_kv_heads": HEADS,
    "cap_feat_dim": CAPTION_DIM,
    "axes_dims": [8, 12, 12],
    "axes_lens": [128, 32, 32],
}


def _tiny_transformer() -> ZImageTransformer2DModel:
    return ZImageTransformer2DModel(**_TRANSFORMER_CONFIG).eval()


def _tiny_control() -> ZImageFunControl:
    return ZImageFunControl(Z_IMAGE_FUN_CONTROL_2_1_LITE, hidden_size=HIDDEN, num_attention_heads=HEADS).eval()


def _loaded(model, lock) -> MagicMock:
    """A `LoadedModel` stand-in: `model` unlocked, `model_on_device` as `lock` says (a context or a raise)."""
    info = MagicMock()
    info.model = model
    info.compute_device = torch.device("cpu")
    info.model_on_device = MagicMock(side_effect=lock)
    return info


def _yielding(model):
    # The transformer and the control are locked with `working_mem_bytes`, the VAE with `working_memory`.
    return lambda **_kwargs: nullcontext((None, model))


def _control_field(**overrides) -> ZImageControlField:
    return ZImageControlField(image_name="canny", control_model=_model(ModelType.ControlNet), **overrides)


def _denoise(**overrides) -> ZImageDenoiseInvocation:
    fields = {
        "latents": None,
        "noise": None,
        "denoise_mask": None,
        "denoising_start": 0.0,
        "denoising_end": 1.0,
        "add_noise": True,
        "transformer": SimpleNamespace(transformer=_model(ModelType.Main), loras=[]),
        "positive_conditioning": ZImageConditioningField(conditioning_name="pos"),
        "negative_conditioning": ZImageConditioningField(conditioning_name="neg"),
        "guidance_scale": 3.0,
        "width": SIDE,
        "height": SIDE,
        "steps": 2,
        "seed": 0,
        "shift": None,
        "scheduler": "euler",
        "control": _control_field(),
        "vae": VAEField(vae=_model(ModelType.VAE)),
    }
    fields.update(overrides)
    return ZImageDenoiseInvocation.model_construct(**fields)


def _control_image() -> Image.Image:
    return Image.linear_gradient("L").resize((SIDE // 2, SIDE // 2)).convert("RGB")


def _context(transformer_info=None, control_info=None, vae_info=None) -> MagicMock:
    infos = {ModelType.Main: transformer_info, ModelType.ControlNet: control_info, ModelType.VAE: vae_info}
    context = MagicMock()
    context.models.load.side_effect = lambda identifier: infos[identifier.type]
    context.models.get_config.return_value = SimpleNamespace(
        base=BaseModelType.ZImage, type=ModelType.Main, format=ModelFormat.Checkpoint
    )
    context.conditioning.load.side_effect = lambda name: ConditioningFieldData(
        conditionings=[ZImageConditioningInfo(prompt_embeds=torch.randn(CAPTION_TOKENS, CAPTION_DIM))]
    )
    context.images.get_pil.side_effect = lambda name: {"canny": _control_image()}[name]
    context.config.get.return_value.force_tiled_decode = False
    return context


def _hook_count(transformer) -> int:
    blocks = [*transformer.noise_refiner, *transformer.layers]
    return sum(len(block._forward_hooks) + len(block._forward_pre_hooks) for block in blocks)


def test_control_without_its_vae_is_refused_before_anything_else_is_loaded() -> None:
    context = _context(transformer_info=_loaded(_tiny_transformer(), _Stop))

    with pytest.raises(ValueError, match="needs the VAE"):
        _denoise(vae=None)._run_diffusion(context)

    assert context.models.load.call_count == 1


def test_another_kind_of_control_model_is_refused() -> None:
    context = _context(
        transformer_info=_loaded(_tiny_transformer(), _Stop), control_info=_loaded(torch.nn.Linear(2, 2), _Stop)
    )

    with pytest.raises(TypeError, match="Expected a Z-Image ControlNet"):
        _denoise()._run_diffusion(context)


def test_the_control_is_locked_with_the_same_reservation_and_hooked_inside_the_exit_stack(
    monkeypatch: pytest.MonkeyPatch, flux_shaped_vae
) -> None:
    transformer, control, vae = _tiny_transformer(), _tiny_control(), flux_shaped_vae()
    transformer_info = _loaded(transformer, _yielding(transformer))
    control_info = _loaded(control, _yielding(control))
    context = _context(transformer_info, control_info, _loaded(vae, _yielding(vae)))
    hooks_at_patching: list[int] = []

    def stop(**_kwargs):
        hooks_at_patching.append(_hook_count(transformer))
        raise _Stop

    monkeypatch.setattr(LayerPatcher, "apply_smart_model_patches", stop)

    with pytest.raises(_Stop):
        _denoise()._run_diffusion(context)

    reserved = transformer_info.model_on_device.call_args.kwargs["working_mem_bytes"]
    assert control_info.model_on_device.call_args.kwargs["working_mem_bytes"] == reserved
    # Two entry pre-hooks, the refiner pair and the three Lite layers.
    assert hooks_at_patching == [2 + 2 + 3]
    assert _hook_count(transformer) == 0


class _GateRecordingTransformer(ZImageTransformer2DModel):
    """The real class, so the real hooks land on its blocks, with the forward replaced: no bfloat16 matmul on the CPU,
    and a record of the control gate at every call."""

    def __init__(self) -> None:
        super().__init__(**_TRANSFORMER_CONFIG)
        self.hooks: FunControlHooks | None = None
        self.gates: list[bool] = []

    def forward(self, x, t, cap_feats, *_args, **_kwargs):
        assert self.hooks is not None
        self.gates.append(self.hooks.active)
        return ([torch.zeros_like(item) for item in x],)


@pytest.mark.parametrize(
    ("scheduler", "begin", "end", "expected_gates"),
    [
        # Four Euler steps with CFG are eight forwards, two per step.
        ("euler", 0.5, 1.0, [False] * 4 + [True] * 4),
        ("euler", 0.0, 0.5, [True] * 6 + [False] * 2),
        # Heun runs two forwards per step but the last; with CFG, four calls per step: user steps 0 and 1 off, 2 and
        # 3 on.
        ("heun", 0.5, 1.0, [False] * 8 + [True] * 6),
        # User steps 0 and 1 on: both forwards of step 1 run with the control, though the second is the fourth call.
        ("heun", 0.0, 0.25, [True] * 8 + [False] * 6),
    ],
    ids=["euler-second-half", "euler-first-half", "heun-second-half", "heun-first-quarter"],
)
def test_the_gate_is_set_once_per_step_for_both_guidance_passes(
    monkeypatch: pytest.MonkeyPatch, flux_shaped_vae, scheduler, begin, end, expected_gates
) -> None:
    """The gate follows `user step / total steps` with both ends included and is re-evaluated every step; the positive
    and the negative pass of a step, and Heun's two forwards of a step, see the same one."""
    transformer, control, vae = _GateRecordingTransformer(), _tiny_control(), flux_shaped_vae()
    install = z_image_denoise.install_fun_control

    @contextmanager
    def capturing_install(*args, **kwargs):
        with install(*args, **kwargs) as hooks:
            transformer.hooks = hooks
            yield hooks

    monkeypatch.setattr(z_image_denoise, "install_fun_control", capturing_install)
    context = _context(
        _loaded(transformer, _yielding(transformer)),
        _loaded(control, _yielding(control)),
        _loaded(vae, _yielding(vae)),
    )

    _denoise(
        steps=4, scheduler=scheduler, control=_control_field(begin_step_percent=begin, end_step_percent=end)
    )._run_diffusion(context)

    assert transformer.gates == expected_gates


@torch.no_grad()
def test_the_control_context_is_the_mode_encoded_control_image_packed_like_the_latents(flux_shaped_vae) -> None:
    """VideoX-Fun's context for control alone, `[control | keep mask | masked image]` with the last two zero, from the
    mode of the VAE's distribution, normalized as the generation latents are, and patched 2x2 the way the transformer
    patches its latents."""
    vae, control, transformer = flux_shaped_vae(), _tiny_control(), _tiny_transformer()
    context = _context(vae_info=_loaded(vae, _yielding(vae)))

    packed = _denoise()._build_control_context(context, control)

    pixels = image_resized_to_grid_as_tensor(_control_image().resize((SIDE, SIDE), Image.Resampling.LANCZOS))
    mode = vae.encode(pixels.unsqueeze(0)).latent_dist.mode()
    latents = (mode - vae.config.shift_factor) * vae.config.scaling_factor
    context_4d = torch.cat([latents, torch.zeros(1, 17, LATENT_SIDE, LATENT_SIDE)], dim=1)
    expected, _, _ = transformer._patchify_image(context_4d[0].unsqueeze(1), 2, 1)
    assert packed.shape == (1, (LATENT_SIDE // 2) ** 2, 132)
    assert torch.equal(packed[0], expected)
