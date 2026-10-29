"""The FLUX.2 [dev] ControlNet node, and what the FLUX.2 denoise does with its field: refuses Klein and a VAE without
batch-norm statistics, reserves the hints over the whole image stream, encodes the control images before the
transformer's lock and pads the context for reference tokens, locks the branch with the same reservation inside the
exit stack, and gates it per user step.

The denoise runs its transformer in bfloat16, which the CPU test fleet cannot matmul, so every run here stops at the
transformer's lock or at the LoRA patcher, or runs the loop with the transformer's forward stood in. The forward with
the hooks is covered in `tests/backend/fun_controlnet`.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from PIL import Image
from pydantic import ValidationError

from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.flux2 import flux2_denoise
from invokeai.app.invocations.flux2.flux2_control import Flux2ControlField, Flux2ControlInvocation
from invokeai.app.invocations.flux2.flux2_denoise import Flux2DenoiseInvocation
from invokeai.app.invocations.model import ModelIdentifierField, VAEField
from invokeai.app.invocations.vae.flux2_vae_encode import Flux2VaeEncodeInvocation
from invokeai.backend.flux2.denoise_working_memory import estimate_flux2_denoise_working_memory
from invokeai.backend.fun_controlnet.flux2 import FLUX2_FUN_CONTROL_IN_DIM, Flux2FunControlBranch
from invokeai.backend.fun_controlnet.hooks import FunControlHooks
from invokeai.backend.model_manager.taxonomy import BaseModelType, Flux2VariantType, ModelFormat, ModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, FLUXConditioningInfo
from invokeai.backend.util import attention
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.working_memory import probe

SIDE = 64  # an 8x8 latent: 16 image tokens
IMAGE_TOKENS = (SIDE // 16) ** 2
REFERENCE_TOKENS = 9
TEXT_TOKENS = 4
HIDDEN = 16
DEV_HIDDEN = 6144  # the width the estimate takes for a [dev] config
NUM_CONTROL_BLOCKS = 4


class _Stop(Exception):
    """Raised where the node has made the decision a test looks at."""


def _model(type_: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(
        key=f"{type_.value}", hash="hash", name=type_.value, base=BaseModelType.Flux2, type=type_
    )


# --- the node ---


def test_the_node_hands_its_images_and_settings_to_the_field() -> None:
    node = Flux2ControlInvocation(
        id="control",
        control_model=_model(ModelType.ControlNet),
        image=ImageField(image_name="canny"),
        inpaint_image=ImageField(image_name="photo"),
        inpaint_mask=ImageField(image_name="mask"),
        control_context_scale=0.7,
        begin_step_percent=0.1,
        end_step_percent=0.9,
        tiled=True,
        tile_size=512,
    )

    assert node.invoke(MagicMock()).control == Flux2ControlField(
        image_name="canny",
        inpaint_image_name="photo",
        inpaint_mask_name="mask",
        control_model=_model(ModelType.ControlNet),
        control_context_scale=0.7,
        begin_step_percent=0.1,
        end_step_percent=0.9,
        tiled=True,
        tile_size=512,
    )


@pytest.mark.parametrize(
    ("images", "message"),
    [
        ({}, "control image"),
        ({"inpaint_mask": ImageField(image_name="mask")}, "go together"),
        ({"inpaint_image": ImageField(image_name="photo")}, "go together"),
    ],
    ids=["nothing", "mask-alone", "image-alone"],
)
def test_the_node_refuses_to_run_without_a_complete_input(images, message) -> None:
    node = Flux2ControlInvocation(id="control", control_model=_model(ModelType.ControlNet), **images)
    with pytest.raises(ValueError, match=message):
        node.invoke(MagicMock())


def test_a_begin_after_the_end_is_refused_when_the_node_is_built() -> None:
    with pytest.raises(ValidationError, match="Begin step percent"):
        Flux2ControlInvocation(
            id="control", control_model=_model(ModelType.ControlNet), begin_step_percent=0.8, end_step_percent=0.2
        )


# --- the denoise ---


@pytest.fixture(autouse=True)
def cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))


@pytest.fixture(autouse=True)
def fused_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the dispatch questions `sdpa_score_matrix_bytes` asks, so no test depends on the torch build of the
    machine it runs on."""
    monkeypatch.setattr(attention, "_diffusers_attention_dispatch", lambda: attention._DISPATCH_TORCH)
    monkeypatch.setattr(attention, "rocm_sdpa_uses_math_kernel", lambda *_args: False)
    monkeypatch.setattr(attention, "_torch_sdpa_materializes_score_matrix", lambda *_args: False)


_TINY_TRANSFORMER = {
    "patch_size": 1,
    "in_channels": 128,
    "num_layers": 8,
    "num_single_layers": 1,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 8,
    "timestep_guidance_channels": 16,
    "axes_dims_rope": (2, 2, 2, 2),
}


def _tiny_transformer() -> Flux2Transformer2DModel:
    return Flux2Transformer2DModel(**_TINY_TRANSFORMER).eval()


def _tiny_branch() -> Flux2FunControlBranch:
    return Flux2FunControlBranch(hidden_size=HIDDEN, num_attention_heads=2, attention_head_dim=8).eval()


def _tiny_vae() -> AutoencoderKLFlux2:
    """The real FLUX.2 VAE class at toy width: 8x compression, 32 latent channels, and a batch norm whose statistics
    are not the identity, so a skipped normalization shows."""
    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(
        block_out_channels=(4, 4, 4, 4), layers_per_block=1, latent_channels=32, norm_num_groups=2, sample_size=64
    ).eval()
    vae.bn.running_mean.uniform_(-0.5, 0.5)
    vae.bn.running_var.uniform_(0.5, 2.0)
    return vae


def _loaded(model, lock) -> MagicMock:
    """A `LoadedModel` stand-in: `model` unlocked, `model_on_device` as `lock` says (a context or a raise)."""
    info = MagicMock()
    info.model = model
    info.compute_device = torch.device("cpu")
    info.model_on_device = MagicMock(side_effect=lock)
    return info


def _yielding(model):
    return lambda working_mem_bytes=None, working_memory=None: nullcontext((None, model))


def _stopping(working_mem_bytes=None, working_memory=None):
    raise _Stop


def _control_field(**overrides) -> Flux2ControlField:
    field = {
        "image_name": "canny",
        "inpaint_image_name": "photo",
        "inpaint_mask_name": "mask",
        "control_model": _model(ModelType.ControlNet),
    }
    field.update(overrides)
    return Flux2ControlField(**field)


def _denoise(**overrides) -> Flux2DenoiseInvocation:
    fields = {
        "latents": None,
        "noise": None,
        "denoise_mask": None,
        "denoising_start": 0.0,
        "denoising_end": 1.0,
        "add_noise": True,
        "transformer": SimpleNamespace(transformer=_model(ModelType.Main), loras=[]),
        "positive_text_conditioning": SimpleNamespace(conditioning_name="pos", mask=None),
        "negative_text_conditioning": SimpleNamespace(conditioning_name="neg"),
        "guidance": 4.0,
        "cfg_scale": 2.0,
        "width": SIDE,
        "height": SIDE,
        "num_steps": 2,
        "scheduler": "euler",
        "seed": 0,
        "vae": VAEField(vae=_model(ModelType.VAE)),
        "kontext_conditioning": None,
        "control": _control_field(),
    }
    fields.update(overrides)
    return Flux2DenoiseInvocation.model_construct(**fields)


def _control_image() -> Image.Image:
    return Image.linear_gradient("L").resize((SIDE // 2, SIDE // 2)).convert("RGB")


def _inpaint_image() -> Image.Image:
    return Image.linear_gradient("L").rotate(90).resize((SIDE, SIDE)).convert("RGB")


def _inpaint_mask() -> Image.Image:
    """White (regenerate) in the top-left quarter."""
    mask = Image.new("L", (SIDE, SIDE), 0)
    mask.paste(255, (0, 0, SIDE // 2, SIDE // 2))
    return mask


def _context(transformer_info=None, control_info=None, vae=None, variant=Flux2VariantType.Dev) -> MagicMock:
    vae = vae if vae is not None else _tiny_vae()
    infos = {
        ModelType.Main: transformer_info,
        ModelType.ControlNet: control_info,
        ModelType.VAE: _loaded(vae, _yielding(vae)),
    }
    context = MagicMock()
    context.models.load.side_effect = lambda identifier: infos[identifier.type]
    context.models.get_config.return_value = SimpleNamespace(
        base=BaseModelType.Flux2, type=ModelType.Main, format=ModelFormat.Checkpoint, variant=variant
    )
    context.conditioning.load.side_effect = lambda name: ConditioningFieldData(
        conditionings=[FLUXConditioningInfo(clip_embeds=torch.zeros(1, 768), t5_embeds=torch.zeros(1, TEXT_TOKENS, 8))]
    )
    images = {"canny": _control_image(), "photo": _inpaint_image(), "mask": _inpaint_mask()}
    context.images.get_pil.side_effect = lambda name: images[name]
    context.config.get.return_value.force_tiled_decode = False
    return context


def _loaded_types(context) -> list[ModelType]:
    return [call.args[0].type for call in context.models.load.call_args_list]


def _hook_count(transformer) -> int:
    return sum(len(block._forward_hooks) + len(block._forward_pre_hooks) for block in transformer.transformer_blocks)


@pytest.mark.parametrize("variant", [Flux2VariantType.Klein9B, Flux2VariantType.Klein4BBase], ids=lambda v: v.value)
def test_control_is_refused_on_klein_before_the_controlnet_is_loaded(variant) -> None:
    context = _context(transformer_info=_loaded(_tiny_transformer(), _stopping), variant=variant)

    with pytest.raises(ValueError, match=r"FLUX.2 \[dev\] only, not with FLUX.2 Klein models"):
        _denoise()._run_diffusion(context)

    assert ModelType.ControlNet not in _loaded_types(context)


def test_control_is_refused_with_a_vae_that_has_no_batch_norm_statistics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Flux2DenoiseInvocation, "_get_bn_stats", lambda self, context: None)
    context = _context(transformer_info=_loaded(_tiny_transformer(), _stopping))

    with pytest.raises(ValueError, match="batch-norm statistics"):
        _denoise()._run_diffusion(context)

    assert ModelType.ControlNet not in _loaded_types(context)


def test_a_model_that_is_not_the_flux2_branch_is_refused() -> None:
    context = _context(transformer_info=_loaded(_tiny_transformer(), _stopping), control_info=_loaded(object(), None))

    with pytest.raises(TypeError, match="Expected a FLUX.2 ControlNet"):
        _denoise()._run_diffusion(context)


class _ReferenceImages:
    """Stands in for `Flux2RefImageExtension`: one reference image's tokens after the generated ones."""

    def __init__(self, **_kwargs) -> None:
        self.ref_image_latents = torch.randn(1, REFERENCE_TOKENS, 128).to(torch.bfloat16)
        self.ref_image_ids = torch.full((1, REFERENCE_TOKENS, 4), 10, dtype=torch.long)

    def ensure_batch_size(self, _batch_size: int) -> None:
        pass


def test_with_control_and_a_reference_image_the_hints_are_reserved_over_the_whole_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hints are as long as the image stream the transformer sees, and that stream carries the reference tokens."""
    monkeypatch.setattr(flux2_denoise, "Flux2RefImageExtension", _ReferenceImages)
    transformer_info = _loaded(_tiny_transformer(), _stopping)
    vae = _tiny_vae()
    context = _context(transformer_info, _loaded(_tiny_branch(), _yielding(_tiny_branch())), vae)
    node = _denoise(kontext_conditioning=[SimpleNamespace()])

    with pytest.raises(_Stop):
        node._run_diffusion(context)

    without_control = estimate_flux2_denoise_working_memory(
        transformer_info.model,
        hidden_size=DEV_HIDDEN,
        image_seq_len=IMAGE_TOKENS,
        ref_image_seq_len=REFERENCE_TOKENS,
        text_seq_len=TEXT_TOKENS,
        batch_size=1,
        lora_count=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    hints = (NUM_CONTROL_BLOCKS + 4) * (IMAGE_TOKENS + REFERENCE_TOKENS) * DEV_HIDDEN * torch.bfloat16.itemsize
    working_memory = transformer_info.model_on_device.call_args.kwargs["working_memory"]
    assert working_memory.reservation_bytes == without_control.reservation_bytes + hints
    # The control image and the masked inpaint image went through the VAE before the transformer was locked.
    vae_info = context.models.load(VAEField(vae=_model(ModelType.VAE)).vae)
    encodes = [call for call in vae_info.model_on_device.call_args_list if "working_memory" in call.kwargs]
    assert len(encodes) == 2


def test_the_control_context_is_zero_for_the_reference_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flux2_denoise, "Flux2RefImageExtension", _ReferenceImages)
    installed: list[torch.Tensor] = []

    @contextmanager
    def recording_install(_transformer, _branch, control_context, *, scale):
        installed.append(control_context)
        raise _Stop
        yield

    monkeypatch.setattr(flux2_denoise, "install_fun_control", recording_install)
    transformer, branch = _tiny_transformer(), _tiny_branch()
    context = _context(_loaded(transformer, _yielding(transformer)), _loaded(branch, _yielding(branch)))

    with pytest.raises(_Stop):
        _denoise(kontext_conditioning=[SimpleNamespace()])._run_diffusion(context)

    (control_context,) = installed
    assert control_context.shape == (1, IMAGE_TOKENS + REFERENCE_TOKENS, FLUX2_FUN_CONTROL_IN_DIM)
    assert control_context.dtype == torch.bfloat16
    assert torch.equal(control_context[:, IMAGE_TOKENS:], torch.zeros(1, REFERENCE_TOKENS, FLUX2_FUN_CONTROL_IN_DIM))
    assert torch.all(control_context[:, :IMAGE_TOKENS].abs().sum(dim=-1) > 0)


@pytest.mark.parametrize(
    "calibrating, device, calibration_lock",
    [(False, "cuda", False), (True, "cuda", True), (True, "cpu", False)],
    ids=["outside-a-scope", "in-a-scope-on-a-gpu", "in-a-scope-on-the-cpu"],
)
def test_the_branch_is_locked_with_the_same_reservation_and_hooked_inside_the_exit_stack(
    monkeypatch: pytest.MonkeyPatch, calibrating: bool, device: str, calibration_lock: bool
) -> None:
    """Inside a calibration scope the transformer's lock keeps the calibration reservation free on a GPU, and so does
    the branch's, which would otherwise use up the window's headroom."""
    transformer, branch = _tiny_transformer(), _tiny_branch()
    transformer_info = _loaded(transformer, _yielding(transformer))
    control_info = _loaded(branch, _yielding(branch))
    control_info.compute_device = torch.device(device)
    monkeypatch.setattr(probe, "current_scope", lambda: object() if calibrating else None)
    context = _context(transformer_info, control_info)
    hooks_at_patching: list[int] = []

    def stop(**_kwargs):
        hooks_at_patching.append(_hook_count(transformer))
        raise _Stop

    monkeypatch.setattr(LayerPatcher, "apply_smart_model_patches", stop)

    with pytest.raises(_Stop):
        _denoise()._run_diffusion(context)

    # A plain reservation: a second estimate would open a calibration window nested in the transformer's.
    reserved = transformer_info.model_on_device.call_args.kwargs["working_memory"].reservation_bytes
    expected = probe.calibration_reservation(reserved) if calibration_lock else reserved
    assert control_info.model_on_device.call_args.kwargs == {"working_mem_bytes": expected}
    assert hooks_at_patching == [1 + NUM_CONTROL_BLOCKS]
    assert _hook_count(transformer) == 0


class _GateRecordingTransformer(Flux2Transformer2DModel):
    """The real class, so the node's checks hold and the real hooks land on its blocks, with the forward replaced: no
    bfloat16 matmul on the CPU, and a record of the control gate at every call."""

    def __init__(self) -> None:
        super().__init__(**_TINY_TRANSFORMER)
        self.hooks: FunControlHooks | None = None
        self.gates: list[bool] = []

    def forward(self, hidden_states: torch.Tensor, *_args, **_kwargs):
        assert self.hooks is not None
        self.gates.append(self.hooks.active)
        return (torch.zeros(*hidden_states.shape[:2], self.out_channels, dtype=hidden_states.dtype),)


@pytest.mark.parametrize(
    ("scheduler", "begin", "end", "expected_gates"),
    [
        ("euler", 0.5, 1.0, [False] * 4 + [True] * 4),
        ("euler", 0.0, 0.5, [True] * 6 + [False] * 2),
        # Heun: two forwards per step, one for the last; CFG doubles each.
        ("heun", 0.5, 1.0, [False] * 8 + [True] * 6),
    ],
    ids=["euler-second-half", "euler-first-half", "heun-second-half"],
)
def test_the_gate_follows_the_user_step_for_every_forward_of_it(
    monkeypatch: pytest.MonkeyPatch, scheduler, begin, end, expected_gates
) -> None:
    """Four steps with CFG: the gate follows `step / total_steps` with both ends included, and the positive and the
    negative forward of a step -- and Heun's two evaluations of it -- see the same one."""
    transformer, branch = _GateRecordingTransformer(), _tiny_branch()
    install = flux2_denoise.install_fun_control

    @contextmanager
    def capturing_install(*args, **kwargs):
        with install(*args, **kwargs) as hooks:
            transformer.hooks = hooks
            yield hooks

    monkeypatch.setattr(flux2_denoise, "install_fun_control", capturing_install)
    context = _context(_loaded(transformer, _yielding(transformer)), _loaded(branch, _yielding(branch)))

    _denoise(
        num_steps=4,
        scheduler=scheduler,
        control=_control_field(begin_step_percent=begin, end_step_percent=end),
    )._run_diffusion(context)

    assert transformer.gates == expected_gates
    assert _hook_count(transformer) == 0


@pytest.mark.parametrize(
    ("tiled", "tile_size", "force_tiled_decode", "expected_tiled"),
    [(False, 0, False, False), (True, 512, False, True), (False, 0, True, True)],
    ids=["untiled", "field", "forced"],
)
def test_the_control_images_are_encoded_at_generation_size_with_the_fields_tiling(
    monkeypatch: pytest.MonkeyPatch, tiled, tile_size, force_tiled_decode, expected_tiled
) -> None:
    vae = _tiny_vae()
    context = _context(vae=vae)
    context.config.get.return_value.force_tiled_decode = force_tiled_decode
    encode = Flux2VaeEncodeInvocation.vae_encode
    encoded_with: list[tuple[tuple[int, ...], bool, int]] = []

    def recording(**kwargs):
        encoded_with.append((tuple(kwargs["image_tensor"].shape), kwargs["tiled"], kwargs["tile_size"]))
        return encode(**kwargs)

    monkeypatch.setattr(Flux2VaeEncodeInvocation, "vae_encode", staticmethod(recording))
    bn_mean, bn_std = vae.bn.running_mean, torch.sqrt(vae.bn.running_var + vae.bn.eps)

    _denoise(control=_control_field(tiled=tiled, tile_size=tile_size))._build_control_context(
        context, bn_mean, bn_std, image_seq_len=IMAGE_TOKENS, ref_image_seq_len=0
    )

    # The control image (half size in the gallery) and the masked inpaint image, both at the generation size.
    assert encoded_with == [((1, 3, SIDE, SIDE), expected_tiled, tile_size)] * 2


def test_control_images_that_do_not_cover_the_latents_are_refused() -> None:
    """Init latents of another size than width x height would otherwise meet the context in the hook as a shape
    error."""
    vae = _tiny_vae()
    bn_mean, bn_std = vae.bn.running_mean, torch.sqrt(vae.bn.running_var + vae.bn.eps)

    with pytest.raises(ValueError, match="generation size"):
        _denoise()._build_control_context(
            _context(vae=vae), bn_mean, bn_std, image_seq_len=4 * IMAGE_TOKENS, ref_image_seq_len=0
        )


# --- the control context the node builds, against an independent restatement ---


def _pixels(image: Image.Image) -> torch.Tensor:
    """An RGB image at generation size as the VAE takes it: (1, 3, H, W) in [-1, 1]."""
    resized = image.convert("RGB").resize((SIDE, SIDE), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(resized, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)


def _patchify(latents: torch.Tensor) -> torch.Tensor:
    """VideoX-Fun `Flux2ControlPipeline._patchify_latents`."""
    b, c, h, w = latents.shape
    return latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def _pack(latents: torch.Tensor) -> torch.Tensor:
    """VideoX-Fun `Flux2ControlPipeline._pack_latents`."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).permute(0, 2, 1)


@torch.no_grad()
def _normalized_encoding(vae: AutoencoderKLFlux2, pixels: torch.Tensor) -> torch.Tensor:
    """`pipeline_flux2_control.py`: encode, take the mode, patchify, batch-normalize with the VAE's running stats."""
    patches = _patchify(vae.encode(pixels).latent_dist.mode())
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var + vae.bn.eps).view(1, -1, 1, 1)
    return _pack((patches - mean) / std)


def _context_the_denoise_installs(monkeypatch: pytest.MonkeyPatch, control: Flux2ControlField):
    """Runs the denoise up to the hooks and returns the control context it hands them, with the VAE it encoded with:
    the node's own image loading, resizing, batch-norm statistics and encode, nothing stood in."""
    installed: list[torch.Tensor] = []

    @contextmanager
    def recording_install(_transformer, _branch, control_context, *, scale):
        installed.append(control_context)
        raise _Stop
        yield

    monkeypatch.setattr(flux2_denoise, "install_fun_control", recording_install)
    transformer, branch, vae = _tiny_transformer(), _tiny_branch(), _tiny_vae()
    context = _context(_loaded(transformer, _yielding(transformer)), _loaded(branch, _yielding(branch)), vae)

    with pytest.raises(_Stop):
        _denoise(control=control)._run_diffusion(context)

    (control_context,) = installed
    return control_context.float(), vae


# The context reaches the hooks in bfloat16; the restatement runs in float32.
BF16 = {"atol": 2e-2, "rtol": 2e-2}


def test_a_control_image_alone_fills_the_control_features_and_leaves_the_inpaint_features_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_context, vae = _context_the_denoise_installs(
        monkeypatch, _control_field(inpaint_image_name=None, inpaint_mask_name=None)
    )

    expected_control = _normalized_encoding(vae, _pixels(_control_image()))
    torch.testing.assert_close(control_context[..., :128], expected_control, **BF16)
    assert torch.equal(control_context[..., 128:], torch.zeros(1, IMAGE_TOKENS, 132))


def test_an_inpaint_pair_alone_fills_the_keep_mask_and_the_masked_image_and_no_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_context, vae = _context_the_denoise_installs(monkeypatch, _control_field(image_name=None))

    assert torch.equal(control_context[..., :128], torch.zeros(1, IMAGE_TOKENS, 128))
    # The mask is white (regenerate) in the top-left quarter: the keep channel is 0 there and 1 elsewhere, at latent
    # resolution, then patchified with the latents.
    keep = torch.ones(1, 1, SIDE // 8, SIDE // 8)
    keep[..., : SIDE // 16, : SIDE // 16] = 0.0
    assert torch.equal(control_context[..., 128:132], _pack(_patchify(keep)))
    masked = _pixels(_inpaint_image())
    masked[..., : SIDE // 2, : SIDE // 2] = 0.0
    torch.testing.assert_close(control_context[..., 132:], _normalized_encoding(vae, masked), **BF16)


def test_a_control_encode_that_runs_out_of_memory_is_retried_in_tiles() -> None:
    """As the FLUX.2 image-to-latents node does: the generation-size encode is the one most likely not to fit."""
    vae = _tiny_vae()
    encode = vae.encode
    tiling_at_call: list[bool] = []

    def out_of_memory_once(*args, **kwargs):
        tiling_at_call.append(vae.use_tiling)
        if len(tiling_at_call) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return encode(*args, **kwargs)

    vae.encode = out_of_memory_once
    context = _context(vae=vae)
    bn_mean, bn_std = vae.bn.running_mean, torch.sqrt(vae.bn.running_var + vae.bn.eps)

    packed = _denoise()._build_control_context(
        context, bn_mean, bn_std, image_seq_len=IMAGE_TOKENS, ref_image_seq_len=0
    )

    # The control image once untiled and once tiled, then the masked inpaint image untiled again.
    assert tiling_at_call == [False, True, False]
    assert packed.shape == (1, IMAGE_TOKENS, FLUX2_FUN_CONTROL_IN_DIM)
    context.logger.warning.assert_called_once()
