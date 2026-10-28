"""The Qwen Image ControlNet node, and what the denoise does with its field: refuses the pairings the branch was not
trained for, reserves the hints and the score matrices beside the activations, locks the branch with the same
reservation inside the exit stack, gates it per step, and builds the packed control context from the field's images
the way it packs its latents.

The denoise runs its transformer in bfloat16, which the CPU test fleet cannot matmul, so every run here stops at the
transformer's lock or at the LoRA patcher, or runs the loop with the transformer's forward stood in. The forward with
the hooks is covered in `tests/backend/fun_controlnet`.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from PIL import Image
from pydantic import ValidationError

from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.model import ModelIdentifierField, VAEField
from invokeai.app.invocations.qwen_image import qwen_image_denoise
from invokeai.app.invocations.qwen_image.qwen_image_control import QwenImageControlField, QwenImageControlInvocation
from invokeai.app.invocations.qwen_image.qwen_image_denoise import QwenImageDenoiseInvocation
from invokeai.app.invocations.vae.qwen_image_image_to_latents import QwenImageImageToLatentsInvocation
from invokeai.backend.fun_controlnet.hooks import FunControlHooks
from invokeai.backend.fun_controlnet.qwen_image import QwenImageFunControlBranch
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    QwenImageConditioningInfo,
)
from invokeai.backend.util import attention
from invokeai.backend.util.attention import SDPA_MATH_BYTES_PER_SCORE_ELEMENT
from invokeai.backend.util.devices import TorchDevice

SIDE = 64  # an 8x8 latent: 16 image tokens
LATENT_SIDE = SIDE // 8
TEXT_TOKENS = 4
CONTROL_LAYERS = (0, 2)
HIDDEN = 16


class _Stop(Exception):
    """Raised where the node has made the decision a test looks at."""


def _model(type_: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(
        key=f"{type_.value}", hash="hash", name=type_.value, base=BaseModelType.QwenImage, type=type_
    )


# --- the node ---


def test_the_node_hands_its_images_and_settings_to_the_field() -> None:
    node = QwenImageControlInvocation(
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

    assert node.invoke(MagicMock()).control == QwenImageControlField(
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
    "images",
    [
        {"image": ImageField(image_name="canny")},
        {"inpaint_image": ImageField(image_name="photo"), "inpaint_mask": ImageField(image_name="mask")},
    ],
    ids=["control-alone", "inpaint-alone"],
)
def test_a_control_image_alone_or_an_inpaint_pair_alone_is_enough(images) -> None:
    node = QwenImageControlInvocation(id="control", control_model=_model(ModelType.ControlNet), **images)
    field = node.invoke(MagicMock()).control
    assert (field.image_name, field.inpaint_image_name, field.inpaint_mask_name) == (
        "canny" if "image" in images else None,
        "photo" if "inpaint_image" in images else None,
        "mask" if "inpaint_mask" in images else None,
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
    node = QwenImageControlInvocation(id="control", control_model=_model(ModelType.ControlNet), **images)
    with pytest.raises(ValueError, match=message):
        node.invoke(MagicMock())


def test_a_begin_after_the_end_is_refused_when_the_node_is_built() -> None:
    with pytest.raises(ValidationError, match="Begin step percent"):
        QwenImageControlInvocation(
            id="control", control_model=_model(ModelType.ControlNet), begin_step_percent=0.8, end_step_percent=0.2
        )


# --- the denoise ---


@pytest.fixture(autouse=True)
def cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))


@pytest.fixture(autouse=True)
def fused_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the dispatch questions `sdpa_score_matrix_bytes` asks, so no test depends on the torch build of the
    machine it runs on: fused, unless a test says otherwise."""
    monkeypatch.setattr(attention, "_diffusers_attention_dispatch", lambda: attention._DISPATCH_TORCH)
    monkeypatch.setattr(attention, "rocm_sdpa_uses_math_kernel", lambda *_args: False)
    monkeypatch.setattr(attention, "_torch_sdpa_materializes_score_matrix", lambda *_args: False)


def _tiny_transformer(zero_cond_t: bool = False) -> QwenImageTransformer2DModel:
    return QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=64,
        out_channels=16,
        num_layers=4,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        axes_dims_rope=(2, 4, 2),
        zero_cond_t=zero_cond_t,
    ).eval()


def _tiny_branch() -> QwenImageFunControlBranch:
    return QwenImageFunControlBranch(
        hidden_size=HIDDEN, num_attention_heads=2, attention_head_dim=8, control_layers=CONTROL_LAYERS
    ).eval()


def _tiny_vae() -> AutoencoderKLQwenImage:
    # The smallest configuration that keeps the real 8x spatial compression and 16 latent channels.
    return AutoencoderKLQwenImage(
        base_dim=4,
        z_dim=16,
        dim_mult=[1, 1, 1, 1],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True],
    ).eval()


def _loaded(model, lock) -> MagicMock:
    """A `LoadedModel` stand-in: `model` unlocked, `model_on_device` as `lock` says (a context or a raise)."""
    info = MagicMock()
    info.model = model
    info.compute_device = torch.device("cpu")
    info.model_on_device = MagicMock(side_effect=lock)
    return info


def _yielding(model):
    return lambda working_mem_bytes=None: nullcontext((None, model))


def _stopping(working_mem_bytes=None):
    raise _Stop


def _control_field(**overrides) -> QwenImageControlField:
    field = {
        "image_name": "canny",
        "inpaint_image_name": "photo",
        "inpaint_mask_name": "mask",
        "control_model": _model(ModelType.ControlNet),
    }
    field.update(overrides)
    return QwenImageControlField(**field)


def _denoise(**overrides) -> QwenImageDenoiseInvocation:
    fields = {
        "latents": None,
        "reference_latents": None,
        "denoise_mask": None,
        "denoising_start": 0.0,
        "denoising_end": 1.0,
        "transformer": SimpleNamespace(transformer=_model(ModelType.Main), loras=[]),
        "positive_conditioning": SimpleNamespace(conditioning_name="pos"),
        "negative_conditioning": SimpleNamespace(conditioning_name="neg"),
        "cfg_scale": 4.0,
        "width": SIDE,
        "height": SIDE,
        "steps": 2,
        "seed": 0,
        "shift": None,
        "control": _control_field(),
        "vae": VAEField(vae=_model(ModelType.VAE)),
    }
    fields.update(overrides)
    return QwenImageDenoiseInvocation.model_construct(**fields)


def _control_image() -> Image.Image:
    return Image.linear_gradient("L").resize((SIDE // 2, SIDE // 2)).convert("RGB")


def _inpaint_image() -> Image.Image:
    return Image.linear_gradient("L").rotate(90).resize((SIDE, SIDE)).convert("RGB")


def _inpaint_mask() -> Image.Image:
    """White (regenerate) in the top-left quarter."""
    mask = Image.new("L", (SIDE, SIDE), 0)
    mask.paste(255, (0, 0, SIDE // 2, SIDE // 2))
    return mask


def _context(tmp_path, transformer_info=None, control_info=None, vae_info=None, masked_prompt=False) -> MagicMock:
    infos = {ModelType.Main: transformer_info, ModelType.ControlNet: control_info, ModelType.VAE: vae_info}
    context = MagicMock()
    context.models.load.side_effect = lambda identifier: infos[identifier.type]
    context.models.get_config.return_value = SimpleNamespace(
        base=BaseModelType.QwenImage, type=ModelType.Main, format=ModelFormat.Checkpoint
    )
    # No `scheduler/` beside it: the node falls back to the Qwen-Image schedule defaults.
    context.models.get_absolute_path.return_value = tmp_path / "qwen_image.safetensors"
    prompt_mask = torch.ones(1, TEXT_TOKENS, dtype=torch.bool) if masked_prompt else None
    context.conditioning.load.side_effect = lambda name: ConditioningFieldData(
        conditionings=[
            QwenImageConditioningInfo(prompt_embeds=torch.zeros(1, TEXT_TOKENS, 8), prompt_embeds_mask=prompt_mask)
        ]
    )
    images = {"canny": _control_image(), "photo": _inpaint_image(), "mask": _inpaint_mask()}
    context.images.get_pil.side_effect = lambda name: images[name]
    context.config.get.return_value.force_tiled_decode = False
    return context


def _hook_count(transformer) -> int:
    return sum(len(block._forward_hooks) + len(block._forward_pre_hooks) for block in transformer.transformer_blocks)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"vae": None}, "needs the VAE"),
        ({"reference_latents": SimpleNamespace(latents_name="reference")}, "reference latents"),
    ],
    ids=["no-vae", "reference-latents"],
)
def test_control_is_refused_without_its_vae_or_beside_reference_latents(tmp_path, overrides, message) -> None:
    context = _context(tmp_path, transformer_info=_loaded(_tiny_transformer(), _stopping))

    with pytest.raises(ValueError, match=message):
        _denoise(**overrides)._run_diffusion(context)

    # Only the transformer was loaded: nothing else is fetched for a run that cannot happen.
    assert context.models.load.call_count == 1


def test_control_is_refused_on_an_edit_transformer(tmp_path) -> None:
    context = _context(tmp_path, transformer_info=_loaded(_tiny_transformer(zero_cond_t=True), _stopping))

    with pytest.raises(ValueError, match="Edit transformer"):
        _denoise()._run_diffusion(context)

    assert context.models.load.call_count == 1


def test_without_control_the_lock_reserves_the_activations_alone(tmp_path) -> None:
    transformer_info = _loaded(_tiny_transformer(), _stopping)
    context = _context(tmp_path, transformer_info=transformer_info)

    with pytest.raises(_Stop):
        _denoise(control=None, vae=None)._run_diffusion(context)

    expected = QwenImageDenoiseInvocation._estimate_working_memory(
        image_seq_len=16, text_seq_len=TEXT_TOKENS, element_size=torch.bfloat16.itemsize
    )
    assert transformer_info.model_on_device.call_args.kwargs["working_mem_bytes"] == expected


def test_with_control_the_lock_reserves_the_hints_too_and_the_images_are_encoded_first(tmp_path) -> None:
    transformer_info = _loaded(_tiny_transformer(), _stopping)
    branch, vae = _tiny_branch(), _tiny_vae()
    vae_info = _loaded(vae, _yielding(vae))
    context = _context(tmp_path, transformer_info, _loaded(branch, _yielding(branch)), vae_info)

    with pytest.raises(_Stop):
        _denoise()._run_diffusion(context)

    expected = QwenImageDenoiseInvocation._estimate_working_memory(
        image_seq_len=16,
        text_seq_len=TEXT_TOKENS,
        element_size=torch.bfloat16.itemsize,
        num_control_blocks=len(CONTROL_LAYERS),
        control_hidden_size=HIDDEN,
    )
    assert transformer_info.model_on_device.call_args.kwargs["working_mem_bytes"] == expected
    # The control image and the masked inpaint image went through the VAE before the transformer was locked.
    assert vae_info.model_on_device.call_count == 2


def test_a_build_that_materializes_the_scores_reserves_them_too(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ROCm on Windows and MPS have no fused kernel for the joint sequence; the 1 GiB base does not cover 24 heads of
    4608^2 scores, so the probe's answer has to reach the reservation."""
    monkeypatch.setattr(attention, "_torch_sdpa_materializes_score_matrix", lambda *_args: True)
    transformer_info = _loaded(_tiny_transformer(), _stopping)
    context = _context(tmp_path, transformer_info=transformer_info)

    with pytest.raises(_Stop):
        _denoise(control=None, vae=None)._run_diffusion(context)

    activations = QwenImageDenoiseInvocation._estimate_working_memory(
        image_seq_len=16, text_seq_len=TEXT_TOKENS, element_size=torch.bfloat16.itemsize
    )
    # The tiny transformer's two heads over the 16 + 4 token joint sequence.
    scores = 2 * (16 + TEXT_TOKENS) ** 2 * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
    assert transformer_info.model_on_device.call_args.kwargs["working_mem_bytes"] == activations + scores


@pytest.mark.parametrize("masked_prompt", [False, True], ids=["no-mask", "mask"])
def test_the_prompt_mask_reaches_the_attention_probe(tmp_path, monkeypatch: pytest.MonkeyPatch, masked_prompt) -> None:
    """A mask narrows which fused kernels a build can use, so the probe is told whether the encoder emitted one."""
    asked_with: list[bool] = []
    monkeypatch.setattr(
        attention, "_torch_sdpa_materializes_score_matrix", lambda *args: asked_with.append(args[-1]) or False
    )
    context = _context(tmp_path, transformer_info=_loaded(_tiny_transformer(), _stopping), masked_prompt=masked_prompt)

    with pytest.raises(_Stop):
        _denoise(control=None, vae=None)._run_diffusion(context)

    assert asked_with == [masked_prompt]


def test_the_branch_is_locked_with_the_same_reservation_and_hooked_inside_the_exit_stack(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformer, branch, vae = _tiny_transformer(), _tiny_branch(), _tiny_vae()
    transformer_info = _loaded(transformer, _yielding(transformer))
    control_info = _loaded(branch, _yielding(branch))
    context = _context(tmp_path, transformer_info, control_info, _loaded(vae, _yielding(vae)))
    hooks_at_patching: list[int] = []

    def stop(**_kwargs):
        hooks_at_patching.append(_hook_count(transformer))
        raise _Stop

    monkeypatch.setattr(LayerPatcher, "apply_smart_model_patches", stop)

    with pytest.raises(_Stop):
        _denoise()._run_diffusion(context)

    reserved = transformer_info.model_on_device.call_args.kwargs["working_mem_bytes"]
    assert control_info.model_on_device.call_args.kwargs["working_mem_bytes"] == reserved
    assert hooks_at_patching == [1 + len(CONTROL_LAYERS)]
    assert _hook_count(transformer) == 0


def test_the_first_preview_is_sent_once_the_transformer_and_the_branch_are_locked(tmp_path) -> None:
    """A Tiny AutoEncoder preview places its TAE next to the locked models; before the locks there are none to fit."""
    transformer, branch, vae = _tiny_transformer(), _tiny_branch(), _tiny_vae()
    events: list[str] = []

    def locking(name: str, model):
        def lock(working_mem_bytes=None):
            events.append(f"lock {name}")
            return nullcontext((None, model))

        return lock

    context = _context(
        tmp_path,
        _loaded(transformer, locking("transformer", transformer)),
        _loaded(branch, locking("branch", branch)),
        _loaded(vae, _yielding(vae)),
    )

    def preview(*_args) -> None:
        events.append("preview")
        raise _Stop

    context.util.sd_step_callback.side_effect = preview

    with pytest.raises(_Stop):
        _denoise()._run_diffusion(context)

    assert events == ["lock transformer", "lock branch", "preview"]


class _GateRecordingTransformer(QwenImageTransformer2DModel):
    """The real class, so the node's isinstance checks hold and the real hooks land on its blocks, with the forward
    replaced: no bfloat16 matmul on the CPU, and a record of the control gate at every call."""

    def __init__(self) -> None:
        super().__init__(
            patch_size=2,
            in_channels=64,
            out_channels=16,
            num_layers=4,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=8,
            axes_dims_rope=(2, 4, 2),
        )
        self.hooks: FunControlHooks | None = None
        self.gates: list[bool] = []

    def forward(self, hidden_states: torch.Tensor, *_args, **_kwargs):
        assert self.hooks is not None
        self.gates.append(self.hooks.active)
        return (torch.zeros(*hidden_states.shape[:2], 4 * self.out_channels, dtype=hidden_states.dtype),)


@pytest.mark.parametrize(
    ("begin", "end", "expected_gates"),
    [
        (0.5, 1.0, [False] * 4 + [True] * 4),
        (0.0, 0.5, [True] * 6 + [False] * 2),
    ],
    ids=["second-half", "first-half"],
)
def test_the_gate_is_set_once_per_step_for_both_guidance_branches(
    tmp_path, monkeypatch: pytest.MonkeyPatch, begin, end, expected_gates
) -> None:
    """Four steps with CFG are eight forwards: the gate follows `step_index / total_steps` with both ends included,
    is re-evaluated every step, and the positive and the negative forward of a step see the same one."""
    transformer, branch, vae = _GateRecordingTransformer(), _tiny_branch(), _tiny_vae()
    install = qwen_image_denoise.install_fun_control

    @contextmanager
    def capturing_install(*args, **kwargs):
        with install(*args, **kwargs) as hooks:
            transformer.hooks = hooks
            yield hooks

    monkeypatch.setattr(qwen_image_denoise, "install_fun_control", capturing_install)
    context = _context(
        tmp_path,
        _loaded(transformer, _yielding(transformer)),
        _loaded(branch, _yielding(branch)),
        _loaded(vae, _yielding(vae)),
    )

    _denoise(steps=4, control=_control_field(begin_step_percent=begin, end_step_percent=end))._run_diffusion(context)

    assert transformer.gates == expected_gates


@pytest.mark.parametrize(
    ("tiled", "tile_size", "force_tiled_decode", "expected_tiled"),
    [(False, 0, False, False), (True, 512, False, True), (False, 0, True, True)],
    ids=["untiled", "field", "forced"],
)
def test_the_control_images_are_encoded_with_the_fields_tiling(
    tmp_path, monkeypatch: pytest.MonkeyPatch, tiled, tile_size, force_tiled_decode, expected_tiled
) -> None:
    vae = _tiny_vae()
    context = _context(tmp_path, vae_info=_loaded(vae, _yielding(vae)))
    context.config.get.return_value.force_tiled_decode = force_tiled_decode
    encode = QwenImageImageToLatentsInvocation.vae_encode
    encoded_with: list[tuple[bool, int]] = []

    def recording(**kwargs):
        encoded_with.append((kwargs["tiled"], kwargs["tile_size"]))
        return encode(**kwargs)

    monkeypatch.setattr(QwenImageImageToLatentsInvocation, "vae_encode", staticmethod(recording))

    _denoise(control=_control_field(tiled=tiled, tile_size=tile_size))._build_control_context(
        context, LATENT_SIDE, LATENT_SIDE
    )

    # Once for the control image, once for the masked inpaint image, both with the field's tiling.
    assert encoded_with == [(expected_tiled, tile_size)] * 2


def _as_denoise_input(image: Image.Image, mode: str) -> torch.Tensor:
    """What the node feeds the VAE: the image at generation size, pixels in [-1, 1], a mask in [0, 1]."""
    resample = Image.Resampling.LANCZOS if mode == "RGB" else Image.Resampling.BILINEAR
    resized = image.convert(mode).resize((SIDE, SIDE), resample)
    return image_resized_to_grid_as_tensor(resized, normalize=mode == "RGB", multiple_of=16).unsqueeze(0)


def test_the_control_context_is_the_encoded_images_packed_like_the_latents(tmp_path) -> None:
    vae = _tiny_vae()
    vae_info = _loaded(vae, _yielding(vae))
    context = _context(tmp_path, vae_info=vae_info)

    packed = _denoise()._build_control_context(context, LATENT_SIDE, LATENT_SIDE)

    assert packed.shape == (1, (LATENT_SIDE // 2) ** 2, 33 * 4)
    # Channel-major within a token: feature c * 4 + dy * 2 + dx of token (i, j) is channel c at (2i + dy, 2j + dx).
    context_4d = QwenImageDenoiseInvocation._unpack_latents(packed, LATENT_SIDE, LATENT_SIDE)
    assert context_4d.shape == (1, 33, LATENT_SIDE, LATENT_SIDE)
    for channel, i, j, dy, dx in [(0, 0, 0, 0, 0), (16, 1, 2, 1, 0), (32, 3, 3, 1, 1), (7, 2, 1, 0, 1)]:
        token = i * (LATENT_SIDE // 2) + j
        assert packed[0, token, channel * 4 + dy * 2 + dx] == context_4d[0, channel, 2 * i + dy, 2 * j + dx]

    def encode(pixels: torch.Tensor) -> torch.Tensor:
        return QwenImageImageToLatentsInvocation.vae_encode(vae_info, pixels).squeeze(2)

    assert torch.equal(context_4d[:, :16], encode(_as_denoise_input(_control_image(), "RGB")))
    expected_keep = torch.ones(LATENT_SIDE, LATENT_SIDE)
    expected_keep[: LATENT_SIDE // 2, : LATENT_SIDE // 2] = 0.0
    assert torch.equal(context_4d[0, 16], expected_keep)
    masked = _as_denoise_input(_inpaint_image(), "RGB")
    masked[..., : SIDE // 2, : SIDE // 2] = 0.0
    assert torch.equal(context_4d[:, 17:], encode(masked))
