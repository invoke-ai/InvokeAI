import math
from typing import Optional

import torch
import torchvision.transforms as tv_transforms
from pydantic import field_validator
from torchvision.transforms.functional import resize as tv_resize
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    QwenImage21ConditioningField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.startup_utils import log_attention_backends
from invokeai.backend.flux.sampling_utils import clip_timestep_schedule_fractional
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.qwen_image_2_1.sampling import build_sigmas
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import QwenImage21ConditioningInfo
from invokeai.backend.util.attention import sdpa_score_matrix_bytes
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger

LATENT_CHANNELS = 64
VAE_SCALE_FACTOR = 16
# One vision-language slot of the transformer's joint sequence covers 2x2 latent tokens.
TOKENS_PER_SLOT = 4

logger = InvokeAILogger.get_logger(__name__)


def clip_sigmas(sigmas: list[float], denoising_start: float, denoising_end: float) -> tuple[list[float], int]:
    """The part of the schedule from noise level `1 - denoising_start` to `1 - denoising_end`, and the index of the
    configured step it starts in.

    Picked by noise level, not by step index: Turbo's distilled table sits above 0.84 for six of its eight steps, so a
    start picked by index would begin a moderate-strength image-to-image almost from pure noise. By level, a strength
    means the same noise for both variants, and a range shorter than a step still runs one step.
    """
    clipped = clip_timestep_schedule_fractional(list(sigmas), denoising_start, denoising_end)
    # The configured step `clipped[0]` falls in: the last one starting at or above the start level.
    first_step = sum(1 for sigma in sigmas if sigma >= 1.0 - denoising_start - 1e-6) - 1
    return clipped, first_step


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, h, w) -> (B, h*w, C): one token per latent pixel, in raster order (patch size 1)."""
    batch, channels, height, width = latents.shape
    return latents.reshape(batch, channels, height * width).transpose(1, 2)


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(B, h*w, C) -> (B, C, h, w)."""
    batch, _, channels = latents.shape
    return latents.transpose(1, 2).reshape(batch, channels, height, width)


@invocation(
    "qwen_image_2_1_denoise",
    title="Denoise - Qwen-Image-2.1",
    tags=["image", "qwen_image_2_1", "qwen-image-2.1"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class QwenImage21DenoiseInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Run the denoising process with a Qwen-Image-2.1 model."""

    latents: Optional[LatentsField] = InputField(
        default=None, description=FieldDescriptions.latents, input=Input.Connection
    )
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None, description=FieldDescriptions.denoise_mask, input=Input.Connection
    )
    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    transformer: TransformerField = InputField(
        description=FieldDescriptions.qwen_image_2_1_model, input=Input.Connection, title="Transformer"
    )
    positive_conditioning: QwenImage21ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: Optional[QwenImage21ConditioningField] = InputField(
        default=None, description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    # True CFG: neg + cfg_scale * (cond - neg), only with a negative prompt and a scale above 1. The model card
    # samples without guidance.
    cfg_scale: float | list[float] = InputField(default=1.0, description=FieldDescriptions.cfg_scale, title="CFG Scale")
    width: int = InputField(default=1024, gt=0, multiple_of=32, description="Width of the generated image.")
    height: int = InputField(default=1024, gt=0, multiple_of=32, description="Height of the generated image.")
    steps: int = InputField(
        default=40,
        gt=0,
        description=f"{FieldDescriptions.steps} Turbo is distilled for its shipped 8-step schedule; other counts "
        "resample it.",
    )
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")

    @field_validator("cfg_scale")
    @classmethod
    def validate_cfg_scale_is_finite(cls, value: float | list[float]) -> float | list[float]:
        values = value if isinstance(value, list) else [value]
        if not all(math.isfinite(item) for item in values):
            raise ValueError("cfg_scale values must be finite.")
        return value

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context).detach().to("cpu")
        name = context.tensors.save(tensor=latents)
        # `LatentsOutput.build` assumes 8x latents; these are 16x, so report the requested size directly.
        return LatentsOutput(
            latents=LatentsField(latents_name=name, seed=self.seed), width=self.width, height=self.height
        )

    def _load_conditioning(
        self, context: InvocationContext, field: QwenImage21ConditioningField, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        data = context.conditioning.load(field.conditioning_name)
        if len(data.conditionings) != 1 or not isinstance(data.conditionings[0], QwenImage21ConditioningInfo):
            raise ValueError("Expected exactly one Qwen-Image-2.1 conditioning.")
        return data.conditionings[0].to(dtype=dtype, device=device).prompt_embeds

    def _prepare_cfg_scale(self, num_steps: int) -> list[float]:
        if isinstance(self.cfg_scale, list):
            if len(self.cfg_scale) != num_steps:
                raise ValueError(
                    f"cfg_scale list has {len(self.cfg_scale)} values for {num_steps} steps. Provide one value per "
                    "configured step, or a single float."
                )
            return self.cfg_scale
        return [self.cfg_scale] * num_steps

    def _prep_inpaint_mask(self, context: InvocationContext, latents: torch.Tensor) -> torch.Tensor | None:
        if self.denoise_mask is None:
            return None
        mask = 1.0 - context.tensors.load(self.denoise_mask.mask_name)
        mask = tv_resize(
            img=mask,
            size=list(latents.shape[-2:]),
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
            antialias=False,
        )
        return mask.to(device=latents.device, dtype=latents.dtype)

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

        if self.denoising_start >= self.denoising_end:
            raise ValueError("denoising_start must be less than denoising_end.")
        if self.denoise_mask is not None and self.latents is None:
            raise ValueError("Initial latents are required when a denoise mask is provided.")

        device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_bfloat16_safe_dtype(device)
        transformer_config = context.models.get_config(self.transformer.transformer)
        transformer_info = context.models.load(self.transformer.transformer)

        latent_height, latent_width = self.height // VAE_SCALE_FACTOR, self.width // VAE_SCALE_FACTOR
        image_seq_len = latent_height * latent_width

        full_sigmas = build_sigmas(getattr(transformer_config, "variant", None), self.steps, image_seq_len)
        sigmas, first_step = clip_sigmas(full_sigmas.tolist(), self.denoising_start, self.denoising_end)
        sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
        cfg_scale = self._prepare_cfg_scale(len(full_sigmas) - 1)[first_step : first_step + len(sigmas) - 1]

        pos_embeds = self._load_conditioning(context, self.positive_conditioning, dtype, device)
        do_cfg = self.negative_conditioning is not None and any(value > 1.0 for value in cfg_scale)
        neg_embeds = (
            self._load_conditioning(context, self.negative_conditioning, dtype, device)
            if do_cfg and self.negative_conditioning is not None
            else None
        )

        # Drawn as the pipeline draws it, (B, frame, C, h, w), on the CPU so a seed means the same on any device.
        noise = torch.randn(
            (1, 1, LATENT_CHANNELS, latent_height, latent_width),
            generator=torch.Generator("cpu").manual_seed(self.seed),
            dtype=torch.float32,
        )[:, 0].to(device)

        init_latents = None
        if self.latents is not None:
            init_latents = context.tensors.load(self.latents.latents_name).to(device=device, dtype=torch.float32)
            if init_latents.dim() == 5:
                init_latents = init_latents.squeeze(2)
            if tuple(init_latents.shape[-2:]) != (latent_height, latent_width):
                raise ValueError(
                    f"Initial latents are {tuple(init_latents.shape[-2:])}, not the {(latent_height, latent_width)} "
                    f"a {self.width}x{self.height} image needs."
                )
            sigma_0 = sigmas[0].item()
            latents = sigma_0 * noise + (1.0 - sigma_0) * init_latents
        elif self.denoising_start > 1e-5:
            raise ValueError("denoising_start must be 0 when no initial latents are given.")
        else:
            latents = noise

        inpaint_mask = self._prep_inpaint_mask(context, noise)
        inpaint_extension = None
        if inpaint_mask is not None:
            assert init_latents is not None
            inpaint_extension = RectifiedFlowInpaintExtension(
                init_latents=init_latents, inpaint_mask=inpaint_mask, noise=noise
            )

        # The pipeline keeps the sampler state in the inference dtype and steps it in fp32 (the scheduler's
        # upcast); matching that is what makes this node reproduce its outputs.
        latents = pack_latents(latents).to(dtype)

        def image_slot_mask(text_len: int) -> torch.Tensor:
            # The joint sequence's slots: the text, then one slot per 2x2 group of target latents. Condition
            # images would sit between the two.
            text = torch.zeros(1, text_len, dtype=torch.bool, device=device)
            target = torch.ones(1, image_seq_len // TOKENS_PER_SLOT, dtype=torch.bool, device=device)
            return torch.cat([text, target], dim=1)

        img_shapes = [[(1, latent_height, latent_width)]]
        pos_img_mask = image_slot_mask(pos_embeds.shape[1])
        neg_img_mask = image_slot_mask(neg_embeds.shape[1]) if neg_embeds is not None else None

        working_memory = self._estimate_working_memory(
            image_seq_len, pos_embeds.shape[1], None if neg_embeds is None else neg_embeds.shape[1], do_cfg
        )
        working_memory += self._attention_score_bytes(
            transformer_info.model,
            image_seq_len + max(pos_embeds.shape[1], 0 if neg_embeds is None else neg_embeds.shape[1]),
            device,
            dtype,
        )
        working_memory += peak_dequant_transient_bytes(transformer_info.model, dtype)
        log_attention_backends(logger, device)

        total_steps = len(sigmas) - 1
        step_callback = self._step_callback(context)
        step_callback(
            PipelineIntermediateState(
                step=0,
                order=1,
                total_steps=total_steps,
                timestep=int(sigmas[0].item() * 1000),
                latents=unpack_latents(latents, latent_height, latent_width),
            )
        )

        with transformer_info.model_on_device(working_mem_bytes=working_memory) as (_, transformer):
            # Text keys and values do not depend on the step under `causal_condition`: the first step prefills
            # them, the rest compute the image tokens only. One cache per guidance branch, for this run only.
            num_blocks = len(transformer.transformer_blocks)
            use_cache = bool(transformer.config.causal_condition)
            pos_cache = QwenImage21KVCache(num_blocks) if use_cache else None
            neg_cache = QwenImage21KVCache(num_blocks) if use_cache and neg_embeds is not None else None

            def predict(embeds: torch.Tensor, img_mask: torch.Tensor, cache, mode, timestep) -> torch.Tensor:
                out = transformer(
                    hidden_states=latents,
                    encoder_hidden_states=embeds,
                    encoder_hidden_states_mask=None,
                    timestep=timestep,
                    img_shapes=img_shapes,
                    img_mask=img_mask,
                    kv_cache=cache,
                    kv_cache_mode=mode,
                    return_dict=False,
                )[0]
                return out[:, -latents.shape[1] :]

            for step_idx in tqdm(range(total_steps)):
                sigma, sigma_next = sigmas[step_idx], sigmas[step_idx + 1]
                # The pipeline hands the transformer `timestep / 1000` after rounding the timestep to the
                # inference dtype; the embedder multiplies it back by 1000.
                timestep = (sigma * 1000).expand(1).to(dtype) / 1000
                mode = None if not use_cache else ("extract" if step_idx == 0 else "cached")

                noise_pred = predict(pos_embeds, pos_img_mask, pos_cache, mode, timestep)
                if neg_embeds is not None and cfg_scale[step_idx] > 1.0:
                    neg_pred = predict(neg_embeds, neg_img_mask, neg_cache, mode, timestep)
                    noise_pred = neg_pred + cfg_scale[step_idx] * (noise_pred - neg_pred)
                elif neg_cache is not None and mode == "extract":
                    # A per-step scale that starts at or below 1 would skip the negative prefill and leave a
                    # later step decoding from an empty cache.
                    predict(neg_embeds, neg_img_mask, neg_cache, mode, timestep)

                # FlowMatchEulerDiscreteScheduler.step to the bit: the 0-dim fp32 step times the prediction stays in
                # the prediction's dtype, and only the sum with the fp32 sample is fp32.
                latents = (latents.float() + (sigma_next - sigma) * noise_pred).to(dtype)

                if inpaint_extension is not None:
                    merged = inpaint_extension.merge_intermediate_latents_with_init_latents(
                        unpack_latents(latents, latent_height, latent_width).float(), sigma_next.item()
                    )
                    latents = pack_latents(merged).to(dtype)

                step_callback(
                    PipelineIntermediateState(
                        step=step_idx + 1,
                        order=1,
                        total_steps=total_steps,
                        timestep=int(sigma.item() * 1000),
                        latents=unpack_latents(latents, latent_height, latent_width),
                    )
                )

        return unpack_latents(latents, latent_height, latent_width).float()

    @staticmethod
    def _estimate_working_memory(image_seq_len: int, pos_text_len: int, neg_text_len: int | None, do_cfg: bool) -> int:
        """Peak transformer activations, in bytes, for the model cache to keep free.

        Measured on an RTX 4090 in bf16 with the transformer fully resident: 0.55 GiB of activations at
        1024x1024 (4096 image tokens) and 2.2 GiB at 2048x2048 (16384), about 0.14 MiB per token. 0.25 MiB per
        token plus a 1 GiB base leaves room for the allocator and the prefix cache, which holds K and V of every
        text token in all 32 blocks (~0.5 MiB per text token, per guidance branch).
        """
        mib, gib = 1024**2, 1024**3
        text_len = max(pos_text_len, neg_text_len or 0)
        estimated = int((image_seq_len + text_len) * 0.25 * mib) + gib
        branches = 2 if do_cfg else 1
        estimated += int(branches * (pos_text_len if neg_text_len is None else text_len) * 0.5 * mib)
        return estimated

    @staticmethod
    def _attention_score_bytes(transformer: object, seq_len: int, device: torch.device, dtype: torch.dtype) -> int:
        """The score matrix attention materializes on a build without a fused kernel for it (0 where it has one)."""
        config = getattr(transformer, "config", None)
        num_heads = getattr(config, "num_attention_heads", None)
        head_dim = getattr(config, "attention_head_dim", None)
        if not isinstance(num_heads, int) or not isinstance(head_dim, int):
            return 0
        # The largest call, the image queries over every key, goes unmasked both in the prefill and on decode steps:
        # only the text segments carry a mask, and they are as short as the prompt. All of them go through diffusers'
        # attention dispatch.
        return sdpa_score_matrix_bytes(
            device=device,
            dtype=dtype,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            via_diffusers_dispatch=True,
        )

    @staticmethod
    def _step_callback(context: InvocationContext):
        def callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.QwenImage21)

        return callback
