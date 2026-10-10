import itertools
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


def slot_runs(image_pad_mask: torch.Tensor | None) -> list[int]:
    """The lengths of the runs of image slots in a prompt's mask: one run per reference image, in order."""
    if image_pad_mask is None:
        return []
    return [len(list(run)) for is_slot, run in itertools.groupby(image_pad_mask[0].tolist()) if is_slot]


def _describe_grids(grids: list[tuple[int, int]]) -> str:
    # A slot is 32x32 pixels of the reference as the encoder read it.
    return ", ".join(f"{columns * 32}x{rows * 32}" for rows, columns in grids) or "none"


def check_reference_slots(
    info: QwenImage21ConditioningInfo, reference_shapes: list[tuple[int, int, int]], prompt: str
) -> None:
    """Refuse references whose latents do not match the prompt's image slots, reference by reference.

    Each reference's latents fill one run of the prompt's slots, 2x2 latents per slot, at the slot grid the
    encoder read it at. A mismatch means the prompt was encoded with other images, in another order or at another
    size than the latents: the transformer would place one reference's latents in another's slots.
    """
    expected = [(h // 2, w // 2) for _, h, w in reference_shapes]
    found = list(info.reference_grids)
    if found != expected or slot_runs(info.image_pad_mask) != [rows * columns for rows, columns in found]:
        raise ValueError(
            f"The {prompt} prompt was encoded with {len(found)} reference image(s) ({_describe_grids(found)}), but "
            f"the denoise node received {len(expected)} reference latent(s) ({_describe_grids(expected)}). Encode "
            "each reference with Image to Latents in reference mode, from the same images and in the same order "
            "as the prompts."
        )


def prefix_length(info: QwenImage21ConditioningInfo) -> int:
    """Tokens before the target in the joint sequence: the prompt, each reference slot widened to 2x2 latents."""
    slots = 0 if info.image_pad_mask is None else int(info.image_pad_mask.sum())
    return info.prompt_embeds.shape[1] + (TOKENS_PER_SLOT - 1) * slots


# K and V of one prefix token in all 32 blocks: 2 x 4096 x 32 x 2 bytes in bf16.
KV_BYTES_PER_PREFIX_TOKEN = 2 * 4096 * 32 * 2


def prefix_cache_fits(cache_bytes: int, device: torch.device) -> bool:
    """Whether the prefix caches can stay on the GPU for the whole run.

    They cannot be offloaded the way the transformer's weights can, so caches larger than half the card would
    leave too little for the model however it streams: four references at CFG above 1 hold ~16 GiB. Past that the
    prefix is recomputed every step instead -- slower, but within the memory of an ordinary run. Memory that is
    not a dedicated card's (CPU, MPS) is not limited here.
    """
    if device.type != "cuda":
        return True
    return cache_bytes <= torch.cuda.get_device_properties(device).total_memory // 2


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
    version="1.1.0",
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
    reference_latents: LatentsField | list[LatentsField] | None = InputField(
        default=None,
        description="Latents of the reference images an edit reads, from Image to Latents in reference mode, in the "
        "order the prompts were encoded with them.",
        input=Input.Connection,
        title="Reference Latents",
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
    ) -> QwenImage21ConditioningInfo:
        data = context.conditioning.load(field.conditioning_name)
        if len(data.conditionings) != 1 or not isinstance(data.conditionings[0], QwenImage21ConditioningInfo):
            raise ValueError("Expected exactly one Qwen-Image-2.1 conditioning.")
        return data.conditionings[0].to(dtype=dtype, device=device)

    def _load_references(self, context: InvocationContext, device: torch.device) -> list[torch.Tensor]:
        """The reference latents, `(1, C, h, w)` each, in the order the prompts were encoded with them."""
        if self.reference_latents is None:
            return []
        fields = self.reference_latents if isinstance(self.reference_latents, list) else [self.reference_latents]
        references = []
        for field in fields:
            reference = context.tensors.load(field.latents_name).to(device=device, dtype=torch.float32)
            if reference.dim() == 5:
                reference = reference.squeeze(2)
            if reference.shape[1] != LATENT_CHANNELS or reference.shape[-2] % 2 or reference.shape[-1] % 2:
                raise ValueError(
                    f"Reference latents are {tuple(reference.shape[1:])}; Qwen-Image-2.1 reads {LATENT_CHANNELS} "
                    "channels with even sides. Encode the reference with Image to Latents - Qwen-Image-2.1 in "
                    "reference mode."
                )
            references.append(reference)
        return references

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

        latent_height, latent_width = self.height // VAE_SCALE_FACTOR, self.width // VAE_SCALE_FACTOR
        image_seq_len = latent_height * latent_width

        full_sigmas = build_sigmas(getattr(transformer_config, "variant", None), self.steps, image_seq_len)
        sigmas, first_step = clip_sigmas(full_sigmas.tolist(), self.denoising_start, self.denoising_end)
        sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
        cfg_scale = self._prepare_cfg_scale(len(full_sigmas) - 1)[first_step : first_step + len(sigmas) - 1]

        pos = self._load_conditioning(context, self.positive_conditioning, dtype, device)
        do_cfg = self.negative_conditioning is not None and any(value > 1.0 for value in cfg_scale)
        neg = (
            self._load_conditioning(context, self.negative_conditioning, dtype, device)
            if do_cfg and self.negative_conditioning is not None
            else None
        )
        pos_embeds = pos.prompt_embeds
        neg_embeds = neg.prompt_embeds if neg is not None else None

        references = self._load_references(context, device)
        reference_shapes = [(1, r.shape[-2], r.shape[-1]) for r in references]
        check_reference_slots(pos, reference_shapes, "positive")
        if neg is not None:
            check_reference_slots(neg, reference_shapes, "negative")
        # Clean latents, condition images first: the transformer drops them into the prompt's image slots.
        reference_tokens = torch.cat([pack_latents(r) for r in references], dim=1).to(dtype) if references else None

        # Only now, with every input checked: loading reads the whole transformer into RAM.
        transformer_info = context.models.load(self.transformer.transformer)

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

        def image_slot_mask(info: QwenImage21ConditioningInfo) -> torch.Tensor:
            # The joint sequence's slots: the prompt with its reference slots, then one slot per 2x2 group of
            # target latents.
            text_len = info.prompt_embeds.shape[1]
            prompt = (
                info.image_pad_mask.bool()
                if info.image_pad_mask is not None
                else torch.zeros(1, text_len, dtype=torch.bool, device=device)
            )
            target = torch.ones(1, image_seq_len // TOKENS_PER_SLOT, dtype=torch.bool, device=device)
            return torch.cat([prompt, target], dim=1)

        img_shapes = [[*reference_shapes, (1, latent_height, latent_width)]]
        pos_img_mask = image_slot_mask(pos)
        neg_img_mask = image_slot_mask(neg) if neg is not None else None

        # Each reference slot widens to 2x2 latent tokens in the joint sequence; with the text they are the
        # prefix the cache holds.
        pos_prefix = prefix_length(pos)
        neg_prefix = prefix_length(neg) if neg is not None else None
        # Text and reference keys and values do not depend on the step under `causal_condition`: the first step
        # prefills them, the rest compute the image tokens only -- when the caches fit.
        cache_bytes = KV_BYTES_PER_PREFIX_TOKEN * (pos_prefix + (neg_prefix or 0))
        use_cache = bool(getattr(transformer_info.model.config, "causal_condition", False))
        if use_cache and not prefix_cache_fits(cache_bytes, device):
            use_cache = False
            logger.info(
                f"Qwen-Image-2.1: the prefix caches would hold {cache_bytes / 1024**3:.1f} GiB, more than half of "
                "this GPU; recomputing the prefix every step instead, which is slower."
            )
        working_memory = self._estimate_working_memory(image_seq_len, pos_prefix, neg_prefix, do_cfg, use_cache)
        working_memory += self._attention_score_bytes(
            transformer_info.model, image_seq_len, max(pos_prefix, neg_prefix or 0), device, dtype
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
            # One cache per guidance branch, for this run only.
            num_blocks = len(transformer.transformer_blocks)
            pos_cache = QwenImage21KVCache(num_blocks) if use_cache else None
            neg_cache = QwenImage21KVCache(num_blocks) if use_cache and neg_embeds is not None else None

            def predict(embeds: torch.Tensor, img_mask: torch.Tensor, cache, mode, timestep) -> torch.Tensor:
                hidden_states = latents if reference_tokens is None else torch.cat([reference_tokens, latents], dim=1)
                out = transformer(
                    hidden_states=hidden_states,
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
    def _estimate_working_memory(
        image_seq_len: int, pos_prefix: int, neg_prefix: int | None, do_cfg: bool, use_cache: bool
    ) -> int:
        """Peak transformer activations, in bytes, for the model cache to keep free.

        Measured on an RTX 4090 in bf16 with the transformer fully resident: 0.55 GiB of activations at
        1024x1024 (4096 image tokens) and 2.2 GiB at 2048x2048 (16384), about 0.14 MiB per token. 0.25 MiB per
        token plus a 1 GiB base leaves room for the allocator and the prefix cache, which holds K and V of every
        prefix token -- text and reference latents -- in all 32 blocks (~0.5 MiB per token, per guidance branch).
        """
        mib, gib = 1024**2, 1024**3
        prefix = max(pos_prefix, neg_prefix or 0)
        estimated = int((image_seq_len + prefix) * 0.25 * mib) + gib
        if use_cache:
            branches = 2 if do_cfg else 1
            estimated += branches * (pos_prefix if neg_prefix is None else prefix) * KV_BYTES_PER_PREFIX_TOKEN
        return estimated

    @staticmethod
    def _attention_score_bytes(
        transformer: object, image_seq_len: int, prefix: int, device: torch.device, dtype: torch.dtype
    ) -> int:
        """The score matrix attention materializes on a build without a fused kernel for it (0 where it has one)."""
        config = getattr(transformer, "config", None)
        num_heads = getattr(config, "num_attention_heads", None)
        head_dim = getattr(config, "attention_head_dim", None)
        if not isinstance(num_heads, int) or not isinstance(head_dim, int):
            return 0
        # The largest call, the image queries over every key, goes unmasked both in the prefill and on decode steps:
        # only the text segments carry a mask, and they are as short as the prompt. All of them go through diffusers'
        # attention dispatch. Its score matrix is image x (prefix + image), which the square helper prices at the
        # side with the same area.
        return sdpa_score_matrix_bytes(
            device=device,
            dtype=dtype,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_len=math.isqrt(image_seq_len * (image_seq_len + prefix)),
            via_diffusers_dispatch=True,
        )

    @staticmethod
    def _step_callback(context: InvocationContext):
        def callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.QwenImage21)

        return callback
