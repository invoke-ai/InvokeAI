"""Wan 2.2 video denoise invocation (T2V / I2V).

Multi-frame counterpart to :mod:`wan_denoise`. Drives the same flow-matching
schedule + expert-swap MoE logic, but the noise tensor has a real temporal
dimension (``T_lat = (num_frames - 1) // 4 + 1``) and the I2V conditioning is
built across all latent frames (first frame conditioned, rest zero).

Kept as a separate file rather than parameterizing ``WanDenoiseInvocation``
so the working single-frame T2I path is not risked by the video work; the
shared bits (expert swapper, scheduler construction, conditioning loading,
LoRA iteration) live in ``wan_denoise`` and are imported here.
"""

from contextlib import ExitStack
from typing import Callable, Iterable, Optional, Tuple

import torch
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    WanConditioningField,
    WanRefImageConditioningField,
)
from invokeai.app.invocations.model import WanTransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.invocations.wan_denoise import (
    WanDenoiseInvocation,
    _ExpertSwapper,
    _resolve_variant,
    _validate_spatial_dimensions,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, WanVariantType
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import WanConditioningInfo
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.wan.sampling_utils import (
    get_default_latent_channels,
    get_spatial_scale_factor,
    make_noise,
    num_latent_frames_for,
)


@invocation(
    "wan_video_denoise",
    title="Denoise Video - Wan 2.2",
    tags=["video", "wan"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class WanVideoDenoiseInvocation(BaseInvocation):
    """Run the Wan 2.2 denoising loop on a multi-frame latent tensor.

    The output is a 5D ``[1, C, T_lat, H/8, W/8]`` latent tensor ready for
    :class:`WanLatentsToVideoInvocation` to VAE-decode and encode as MP4.

    Mirrors :class:`WanDenoiseInvocation` for the per-step logic (CFG, MoE
    expert swap at the boundary timestep, LoRA patching, scheduler selection).
    Differences from the image denoise:

    * The noise tensor has a real temporal dim built from ``num_frames``.
    * The I2V condition is built across all latent frames (frame 0
      conditioned, rest zero) via
      :func:`encode_reference_image_to_video_condition` upstream — the
      ``ref_image`` field on this node carries a tensor of shape
      ``[1, 20, T_lat, H_lat, W_lat]`` instead of ``[1, 20, 1, ...]``.
    * No ``denoising_start`` / ``denoising_end`` / initial-latents inputs.
      The image denoise node uses those for img2img (noise injection on an
      existing latent), but image-conditioned video generation flows through
      the reference-frame conditioning mechanism instead — the first frame
      drives subsequent frames at every step, so a partial-schedule run from
      an initial latent has no analogue here. Run the schedule from t=1
      to t=0 every time. The base ``WanDenoiseInvocation`` still handles
      the img2img case for stills.
    """

    transformer: WanTransformerField = InputField(
        description=(
            "Wan transformer field. Supported: T2V-A14B / I2V-A14B (dual-expert) and "
            "TI2V-5B (single-expert, handles both T2V and I2V). All three accept a "
            "Reference Image input for image-to-video; A14B uses the 36-channel concat "
            "scheme while TI2V-5B uses the expand_timesteps first-frame-mask blend."
        ),
        input=Input.Connection,
        title="Transformer",
    )
    positive_conditioning: WanConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: Optional[WanConditioningField] = InputField(
        default=None, description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    ref_image: Optional[WanRefImageConditioningField] = InputField(
        default=None,
        description=FieldDescriptions.wan_ref_image,
        input=Input.Connection,
        title="Reference Image",
    )

    guidance_scale: float = InputField(
        default=5.0,
        ge=1.0,
        description="Classifier-free guidance scale. Wan 2.2 video reference uses 5.0 for the "
        "high-noise expert and 4.0 for the low-noise expert.",
        title="Guidance Scale",
    )
    guidance_scale_low_noise: Optional[float] = InputField(
        default=4.0,
        ge=0.0,
        description="Optional separate CFG scale for the low-noise expert (Wan 2.2 A14B only). "
        "Values below 1.0 fall back to the primary 'Guidance Scale'.",
        title="Guidance Scale (Low Noise)",
    )

    # Wan transformer patch_size=(1, 2, 2) × VAE spatial 8x => H/W multiple of 16.
    width: int = InputField(default=832, multiple_of=16, description="Width of the generated video.")
    height: int = InputField(default=480, multiple_of=16, description="Height of the generated video.")
    num_frames: int = InputField(
        default=81,
        ge=5,
        description="Number of output frames. Must satisfy (num_frames - 1) %% 4 == 0 so the latent "
        "temporal dim divides cleanly. Wan 2.2 was trained at 81 frames @ 16 FPS (~5 s).",
        title="Number of Frames",
    )
    steps: int = InputField(default=40, gt=0, description="Number of denoising steps.")
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        # Keep the 5D shape (B, C, T, H, W) — wan_latents_to_video expects it.
        latents = latents.detach().to("cpu")
        name = context.tensors.save(tensor=latents)
        # LatentsOutput.build uses latents.size()[3] / [2] for width / height.
        # For 5D the spatial dims are at indices 4 / 3 instead of 3 / 2, so we
        # call the constructor directly with the actual H/W from the inputs.
        from invokeai.app.invocations.fields import LatentsField

        return LatentsOutput(
            latents=LatentsField(latents_name=name, seed=self.seed),
            width=self.width,
            height=self.height,
        )

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        if (self.num_frames - 1) % 4 != 0:
            raise ValueError(
                f"num_frames must satisfy (num_frames - 1) %% 4 == 0 for the Wan VAE's temporal "
                f"compression (got {self.num_frames}). Try 5, 9, 13, ..., 81, 85, ..."
            )

        device = TorchDevice.choose_torch_device()
        inference_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        variant = _resolve_variant(context, self.transformer)
        _validate_spatial_dimensions(variant, self.width, self.height)
        spatial_scale = get_spatial_scale_factor(variant)

        # Reuse the image denoise's scheduler construction so we pick up whatever
        # scheduler the variant ships with (UniPCMultistepScheduler with the
        # variant's flow_shift, or whatever an on-disk config specifies).
        scheduler_builder = WanDenoiseInvocation._build_scheduler  # bound on instance below
        # Bind a minimal instance to call _build_scheduler — it only reads
        # self.transformer, which is shape-compatible.
        proxy = WanDenoiseInvocation.model_construct(
            transformer=self.transformer,
            positive_conditioning=self.positive_conditioning,
        )
        scheduler = scheduler_builder(proxy, context, device)

        pos_cond = self._load_conditioning(context, self.positive_conditioning, device=device, dtype=inference_dtype)
        low_cfg_enabled = (
            self.transformer.transformer_low_noise is not None
            and self.guidance_scale_low_noise is not None
            and self.guidance_scale_low_noise >= 1.0
            and self.guidance_scale_low_noise != 1.0
        )
        do_cfg = self.negative_conditioning is not None and (self.guidance_scale != 1.0 or low_cfg_enabled)
        neg_cond: WanConditioningInfo | None = None
        if do_cfg:
            assert self.negative_conditioning is not None
            neg_cond = self._load_conditioning(
                context, self.negative_conditioning, device=device, dtype=inference_dtype
            )

        # I2V condition tensor. Two flavours:
        #   * A14B I2V — [1, 20, T_lat, H_lat, W_lat] (4 mask + 16 latent channels).
        #     Concatenated to noise latents along the channel dim each step → 36ch.
        #   * TI2V-5B I2V — [1, 48, 1, H_lat, W_lat] (single latent frame, same
        #     channel count as the noise latents). Blended with noise via a
        #     first_frame_mask at every step (expand_timesteps path).
        # Variant dispatch happens via the condition tensor's channel count below.
        ref_condition: torch.Tensor | None = None
        if self.ref_image is not None:
            if variant not in (WanVariantType.I2V_A14B, WanVariantType.TI2V_5B):
                raise ValueError(
                    f"Reference-image conditioning is only supported by Wan 2.2 I2V variants "
                    f"(I2V-A14B or TI2V-5B). The selected transformer is {variant.value!r}. "
                    "Remove the Reference Image input or load an I2V variant."
                )
            if self.ref_image.width != self.width or self.ref_image.height != self.height:
                raise ValueError(
                    f"Reference-image dimensions ({self.ref_image.width}x{self.ref_image.height}) must "
                    f"match denoise dimensions ({self.width}x{self.height})."
                )
            # A14B encodes one condition tensor per pixel-frame count, so the
            # encoder's num_frames must match. TI2V-5B's condition is always
            # single-frame regardless of the output length, so the field's
            # num_frames is informational only and we skip this check.
            if variant == WanVariantType.I2V_A14B and self.ref_image.num_frames != self.num_frames:
                raise ValueError(
                    f"Reference-image num_frames ({self.ref_image.num_frames}) must match denoise "
                    f"num_frames ({self.num_frames}). Re-run the Reference Image - Wan 2.2 node with "
                    f"num_frames={self.num_frames}."
                )
            ref_condition = context.tensors.load(self.ref_image.condition_tensor_name).to(
                device=device, dtype=inference_dtype
            )

        scheduler.set_timesteps(num_inference_steps=self.steps, device=device)
        timesteps = scheduler.timesteps
        total_steps = len(timesteps)

        # fp32 latents through the loop; cast to inference_dtype only when
        # calling the transformer (same as wan_denoise).
        latent_dtype = torch.float32
        # 48 for TI2V-5B (Wan 2.2-VAE z_dim=48), 16 for A14B variants.
        latent_channels = get_default_latent_channels(variant)
        t_lat = num_latent_frames_for(self.num_frames)

        latents = make_noise(
            batch_size=1,
            latent_channels=latent_channels,
            height=self.height,
            width=self.width,
            spatial_scale_factor=spatial_scale,
            device=device,
            dtype=latent_dtype,
            seed=self.seed,
            num_latent_frames=t_lat,
        )

        if total_steps <= 0:
            return latents

        # Sanity-check ref-condition shape per variant. A14B expects matched T_lat;
        # TI2V-5B expects a single latent frame regardless of output length.
        if ref_condition is not None:
            if variant == WanVariantType.TI2V_5B:
                if ref_condition.shape[1] != 48 or ref_condition.shape[2] != 1:
                    raise ValueError(
                        f"TI2V-5B reference condition must be shape [1, 48, 1, H_lat, W_lat] "
                        f"(got channels={ref_condition.shape[1]}, frames={ref_condition.shape[2]}). "
                        "Re-run the Reference Image - Wan 2.2 node with a TI2V-5B VAE."
                    )
            elif ref_condition.shape[2] != t_lat:
                raise ValueError(
                    f"Reference-image condition has {ref_condition.shape[2]} latent frames but the "
                    f"denoise loop expected {t_lat}. Ensure the ref-image encoder was called with "
                    f"the same num_frames ({self.num_frames})."
                )

        # Build the TI2V-5B first-frame mask once: 0 at frame 0 (locked to the
        # condition), 1 elsewhere (free to denoise). Broadcasts across channel dim.
        first_frame_mask: torch.Tensor | None = None
        if ref_condition is not None and variant == WanVariantType.TI2V_5B:
            _, _, _, h_lat, w_lat = latents.shape
            first_frame_mask = torch.ones(1, 1, t_lat, h_lat, w_lat, device=device, dtype=inference_dtype)
            first_frame_mask[:, :, 0] = 0

        step_callback = self._build_step_callback(context)

        high_model = self.transformer.transformer
        low_model = self.transformer.transformer_low_noise
        low_config = context.models.get_config(low_model) if low_model is not None else None
        num_train_timesteps = int(scheduler.config.num_train_timesteps)
        boundary_timestep = self.transformer.boundary_ratio * num_train_timesteps if low_model is not None else None

        # No fallback between the LoRA lists — see the matching comment in
        # wan_denoise: an empty low list means "no LoRAs on the low expert".
        high_loras = self.transformer.loras
        low_loras = self.transformer.loras_low_noise
        high_config = context.models.get_config(high_model)
        high_is_quantized = high_config.format == ModelFormat.GGUFQuantized
        low_is_quantized = low_config.format == ModelFormat.GGUFQuantized if low_config is not None else False

        def high_lora_factory() -> Iterable[Tuple[ModelPatchRaw, float]]:
            return proxy._lora_iterator(context, high_loras)

        def low_lora_factory() -> Iterable[Tuple[ModelPatchRaw, float]]:
            return proxy._lora_iterator(context, low_loras)

        with ExitStack() as exit_stack:
            swapper = _ExpertSwapper(
                context=context,
                high_model=high_model,
                low_model=low_model,
                inference_dtype=inference_dtype,
                high_lora_factory=high_lora_factory if high_loras else None,
                low_lora_factory=low_lora_factory if low_loras else None,
                high_is_quantized=high_is_quantized,
                low_is_quantized=low_is_quantized,
            )
            exit_stack.callback(swapper.close)

            for step_idx, t in enumerate(
                tqdm(timesteps, desc=f"Denoising Wan 2.2 video ({self.num_frames} frames)", total=total_steps)
            ):
                if low_model is not None and float(t) < float(boundary_timestep):
                    active_label = _ExpertSwapper.LOW
                    low_cfg = self.guidance_scale_low_noise
                    active_cfg = low_cfg if (low_cfg is not None and low_cfg >= 1.0) else self.guidance_scale
                else:
                    active_label = _ExpertSwapper.HIGH
                    active_cfg = self.guidance_scale

                transformer = swapper.get(active_label)

                latent_model_input = latents.to(dtype=inference_dtype)

                # Per-variant conditioning. Two distinct mechanisms:
                if first_frame_mask is not None:
                    # TI2V-5B I2V (expand_timesteps): blend the condition into frame 0
                    # and the noisy latents elsewhere. Per-token timestep tensor
                    # gates the model so it sees timestep=0 at frame-0 positions
                    # (nothing to denoise) and the normal `t` everywhere else.
                    assert ref_condition is not None
                    latent_model_input = (1 - first_frame_mask) * ref_condition + first_frame_mask * latent_model_input
                    # Strided slice matches the transformer's spatial patch_size=2;
                    # flatten gives per-token timesteps. Shape: [1, T_lat * H_lat//2 * W_lat//2].
                    temp_ts = (first_frame_mask[0, 0, :, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1).to(dtype=inference_dtype)
                elif ref_condition is not None:
                    # A14B I2V: concat 20-ch condition along channel dim → 36-ch input.
                    latent_model_input = torch.cat([latent_model_input, ref_condition], dim=1)
                    timestep = t.expand(latents.shape[0])
                else:
                    # T2V (any variant): scalar timestep per batch.
                    timestep = t.expand(latents.shape[0])

                noise_pred_cond = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=pos_cond.prompt_embeds.unsqueeze(0),
                    attention_kwargs=None,
                    return_dict=False,
                )[0]

                if neg_cond is not None and active_cfg != 1.0:
                    noise_pred_uncond = transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=neg_cond.prompt_embeds.unsqueeze(0),
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + active_cfg * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                step_callback(
                    PipelineIntermediateState(
                        step=step_idx + 1,
                        order=1,
                        total_steps=total_steps,
                        timestep=int(t.item()),
                        # Preview shows the middle frame for video.
                        latents=latents[:, :, t_lat // 2],
                    )
                )

        # TI2V-5B: frame 0's latents drifted through the scheduler step at each
        # iteration; restore them to the clean condition before VAE-decoding so
        # the first output frame matches the reference image. Mirrors
        # ``WanImageToVideoPipeline`` (pipeline_wan_i2v.py:813-814).
        if first_frame_mask is not None:
            assert ref_condition is not None
            latents = (1 - first_frame_mask) * ref_condition.to(dtype=latents.dtype) + first_frame_mask * latents

        return latents

    def _load_conditioning(
        self,
        context: InvocationContext,
        cond_field: WanConditioningField,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> WanConditioningInfo:
        cond_data = context.conditioning.load(cond_field.conditioning_name)
        assert len(cond_data.conditionings) == 1
        cond_info = cond_data.conditionings[0]
        assert isinstance(cond_info, WanConditioningInfo)
        return cond_info.to(device=device, dtype=dtype)

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.Wan)

        return step_callback
