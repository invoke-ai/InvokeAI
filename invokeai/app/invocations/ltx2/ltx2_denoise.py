"""LTX-2 denoise: joint video and audio sampling from one transformer."""

from contextlib import nullcontext
from typing import Literal

import torch
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    LTX2AudioConditioningField,
    LTX2ConditioningField,
    LTX2FullVideoConditioningField,
    LTX2VideoConditioningField,
    OutputField,
)
from invokeai.app.invocations.model import LTX2TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.misc import SEED_MAX
from invokeai.backend.ltx2.constants import (
    LTX2_AUDIO_CFG_SCALE,
    LTX2_CANVAS_MULTIPLE,
    LTX2_CFG_SCALE,
    LTX2_DEFAULT_FPS,
    LTX2_DEFAULT_NUM_FRAMES,
    LTX2_DEV_STEPS,
    LTX2_DISTILLED_STEPS,
    LTX2_GUIDANCE_RESCALE,
    LTX2_MODALITY_SCALE,
    LTX2_STAGE_2_NOISE_SCALE,
    LTX2_STG_SCALE,
)
from invokeai.backend.ltx2.denoise import build_denoise_state, build_refine_state, denoise, preview_latent_frame
from invokeai.backend.ltx2.guidance import LTX2Guidance
from invokeai.backend.ltx2.packing import (
    latent_frame_count,
    require_patch_geometry,
    unpack_video_latents,
    validate_num_frames,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, LTX2VariantType
from invokeai.backend.patches.layer_patcher import LayerPatcher, PatchSpec
from invokeai.backend.patches.lora_conversions.ltx2_lora_constants import LTX2_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import (
    peak_dequant_transient_bytes,
    requires_sidecar_patching,
)
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import LTX2ConditioningInfo
from invokeai.backend.util.devices import TorchDevice

LTX2Schedule = Literal["auto", "dev", "distilled"]

LTX2_SCHEDULE_LABELS: dict[str, str] = {
    "auto": "Auto (from the model's variant)",
    "dev": "Dev (shifted, guided)",
    "distilled": "Distilled (8 fixed steps, no guidance)",
}


@invocation_output("ltx2_denoise_output")
class LTX2DenoiseOutput(BaseInvocationOutput):
    """Joint video and audio latents from one LTX-2 denoise run."""

    video_latents: LatentsField = OutputField(description="Video latents [1, 128, T_lat, H/32, W/32].")
    audio_latents: LatentsField = OutputField(description="Packed audio latents [1, T_audio, 128].")
    width: int = OutputField(description="Pixel width of the video latents.")
    height: int = OutputField(description="Pixel height of the video latents.")
    num_frames: int = OutputField(description="Pixel-frame count of the video latents.")


@invocation(
    "ltx2_denoise",
    title="Denoise - LTX-2",
    tags=["ltx", "ltx2", "video", "audio", "denoise"],
    category="latents",
    version="1.5.0",
    classification=Classification.Prototype,
)
class LTX2DenoiseInvocation(BaseInvocation):
    """Generates LTX-2 video and audio latents together.

    One transformer forward produces both modalities at every step, so the soundtrack is generated
    with the picture rather than dubbed onto it. Wire an image conditioning to start the clip from
    a frame.

    The guidance-distilled checkpoint runs a fixed eight-step schedule with no guidance at all: its
    step count and guidance scales below are ignored, and the negative prompt never reaches it.
    """

    transformer: LTX2TransformerField = InputField(
        description=FieldDescriptions.ltx2_model, input=Input.Connection, title="Transformer"
    )
    positive_conditioning: LTX2ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: LTX2ConditioningField | None = InputField(
        default=None,
        description="Conditioning for the unconditional pass. Required whenever a guidance scale is above 1.",
        input=Input.Connection,
    )
    video_conditioning: LTX2VideoConditioningField | None = InputField(
        default=None,
        description=FieldDescriptions.ltx2_video_conditioning,
        input=Input.Connection,
        title="Image Conditioning",
    )
    keyframe_conditioning: LTX2VideoConditioningField | None = InputField(
        default=None,
        description="A frame held somewhere other than the start -- a last frame, or any interior "
        "keyframe. Its own `frame_index` says where; -1 is the last frame.",
        input=Input.Connection,
        title="Keyframe Conditioning",
    )
    audio_conditioning: LTX2AudioConditioningField | None = InputField(
        default=None,
        description=FieldDescriptions.ltx2_audio_conditioning,
        input=Input.Connection,
        title="Audio Conditioning",
    )
    audio_prefix_conditioning: LTX2AudioConditioningField | None = InputField(
        default=None,
        description="The opening of the soundtrack, held while the rest is generated. A continuation "
        "uses it so the join blends the source's real sound against a reproduction of itself rather "
        "than against newly invented sound.",
        input=Input.Connection,
        title="Audio Prefix Conditioning",
    )
    full_video_conditioning: LTX2FullVideoConditioningField | None = InputField(
        default=None,
        description=FieldDescriptions.ltx2_full_video_conditioning,
        input=Input.Connection,
        title="Video Conditioning",
    )
    width: int = InputField(default=1248, gt=0, multiple_of=LTX2_CANVAS_MULTIPLE, description="Canvas width.")
    height: int = InputField(default=704, gt=0, multiple_of=LTX2_CANVAS_MULTIPLE, description="Canvas height.")
    num_frames: int = InputField(
        default=LTX2_DEFAULT_NUM_FRAMES,
        gt=0,
        description="Frames to generate. The causal VAE encodes the first frame alone and then groups of "
        "eight, so this must be 8n + 1.",
    )
    fps: float = InputField(
        default=LTX2_DEFAULT_FPS,
        ge=1,
        le=120,
        description="Frames per second. Sets the clip's duration, which is also what the audio stream is "
        "generated to fill, and the transformer's positional clock.",
    )
    steps: int = InputField(default=LTX2_DEV_STEPS, gt=0, description="Number of denoising steps.")
    cfg_scale: float = InputField(
        default=LTX2_CFG_SCALE, ge=1.0, description="Classifier-free guidance scale for the video stream."
    )
    audio_cfg_scale: float = InputField(
        default=LTX2_AUDIO_CFG_SCALE,
        ge=1.0,
        description="Classifier-free guidance scale for the audio stream. The release guides audio much "
        "harder than video.",
        title="Audio CFG Scale",
    )
    stg_scale: float = InputField(
        default=LTX2_STG_SCALE,
        ge=0.0,
        description="Spatio-temporal guidance scale. Steers away from a pass whose self-attention is skipped "
        "in one block, which sharpens motion. 0 turns it off and saves a forward per step.",
        title="STG Scale",
    )
    modality_scale: float = InputField(
        default=LTX2_MODALITY_SCALE,
        ge=1.0,
        description="Modality-isolation guidance scale. Steers away from a pass with the audio/video "
        "cross-attention disabled, which tightens the two streams' agreement. 1 turns it off.",
    )
    guidance_rescale: float = InputField(
        default=LTX2_GUIDANCE_RESCALE,
        ge=0.0,
        le=1.0,
        description="How far to pull the guided prediction's contrast back toward the unguided one.",
    )
    schedule: LTX2Schedule = InputField(
        default="auto",
        description="Which noise schedule to sample. 'Auto' follows the loaded transformer's variant.",
        ui_choice_labels=LTX2_SCHEDULE_LABELS,
    )
    seed: int = InputField(default=0, ge=0, le=SEED_MAX, description="Randomness seed for reproducibility.")
    latents: LatentsField | None = InputField(
        default=None,
        description="Upscaled video latents to refine, from Upscale Latents - LTX-2. Present only on the "
        "second pass of a two-stage run; leave unwired to generate from noise.",
        input=Input.Connection,
        title="Video Latents",
    )
    audio_latents: LatentsField | None = InputField(
        default=None,
        description="The base pass's audio latents, carried into the refine pass unchanged. Required "
        "whenever video latents are wired.",
        input=Input.Connection,
        title="Audio Latents",
    )
    noise_scale: float = InputField(
        default=LTX2_STAGE_2_NOISE_SCALE,
        gt=0.0,
        lt=1.0,
        description="Where the refine pass re-enters the schedule: the noise level the upscaled latents are "
        "taken back to, and the highest level it samples. Lower keeps more of the base pass. Ignored "
        "without wired video latents.",
    )

    def _materialize_lora_patches(self, context: InvocationContext) -> list[PatchSpec]:
        """Load every LoRA on the transformer field into (patch, weight, cache-pin) specs."""
        patch_specs: list[PatchSpec] = []
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}."
                )
            patch_specs.append((lora_info.model, lora.weight, lora_info.model_in_ram()))
        return patch_specs

    @staticmethod
    def _estimate_working_memory(video_rows: int, audio_rows: int, patch_bytes: int = 0) -> int:
        """Estimate peak transformer activation bytes so the model cache reserves enough headroom.

        The 22B transformer is partially loaded on any card this runs on, and without a hint the
        cache reserves only its small default, packs VRAM with weights, and the first forward dies.

        Attention runs through SDPA without materializing scores, so activations scale linearly with
        the packed row count: per video row the concurrently live bf16 terms inside one block are
        the 4096-wide QKV and attention output and the gelu feed-forward's intermediates. Audio rows
        are the same shape at half the width, and there are two orders of magnitude fewer of them.
        Guidance passes run one after another rather than as one batch, so they do not multiply the
        peak; the per-pass x0 predictions that are all live at the combine are one float32 row each
        and are inside the constants below.

        Measured on a W7900 (gfx1100, released int8-convrot transformer), as peak *reserved* minus
        the resident model: 0.46 GiB at 320 video rows (512x320x9), 2.81 at 13728 (1248x704x121),
        5.73 at 28672 (1792x1024x121) and 13.34 at 66048 (2752x1536x121). Guidance does not move the
        peak -- the passes run one after another -- and the last two were taken on the refine pass of
        a two-stage run, which is where the largest row counts occur.

        The slope is fitted to the *top* of that range, where the reservation has to hold: the four
        points imply about 0.21 MiB/row between the widest two, and 1/4.5 MiB rounds that up to a
        uniform margin (+15% at 66048 rows, wider below). A slope fitted to the narrow end instead
        looks generous at small canvases and converges on the measurement exactly where running out
        would be most expensive. The base additionally covers block weights arriving on device under
        partial loading, which a fully resident measurement does not see.
        """
        MiB = 1024**2
        estimated = video_rows * int(MiB // 4.5) + audio_rows * (MiB // 10) + 1024**3
        if patch_bytes > 0:
            # `patch_bytes` is what the patches actually weigh, and is passed as zero unless they
            # will be sidecar-patched: the direct path returns each patch to the CPU as it goes, so
            # nothing of it stays resident. The extra gibibyte covers the transient a sidecar layer
            # holds while it computes the low-rank residual -- a second copy of that layer's output,
            # alongside the dequantized weight the term above already accounts for.
            estimated += patch_bytes + 1024**3
        return estimated

    def _resolve_distilled(self, context: InvocationContext) -> bool:
        if self.schedule != "auto":
            return self.schedule == "distilled"
        if self.transformer.loras and self.transformer.variant != LTX2VariantType.Distilled.value:
            # `auto` follows the transformer's *variant*, which names the checkpoint -- and a LoRA
            # does not change it. A step-distillation LoRA on a Dev checkpoint therefore resolves to
            # the guided ~30-step schedule and samples the LoRA's 8 steps on it, which produces a
            # broken clip and looks like a broken model rather than a wiring mistake. The panel sets
            # `schedule` explicitly for this reason; a hand-built graph has to be told.
            #
            # Info rather than a warning, and silent on a distilled checkpoint: this cannot tell a
            # step-distillation LoRA from an ordinary style one, so on Dev it is a note for the
            # minority case rather than a claim about this run. On a distilled checkpoint `auto` has
            # already resolved correctly and saying anything would be simply wrong.
            context.logger.info(
                "LTX-2 schedule is Auto with %d LoRA(s) applied. Auto follows the checkpoint, which a LoRA "
                "does not change -- if one of these is a step-distillation LoRA, set Schedule to 'Distilled' "
                "or the run will sample its step count on the guided schedule.",
                len(self.transformer.loras),
            )
        if self.transformer.variant is None:
            raise ValueError(
                "The schedule is set to Auto but the transformer carries no variant, so there is nothing to "
                "follow. Choose 'Dev' or 'Distilled' explicitly."
            )
        return self.transformer.variant == LTX2VariantType.Distilled.value

    def _resolve_guidance(self, context: InvocationContext, distilled: bool) -> LTX2Guidance:
        if distilled:
            # Not a preference the user can override: the distilled checkpoint has guidance baked
            # into its weights, and steering it produces a saturated, broken clip.
            requested = (self.cfg_scale, self.audio_cfg_scale, self.stg_scale, self.modality_scale)
            if requested != (1.0, 1.0, 0.0, 1.0):
                context.logger.info(
                    "The distilled LTX-2 checkpoint is guidance-distilled; ignoring the guidance scales on "
                    "this node and running one forward per step."
                )
            return LTX2Guidance(cfg_scale=1.0, audio_cfg_scale=1.0, stg_scale=0.0, modality_scale=1.0, rescale=0.0)
        return LTX2Guidance(
            cfg_scale=self.cfg_scale,
            audio_cfg_scale=self.audio_cfg_scale,
            stg_scale=self.stg_scale,
            modality_scale=self.modality_scale,
            rescale=self.guidance_rescale,
        )

    def _load_conditioning(self, context: InvocationContext, field: LTX2ConditioningField) -> LTX2ConditioningInfo:
        data = context.conditioning.load(field.conditioning_name)
        assert len(data.conditionings) == 1
        info = data.conditionings[0]
        assert isinstance(info, LTX2ConditioningInfo)
        return info

    def _require_matching(
        self,
        node_title: str,
        integers: dict[str, tuple[int, int]],
        floats: dict[str, tuple[float, float]],
    ) -> None:
        """Refuse a conditioning encode prepared for a different run than this node will make."""
        mismatched = [
            f"{name} {theirs} vs {mine}"
            for name, (theirs, mine) in ((*integers.items(), *floats.items()))
            if (theirs != mine if name not in floats else abs(float(theirs) - float(mine)) > 1e-6)
        ]
        if mismatched:
            raise ValueError(
                f"The conditioning from {node_title} was prepared for a different run than this denoise: "
                f"{', '.join(mismatched)}. Wire this node's own values into that one, or take its outputs."
            )

    def _resolve_keyframe(self, context: InvocationContext) -> tuple[torch.Tensor | None, int | None]:
        """The held frame and the latent index it belongs at, with negatives resolved.

        The index is resolved here rather than at the conditioning node because that node does not
        know how long the clip is -- which is what lets one encode stay valid when the frame count
        changes, instead of silently landing at the wrong instant.
        """
        if self.keyframe_conditioning is None:
            return None, None

        if (self.keyframe_conditioning.width, self.keyframe_conditioning.height) != (self.width, self.height):
            raise ValueError(
                f"The keyframe conditioning was prepared for a {self.keyframe_conditioning.width}x"
                f"{self.keyframe_conditioning.height} canvas but this denoise runs at {self.width}x{self.height}. "
                "Re-run Image Conditioning - LTX-2 with matching width and height."
            )

        latent_frames = latent_frame_count(self.num_frames)
        index = self.keyframe_conditioning.frame_index
        resolved = latent_frames + index if index < 0 else index

        if resolved == 0:
            raise ValueError(
                "A keyframe at the first frame is what Image Conditioning's own output is for: wire it to "
                "Image Conditioning rather than Keyframe Conditioning, or give it a non-zero frame index."
            )
        if not 0 < resolved < latent_frames:
            raise ValueError(
                f"Keyframe index {index} resolves to latent frame {resolved}, which is outside a "
                f"{self.num_frames}-frame clip ({latent_frames} latent frames)."
            )

        return context.tensors.load(self.keyframe_conditioning.latents_name), resolved

    def _load_image_latents(self, context: InvocationContext) -> torch.Tensor | None:
        if self.video_conditioning is None:
            return None
        if self.video_conditioning.frame_index != 0:
            # The field's own description says a negative index makes the frame last, and this slot
            # cannot honour that: it overwrites the opening grid tokens. Silently holding it at
            # frame 0 would do the opposite of what the graph asked for.
            raise ValueError(
                f"Image Conditioning holds a frame at the start, but this one was encoded for frame "
                f"{self.video_conditioning.frame_index}. Wire it to Keyframe Conditioning instead, or "
                f"set its frame index to 0."
            )
        if (self.video_conditioning.width, self.video_conditioning.height) != (self.width, self.height):
            raise ValueError(
                f"The image conditioning was prepared for a {self.video_conditioning.width}x"
                f"{self.video_conditioning.height} canvas but this denoise runs at {self.width}x{self.height}. "
                "Re-run Image Conditioning - LTX-2 with matching width and height."
            )
        return context.tensors.load(self.video_conditioning.latents_name)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LTX2DenoiseOutput:
        # The canvas is checked by the fields themselves (`multiple_of`), so a bad one never
        # enqueues; the frame grid cannot be expressed that way, and checking it here is what stops
        # a mistyped count from loading a 12B encoder and a 22B transformer before it is refused.
        # `build_denoise_state` checks both again for callers that do not come through this node.
        validate_num_frames(self.num_frames)

        # Refused here, beside the frame grid, rather than where the state is built: a half-wired
        # refine pass is a graph mistake, and finding it after the 22B transformer has loaded is the
        # difference between a message and a wasted minute. (The prompt encode is a separate node
        # and has already run by now; only the transformer load is saved.)
        if self.latents is not None and self.audio_latents is None:
            raise ValueError(
                "The refine pass needs the base pass's audio latents as well as its video: wire both "
                "outputs of the first LTX-2 denoise node through."
            )
        if self.latents is None and self.audio_latents is not None:
            raise ValueError(
                "Audio latents were wired without video latents. The refine pass takes both or neither; "
                "a base pass generates its own audio."
            )
        frozen = self.audio_conditioning is not None or self.full_video_conditioning is not None

        # A refine pass re-noises every token, so a modality held clean in the base pass would be
        # regenerated here unless it were re-applied -- and for video that needs a fresh encode at
        # the refine canvas, exactly as the first frame does. Rather than silently drop the
        # conditioning, the combination is refused until that path exists.
        if frozen and self.latents is not None:
            raise ValueError(
                "Audio- and video-conditioned generation does not run a refine pass yet: the held "
                "modality would have to be re-encoded at the second canvas. Use a single-stage "
                "target resolution."
            )

        # The geometry each conditioning field carries is checked here rather than left to the
        # latent-shape check in `build_denoise_state`: that one cannot see `fps` at all -- no tensor
        # shape encodes it -- and a mismatch there would surface as a raw tuple after the text
        # encoder has run. A graph that sets these separately from the node is a wiring mistake.
        if self.audio_conditioning is not None:
            self._require_matching(
                "Audio Conditioning - LTX-2",
                {"num_frames": (self.audio_conditioning.num_frames, self.num_frames)},
                {"fps": (self.audio_conditioning.fps, self.fps)},
            )
        if self.audio_prefix_conditioning is not None:
            # Only the rate: this field's `num_frames` is the HELD span, not the clip's, and it is
            # the rate that cannot be recovered from any tensor shape. A prefix sized at one rate
            # and held at another covers a different stretch of time than the picture it belongs
            # to, so the join crossfades real source audio into the new material -- silently, since
            # the row count stays well inside what the clip has room for.
            self._require_matching(
                "Extend Conditioning - LTX-2",
                {},
                {"fps": (self.audio_prefix_conditioning.fps, self.fps)},
            )
        if self.full_video_conditioning is not None:
            self._require_matching(
                "Video Conditioning - LTX-2",
                {
                    "num_frames": (self.full_video_conditioning.num_frames, self.num_frames),
                    "width": (self.full_video_conditioning.width, self.width),
                    "height": (self.full_video_conditioning.height, self.height),
                },
                {"fps": (self.full_video_conditioning.fps, self.fps)},
            )

        distilled = self._resolve_distilled(context)
        guidance = self._resolve_guidance(context, distilled)
        if distilled and self.steps != LTX2_DISTILLED_STEPS:
            context.logger.info(
                f"The distilled LTX-2 schedule is a fixed {LTX2_DISTILLED_STEPS} steps; ignoring the "
                f"{self.steps} requested."
            )

        positive = self._load_conditioning(context, self.positive_conditioning)
        negative = None
        if guidance.needs_negative_conditioning:
            if self.negative_conditioning is None:
                raise ValueError(
                    "Classifier-free guidance is on (a CFG scale above 1) but no negative conditioning is "
                    "wired. Connect Prompt - LTX-2's negative output, or set both CFG scales to 1."
                )
            negative = self._load_conditioning(context, self.negative_conditioning)

        keyframe_latents, keyframe_index = self._resolve_keyframe(context)
        audio_prefix_latents = (
            context.tensors.load(self.audio_prefix_conditioning.latents_name)
            if self.audio_prefix_conditioning is not None
            else None
        )

        if self.latents is not None:
            state = build_refine_state(
                video_latents=context.tensors.load(self.latents.latents_name),
                audio_latents=context.tensors.load(self.audio_latents.latents_name),
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                fps=self.fps,
                seed=self.seed,
                distilled=distilled,
                num_steps=self.steps,
                noise_scale=self.noise_scale,
                # Encoded at *this* pass's canvas: the refine pass re-noises frame 0 along with
                # everything else, so an anchor has to be re-applied here or image-to-video would
                # mean something different on a two-stage preset.
                image_latents=self._load_image_latents(context),
                conditioning_strength=self.video_conditioning.strength if self.video_conditioning else 1.0,
                keyframe_latents=keyframe_latents,
                keyframe_latent_index=keyframe_index,
                keyframe_strength=self.keyframe_conditioning.strength if self.keyframe_conditioning else 1.0,
                audio_prefix_latents=audio_prefix_latents,
            )
        else:
            state = build_denoise_state(
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                fps=self.fps,
                seed=self.seed,
                distilled=distilled,
                num_steps=self.steps,
                image_latents=self._load_image_latents(context),
                conditioning_strength=self.video_conditioning.strength if self.video_conditioning else 1.0,
                keyframe_latents=keyframe_latents,
                keyframe_latent_index=keyframe_index,
                keyframe_strength=self.keyframe_conditioning.strength if self.keyframe_conditioning else 1.0,
                audio_prefix_latents=audio_prefix_latents,
                frozen_audio_latents=(
                    context.tensors.load(self.audio_conditioning.latents_name)
                    if self.audio_conditioning is not None
                    else None
                ),
                frozen_video_latents=(
                    context.tensors.load(self.full_video_conditioning.latents_name)
                    if self.full_video_conditioning is not None
                    else None
                ),
            )

        device = TorchDevice.choose_torch_device()
        inference_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)
        transformer_info = context.models.load(self.transformer.transformer)
        transformer_config = context.models.get_config(self.transformer.transformer)

        # Materialized before the model lock: loading a LoRA takes the cache's own lock, and doing
        # that while holding this one is the deadlock `apply_smart_model_patches` documents. It also
        # has to happen before the reservation below, which is sized from these patches.
        lora_patch_specs = self._materialize_lora_patches(context)
        # Resolved from the unlocked model, like the dequant transient below: only a build whose
        # weights are buffers rather than parameters takes the sidecar path, and only that path
        # leaves patch tensors on the device.
        # Walks the whole 22B module tree, so it is only asked when there is something to patch.
        force_sidecar = bool(lora_patch_specs) and requires_sidecar_patching(
            transformer_info.model, transformer_config.format
        )

        estimated_working_memory = self._estimate_working_memory(
            # The sequence as built, not the grid: a held keyframe rides on the end of it and is
            # attended over in every forward -- 858 rows on top of 13728 at 1248x704 x121 -- and the
            # fit is tightest exactly where running out is most expensive.
            state.video_latents.shape[1],
            state.audio_latents_count,
            # The patches' own byte count, not a per-LoRA guess. This reservation is subtracted from
            # VRAM *before* any weight streams in, so over-reserving is not free -- it directly buys
            # fewer resident transformer weights, and a blind constant large enough for the rank-450
            # distilled accelerator (8.3 GiB) would cost a rank-16 style LoRA the same.
            sum(spec[0].calc_size() for spec in lora_patch_specs) if force_sidecar else 0,
        )
        # An int8-convrot build materializes each linear's dequantized weight inside the forward,
        # which the model's resident size does not account for. Read from the unlocked model, before
        # the VRAM lock the reservation applies to; zero on a bf16 build.
        estimated_working_memory += peak_dequant_transient_bytes(transformer_info.model, inference_dtype)

        with (
            transformer_info.model_on_device(working_mem_bytes=estimated_working_memory) as (
                cached_weights,
                transformer,
            ),
            # Skipped entirely when there is nothing to patch: entering it takes the loader lock
            # and copies the cached-weight mapping, neither of which an unaccelerated run should
            # pay for. Same guard MiniMax H3 uses.
            (
                LayerPatcher.apply_smart_model_patches(
                    model=transformer,
                    patches=lora_patch_specs,
                    prefix=LTX2_LORA_TRANSFORMER_PREFIX,
                    dtype=inference_dtype,
                    cached_weights=cached_weights,
                    # An int8-convrot / nvfp4 build keeps its weights as buffers rather than
                    # parameters, so the patcher's own fp8 and CPU fallbacks would answer "not
                    # quantized" and merge the delta into a weight that is never read. This is the
                    # only reliable signal.
                    force_sidecar_patching=force_sidecar,
                )
                if lora_patch_specs
                else nullcontext()
            ),
        ):
            require_patch_geometry(transformer.config)
            # Named per stage: a two-stage run drives this bar to 100%, then starts a second one
            # from 0 with the upscale in between, which reads as a restart unless it says otherwise.
            context.util.signal_progress(
                "Refining LTX-2 audio-video" if self.latents is not None else "Denoising LTX-2 audio-video"
            )
            stage = "Refining" if self.latents is not None else "Denoising"
            progress = tqdm(total=state.num_steps, desc=f"{stage} LTX-2 ({self.num_frames} frames)")

            def step_callback(step: int, total_steps: int, video_x0: torch.Tensor) -> None:
                progress.update(1)
                context.util.sd_step_callback(
                    PipelineIntermediateState(
                        step=step,
                        order=1,
                        total_steps=total_steps,
                        timestep=0,
                        latents=preview_latent_frame(video_x0, state),
                    ),
                    BaseModelType.LTX2,
                )

            try:
                video_latents, audio_latents = denoise(
                    transformer=transformer,
                    state=state,
                    positive=positive,
                    negative=negative,
                    guidance=guidance,
                    fps=self.fps,
                    dtype=inference_dtype,
                    device=device,
                    step_callback=step_callback,
                    is_canceled=context.util.is_canceled,
                )
            finally:
                progress.close()

        video_5d = unpack_video_latents(video_latents, state.latent_frames, state.latent_height, state.latent_width)
        return LTX2DenoiseOutput(
            video_latents=LatentsField(
                latents_name=context.tensors.save(tensor=video_5d.detach().to(device="cpu", dtype=torch.float32)),
                seed=self.seed,
            ),
            audio_latents=LatentsField(
                latents_name=context.tensors.save(tensor=audio_latents.detach().to(device="cpu", dtype=torch.float32)),
                seed=self.seed,
            ),
            width=self.width,
            height=self.height,
            num_frames=self.num_frames,
        )
