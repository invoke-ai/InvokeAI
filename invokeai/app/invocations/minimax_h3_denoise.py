"""MiniMax H3 denoise invocation (T2VA / FL2VA).

One packed sequence carries the text conditioning, the optional keyframe conditioning rows,
the audio latents and the video latents through a single transformer forward per step; video
and audio step down two different flow schedules (shift 12.0 / 3.0). The checkpoint is
guidance-distilled: no negative prompt, no CFG, one forward per step.
"""

from contextlib import ExitStack

import torch
from PIL import Image
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
    MiniMaxH3ConditioningField,
    MiniMaxH3FrameConditioningField,
    OutputField,
)
from invokeai.app.invocations.model import MiniMaxH3TransformerField
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.minimax_h3.denoise import denoise
from invokeai.backend.minimax_h3.packing import (
    MINIMAX_H3_CANVAS_MULTIPLE,
    MiniMaxH3PackedSequence,
    audio_latent_num_frames,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from invokeai.backend.minimax_h3.sampling import (
    MINIMAX_H3_PATCH_SIZE,
    MINIMAX_H3_SPATIAL_COMPRESSION,
    MINIMAX_H3_VAE_LATENT_CHANNELS,
    build_denoise_state,
    validate_canvas,
    validate_num_frames,
)
from invokeai.backend.minimax_h3.taehv_decoder import TAEH3_PREVIEW_MODEL_URL, TAEH3Decoder
from invokeai.backend.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from invokeai.backend.model_manager.load.load_base import LoadedModelWithoutConfig
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import MiniMaxH3ConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation_output("minimax_h3_denoise_output")
class MiniMaxH3DenoiseOutput(BaseInvocationOutput):
    """Joint video + audio latents from one MiniMax H3 denoise run."""

    video_latents: LatentsField = OutputField(description="5D video latents [1, 24, T_lat, H/16, W/16].")
    audio_latents: LatentsField = OutputField(
        description="Audio latents [2, 32, T_audio] (one item per stereo channel)."
    )
    width: int = OutputField(description="Pixel width of the video latents.")
    height: int = OutputField(description="Pixel height of the video latents.")
    num_frames: int = OutputField(description="Pixel-frame count of the video latents.")


@invocation(
    "minimax_h3_denoise",
    title="Denoise - MiniMax H3",
    tags=["latents", "video", "audio", "minimax"],
    category="latents",
    version="1.1.0",
    classification=Classification.Prototype,
)
class MiniMaxH3DenoiseInvocation(BaseInvocation):
    """Run the MiniMax H3 joint audio-video denoising loop."""

    transformer: MiniMaxH3TransformerField = InputField(
        description="MiniMax H3 FL2VA transformer.", input=Input.Connection, title="Transformer"
    )
    positive_conditioning: MiniMaxH3ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    frame_conditioning: MiniMaxH3FrameConditioningField | None = InputField(
        default=None,
        description=FieldDescriptions.minimax_h3_frame_conditioning,
        input=Input.Connection,
        title="Frame Conditioning",
    )
    width: int = InputField(
        default=1344,
        gt=0,
        multiple_of=MINIMAX_H3_CANVAS_MULTIPLE,
        description="Width of the generated video. H3's native canvas has a 768px short edge (max 768x1344).",
    )
    height: int = InputField(
        default=768,
        gt=0,
        multiple_of=MINIMAX_H3_CANVAS_MULTIPLE,
        description="Height of the generated video.",
    )
    num_frames: int = InputField(
        default=124,
        ge=5,
        description="Number of output frames at the fixed 24 fps. Must be of the form 17n+5 "
        "(5, 22, ..., 124, ...); durations must stay within 5-15 s, except exactly 5 frames "
        "for a still image.",
        title="Number of Frames",
    )
    steps: int = InputField(
        default=50,
        ge=2,
        description="Number of denoising steps (sigma grid points, terminal included: N steps = N-1 model evaluations).",
    )
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")

    @staticmethod
    def _estimate_working_memory(layout: MiniMaxH3PackedSequence) -> int:
        """Estimate peak transformer activation memory (bytes) so the model cache reserves enough headroom.

        The 61.7 GiB bf16 transformer is partially loaded on most cards; without this hint the cache
        reserves only the small default working memory, packs the device with weights, and the first
        forward OOMs (a 124-frame 768x1344 t2v is a ~38k-row packed sequence).

        Attention runs through SDPA without materializing scores, so activations scale ~linearly with
        the packed row count. The dominant per-row bf16 terms concurrently alive inside one block are
        QKV/attention-out at the 7168-wide attention inner dim and the SwiGLU intermediates at
        ffn_dim 14336 (~0.16 MiB/row measured together with the residual stream and the fp32 output
        heads); 0.25 MiB/row leaves ~1.5x margin. Padding rows additionally materialize a boolean
        (seq x seq) attention mask - and SDPA's convert_boolean_attn_mask then allocates an
        additive copy at query dtype (2 bytes) plus a possible alignment-pad transient of the same
        size, so budget 5 bytes per mask entry, not 1. (No current code path emits padding rows;
        this arms the estimate for ref2va-style layouts.) The fixed base covers streamed block
        weights arriving on device under partial load (per-op transients up to ~0.5 GiB), the
        prompt embeds, rotary tables, per-step preview unpatchify, and allocator slack.
        """
        MB = 1024**2
        GB = 1024**3
        estimated = layout.sequence_length * int(0.25 * MB)
        if bool((layout.token_tags < 0).any()):
            estimated += 5 * layout.sequence_length**2
        estimated += 2 * GB
        return estimated

    @staticmethod
    def _load_preview_decoder(context: InvocationContext) -> LoadedModelWithoutConfig | None:
        """Fetch (a one-time ~23 MB download) and load the taeh3 preview decoder.

        Previews degrade gracefully to the linear latent->RGB projection when the download is
        unavailable (offline installs), so any failure here is a warning, never an error.
        """
        try:
            return context.models.load_remote_model(TAEH3_PREVIEW_MODEL_URL, TAEH3Decoder.load_model)
        except Exception as e:
            context.logger.warning(
                f"MiniMax H3 preview decoder unavailable ({e}); previews fall back to latent projection."
            )
            return None

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> MiniMaxH3DenoiseOutput:
        validate_num_frames(self.num_frames)
        validate_canvas(self.height, self.width)

        device = TorchDevice.choose_torch_device()

        cond_data = context.conditioning.load(self.positive_conditioning.conditioning_name)
        assert len(cond_data.conditionings) == 1
        cond_info = cond_data.conditionings[0]
        assert isinstance(cond_info, MiniMaxH3ConditioningInfo)

        latent_height = self.height // MINIMAX_H3_SPATIAL_COMPRESSION
        latent_width = self.width // MINIMAX_H3_SPATIAL_COMPRESSION
        num_latent_frames = video_latent_num_frames(self.num_frames)
        num_audio_latents = audio_latent_num_frames(self.num_frames)

        keyframe_anchors: tuple[str, ...] = ()
        clean_condition_rows: torch.Tensor | None = None
        if self.frame_conditioning is not None:
            if (self.frame_conditioning.width, self.frame_conditioning.height) != (self.width, self.height):
                raise ValueError(
                    f"Frame conditioning canvas ({self.frame_conditioning.width}x"
                    f"{self.frame_conditioning.height}) must match denoise dimensions "
                    f"({self.width}x{self.height}). Re-run Frame Conditioning - MiniMax H3."
                )
            keyframe_anchors = tuple(self.frame_conditioning.keyframe_anchors)
            clean_condition_rows = context.tensors.load(self.frame_conditioning.condition_rows_name)

        # Keyframes must reach the text conditioning (vision context) and the VAE condition
        # rows together, on the same canvas — the model was trained with them coupled.
        cond_anchors = tuple(cond_info.keyframe_anchors)
        if cond_anchors != keyframe_anchors:
            raise ValueError(
                f"Keyframe mismatch: the prompt was encoded with keyframes {list(cond_anchors) or 'none'} "
                f"but frame conditioning provides {list(keyframe_anchors) or 'none'}. Wire the same "
                "first/last images to both Prompt - MiniMax H3 and Frame Conditioning - MiniMax H3."
            )
        if cond_anchors and (cond_info.width, cond_info.height) != (self.width, self.height):
            raise ValueError(
                f"The prompt's keyframes were prepared at {cond_info.width}x{cond_info.height} but this "
                f"denoise runs at {self.width}x{self.height}. Re-run Prompt - MiniMax H3 with matching "
                "width/height."
            )

        state = build_denoise_state(
            text_token_tags=cond_info.text_token_tags,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            num_inference_steps=self.steps,
            seed=self.seed,
            device=device,
            keyframe_anchors=keyframe_anchors,
            clean_condition_rows=clean_condition_rows,
        )

        num_condition_video_rows = state.layout.num_condition_video_rows

        preview_failed = False

        def make_step_callback(preview_decoder: TAEH3Decoder | None):
            def step_callback(step: int, total_steps: int, pred_x0_video_rows: torch.Tensor) -> None:
                if context.util.is_canceled():
                    raise CanceledException
                # Unpack the generated rows' x-hat-0 estimate to a 5D grid.
                latents_5d = unpatchify_video_tokens(
                    pred_x0_video_rows,
                    num_latent_frames,
                    latent_height,
                    latent_width,
                    MINIMAX_H3_VAE_LATENT_CHANNELS,
                    MINIMAX_H3_PATCH_SIZE,
                )
                nonlocal preview_failed
                if preview_decoder is not None and not preview_failed:
                    try:
                        # A two-latent-frame window ending at the middle frame: the extra frame is
                        # causal warmup so the decoder's temporal memory is warm for the shown frame.
                        mid = num_latent_frames // 2
                        window = latents_5d[:, :, max(0, mid - 1) : mid + 1]
                        frame = preview_decoder.decode_preview_frame(window)
                        image = Image.fromarray(frame.mul(255).round().byte().permute(1, 2, 0).cpu().numpy())
                        context.util.signal_progress(
                            "Denoising MiniMax H3 audio-video", step / total_steps, image, (self.width, self.height)
                        )
                        return
                    except CanceledException:
                        raise
                    except Exception:
                        preview_failed = True
                        context.logger.warning(
                            "MiniMax H3 preview decode failed; falling back to latent-projection previews.",
                            exc_info=True,
                        )
                # Fallback: linear latent->RGB projection of the middle temporal slice.
                context.util.sd_step_callback(
                    PipelineIntermediateState(
                        step=step,
                        order=1,
                        total_steps=total_steps,
                        timestep=0,
                        latents=latents_5d[:, :, num_latent_frames // 2],
                    ),
                    BaseModelType.MiniMaxH3,
                )

            return step_callback

        estimated_working_memory = self._estimate_working_memory(state.layout)
        transformer_info = context.models.load(self.transformer.transformer)
        # The preview decoder's cache record is created after the transformer's RAM load (whose
        # make_room could otherwise drop the unlocked 23 MB record) and locked before the
        # transformer's VRAM lock, so the partial load accounts for it. Previews are strictly
        # best-effort: no failure here may abort the generation.
        preview_decoder_info = self._load_preview_decoder(context)
        with ExitStack() as stack:
            preview_decoder: TAEH3Decoder | None = None
            if preview_decoder_info is not None:
                try:
                    preview_model = stack.enter_context(preview_decoder_info)
                    assert isinstance(preview_model, TAEH3Decoder)
                    preview_decoder = preview_model
                except Exception:
                    context.logger.warning(
                        "Could not lock the MiniMax H3 preview decoder; previews fall back to latent projection.",
                        exc_info=True,
                    )
            step_callback = make_step_callback(preview_decoder)

            _, transformer = stack.enter_context(
                transformer_info.model_on_device(working_mem_bytes=estimated_working_memory)
            )
            assert isinstance(transformer, MiniMaxH3Transformer3DModel)
            context.util.signal_progress("Denoising MiniMax H3 audio-video")
            # steps counts sigma grid points (terminal included) -> steps-1 model evaluations.
            progress = tqdm(total=len(state.timesteps), desc=f"Denoising MiniMax H3 ({self.num_frames} frames)")

            def callback_with_progress(step: int, total_steps: int, pred_x0_video_rows: torch.Tensor) -> None:
                progress.update(1)
                step_callback(step, total_steps, pred_x0_video_rows)

            try:
                video_rows, audio_rows = denoise(
                    transformer=transformer,
                    state=state,
                    prompt_embeds=cond_info.prompt_embeds.to(device),
                    step_callback=callback_with_progress,
                    is_canceled=context.util.is_canceled,
                )
            finally:
                progress.close()

        video_latents = unpatchify_video_tokens(
            video_rows[num_condition_video_rows:],
            num_latent_frames,
            latent_height,
            latent_width,
            MINIMAX_H3_VAE_LATENT_CHANNELS,
            MINIMAX_H3_PATCH_SIZE,
        )
        audio_latents = unpack_audio_tokens(audio_rows[state.layout.num_condition_audio_rows :], num_audio_latents)

        video_latents = video_latents.detach().to(device="cpu", dtype=torch.float32)
        audio_latents = audio_latents.detach().to(device="cpu", dtype=torch.float32)

        video_name = context.tensors.save(tensor=video_latents)
        audio_name = context.tensors.save(tensor=audio_latents)
        return MiniMaxH3DenoiseOutput(
            video_latents=LatentsField(latents_name=video_name, seed=self.seed),
            audio_latents=LatentsField(latents_name=audio_name, seed=self.seed),
            width=self.width,
            height=self.height,
            num_frames=self.num_frames,
        )
