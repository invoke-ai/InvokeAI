"""A clip's soundtrack as LTX-2 conditioning: generate a picture for existing audio."""

import torch

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
    LTX2AudioConditioningField,
    OutputField,
    VideoField,
)
from invokeai.app.invocations.model import LTX2VocoderField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_audio import extract_audio_pcm
from invokeai.backend.ltx2.audio_conditioning import encode_audio_latents
from invokeai.backend.ltx2.constants import (
    LTX2_AUDIO_LATENTS_PER_SECOND,
    LTX2_DEFAULT_FPS,
    LTX2_NUM_FRAMES_MAX,
)
from invokeai.backend.ltx2.packing import audio_latent_count, snap_num_frames_down
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.util.vae_working_memory import estimate_audio_working_memory_ltx2


@invocation_output("ltx2_audio_conditioning_output")
class LTX2AudioConditioningOutput(BaseInvocationOutput):
    """A soundtrack to generate a picture for, and the clip length it implies."""

    audio_conditioning: LTX2AudioConditioningField = OutputField(
        description=FieldDescriptions.ltx2_audio_conditioning, title="Audio Conditioning"
    )
    num_frames: int = OutputField(description="Frames the soundtrack covers, snapped down to 8n + 1.")
    fps: float = OutputField(description="The frame rate that count was derived at.")


@invocation(
    "ltx2_audio_conditioning",
    title="Audio Conditioning - LTX-2",
    tags=["ltx", "ltx2", "video", "audio", "conditioning"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class LTX2AudioConditioningInvocation(BaseInvocation):
    """Encodes a clip's soundtrack so LTX-2 can generate a picture for it.

    LTX-2 conditions picture and sound through one mask, so this is the mirror of first-frame image
    conditioning: the audio rows are held clean at every step and the video is sampled freely.

    The soundtrack's own length decides the generation's. Audio is encoded at 25 latents per second
    and the resulting frame count is snapped *down* to the VAE's 8n + 1 grid, so a clip is never
    asked to cover more picture than it has sound for; the remainder is dropped.
    """

    video: VideoField = InputField(description="The clip whose soundtrack conditions the generation.")
    audio_vae: VAEField = InputField(
        description=FieldDescriptions.ltx2_audio_vae, input=Input.Connection, title="Audio VAE"
    )
    vocoder: LTX2VocoderField = InputField(
        # The vocoder is here for its analysis filters, not to synthesise anything: the log-mel
        # transform the audio VAE encodes from ships as buffers on the vocoder and nowhere else.
        description="The vocoder, whose checkpoint carries the log-mel filters the VAE encodes from.",
        input=Input.Connection,
        title="Vocoder",
    )
    fps: float = InputField(
        default=LTX2_DEFAULT_FPS,
        ge=1,
        le=120,
        description="Frame rate to convert the soundtrack's duration into a frame count.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LTX2AudioConditioningOutput:
        path = context.videos.get_path(self.video.video_name)
        decoded = extract_audio_pcm(path, float_pcm=True)

        if decoded is None:
            raise ValueError(
                f"'{self.video.video_name}' has no audio track, so there is nothing to generate a picture for. "
                f"Pick a clip with sound, or use text-to-video."
            )

        samples, sample_rate = decoded
        audio_vae_info = context.models.load(self.audio_vae.vae)
        if audio_vae_info.config.base is not BaseModelType.LTX2:
            raise ValueError(f"Expected an LTX-2 audio VAE; got a {audio_vae_info.config.base.value} one.")
        vocoder_info = context.models.load(self.vocoder.vocoder)

        # The recording's own length is the only thing bounding this encode -- the STFT, the mel and
        # the VAE forward all run untiled over the whole signal -- so it is capped before any of
        # them, at the longest generation the model will make. Everything past that is soundtrack
        # the run could not cover anyway.
        seconds_cap = snap_num_frames_down(LTX2_NUM_FRAMES_MAX) / self.fps
        if samples.shape[1] > seconds_cap * sample_rate:
            samples = samples[:, : int(seconds_cap * sample_rate)]

        # The same shape as the decode's estimate, which is fitted per audio latent: the encode runs
        # the same three stages backwards over the same signal.
        working_memory = estimate_audio_working_memory_ltx2(
            int(samples.shape[1] / sample_rate * LTX2_AUDIO_LATENTS_PER_SECOND)
        )
        context.util.signal_progress("Encoding the soundtrack for LTX-2 conditioning")
        with (
            audio_vae_info.model_on_device(working_mem_bytes=working_memory) as (_, audio_vae),
            vocoder_info.model_on_device() as (_, vocoder),
        ):
            latents = encode_audio_latents(audio_vae, vocoder, torch.from_numpy(samples), sample_rate=sample_rate)

        encoded = int(latents.shape[1])
        seconds = encoded / LTX2_AUDIO_LATENTS_PER_SECOND
        num_frames = snap_num_frames_down(int(seconds * self.fps))

        if num_frames < 1 + 8:
            raise ValueError(
                f"'{self.video.video_name}' is {seconds:.2f}s long, which is under one frame group at "
                f"{self.fps:g} fps. Use a longer clip."
            )

        # Trimmed to the count the reported frame span covers. The frame count was snapped *down*,
        # so it spans slightly less than the recording, and the denoise sizes the audio stream from
        # it -- `audio_latent_count` is the same function `build_denoise_state` uses, so the two
        # agree by construction rather than by coincidence. The dropped tail is the fraction of a
        # second past the last whole frame group; the mux keeps the recording's own samples.
        num_audio_latents = audio_latent_count(num_frames, self.fps)
        if num_audio_latents > encoded:
            raise ValueError(
                f"'{self.video.video_name}' encoded to {encoded} audio latents but {num_frames} frames at "
                f"{self.fps:g} fps need {num_audio_latents}. Use a longer clip."
            )
        latents = latents[:, :num_audio_latents]

        return LTX2AudioConditioningOutput(
            audio_conditioning=LTX2AudioConditioningField(
                latents_name=context.tensors.save(tensor=latents),
                num_audio_latents=num_audio_latents,
                num_frames=num_frames,
                fps=self.fps,
                source_video_name=self.video.video_name,
            ),
            num_frames=num_frames,
            fps=self.fps,
        )
