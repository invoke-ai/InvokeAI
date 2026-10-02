"""LTX-2 node contracts that do not need a model: what each node refuses, and why."""

import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image
from pydantic import ValidationError

import invokeai.app.invocations.ltx2.ltx2_audio_conditioning as ltx2_audio_conditioning
import invokeai.app.invocations.ltx2.ltx2_extend_conditioning as ltx2_extend_conditioning
import invokeai.app.invocations.ltx2.ltx2_video_conditioning as ltx2_video_conditioning
import invokeai.app.invocations.vae.ltx2_latents_to_video as ltx2_latents_to_video
from invokeai.app.invocations.fields import (
    LatentsField,
    LTX2AudioConditioningField,
    LTX2ConditioningField,
    LTX2FullVideoConditioningField,
    LTX2VideoConditioningField,
    VideoField,
)
from invokeai.app.invocations.ltx2.ltx2_audio_conditioning import LTX2AudioConditioningInvocation
from invokeai.app.invocations.ltx2.ltx2_denoise import LTX2DenoiseInvocation
from invokeai.app.invocations.ltx2.ltx2_extend_conditioning import (
    LTX2_MAX_EXTEND_CONTEXT_FRAMES,
    LTX2ExtendConditioningInvocation,
)
from invokeai.app.invocations.ltx2.ltx2_ideal_dimensions import LTX2IdealDimensionsInvocation
from invokeai.app.invocations.ltx2.ltx2_latent_upsample import LTX2LatentUpsampleInvocation
from invokeai.app.invocations.ltx2.ltx2_model_loader import LTX2ModelLoaderInvocation
from invokeai.app.invocations.ltx2.ltx2_video_conditioning import LTX2VideoConditioningInvocation
from invokeai.app.invocations.model import (
    LoRAField,
    LTX2LatentUpsamplerField,
    LTX2TransformerField,
    LTX2VocoderField,
    ModelIdentifierField,
    VAEField,
)
from invokeai.app.invocations.vae.ltx2_latents_to_video import LTX2LatentsToVideoInvocation
from invokeai.backend.ltx2.clip_frames import CanvasClip
from invokeai.backend.ltx2.constants import (
    LTX2_AUDIO_LATENT_CHANNELS,
    LTX2_AUDIO_LATENT_MEL_BINS,
    LTX2_LATENT_CHANNELS,
)
from invokeai.backend.ltx2.denoise import build_denoise_state
from invokeai.backend.ltx2.image_conditioning import fit_to_canvas, recompress_h264
from invokeai.backend.ltx2.packing import audio_latent_count, latent_frame_count, validate_num_frames
from invokeai.backend.model_manager.configs.main import Main_Diffusers_LTX2_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, LTX2VariantType, ModelFormat, ModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import LTX2ConditioningInfo
from invokeai.backend.util.devices import TorchDevice


def _identifier(key: str = "transformer") -> ModelIdentifierField:
    return ModelIdentifierField(
        key=key, hash="hash", name=key, base=BaseModelType.LTX2, type=ModelType.Main, format=ModelFormat.Checkpoint
    )


def _denoise(**kwargs) -> LTX2DenoiseInvocation:
    defaults = {
        "transformer": LTX2TransformerField(transformer=_identifier(), variant=LTX2VariantType.Dev.value),
        "positive_conditioning": LTX2ConditioningField(conditioning_name="positive"),
        "negative_conditioning": LTX2ConditioningField(conditioning_name="negative"),
    }
    return LTX2DenoiseInvocation(id="denoise", **{**defaults, **kwargs})


def _context() -> MagicMock:
    context = MagicMock()
    context.logger = MagicMock()
    return context


@pytest.mark.parametrize(
    ("variant", "schedule", "expected"),
    [
        (LTX2VariantType.Distilled.value, "auto", True),
        (LTX2VariantType.Dev.value, "auto", False),
        (LTX2VariantType.Dev.value, "distilled", True),
        (LTX2VariantType.Distilled.value, "dev", False),
    ],
)
def test_the_schedule_follows_the_variant_unless_it_is_set_explicitly(
    variant: str, schedule: str, expected: bool
) -> None:
    """Sampling a distilled checkpoint on the dev schedule returns noise, so the default reads the
    variant the loader stamped rather than trusting a literal in the graph."""
    node = _denoise(transformer=LTX2TransformerField(transformer=_identifier(), variant=variant), schedule=schedule)
    assert node._resolve_distilled(_context()) is expected


def test_an_unstamped_transformer_cannot_resolve_the_schedule_automatically() -> None:
    node = _denoise(transformer=LTX2TransformerField(transformer=_identifier(), variant=None), schedule="auto")
    with pytest.raises(ValueError, match="Auto"):
        node._resolve_distilled(_context())


def test_the_distilled_schedule_ignores_the_guidance_scales_and_says_so() -> None:
    """Guidance is baked into the distilled weights; steering it produces a saturated, broken clip,
    so this is not a preference a graph gets to override."""
    context = _context()
    guidance = _denoise()._resolve_guidance(context, distilled=True)

    assert guidance.passes == ("cond",)
    assert (guidance.cfg_scale, guidance.audio_cfg_scale, guidance.stg_scale, guidance.modality_scale) == (
        1.0,
        1.0,
        0.0,
        1.0,
    )
    assert context.logger.info.called


def test_the_dev_schedule_passes_the_graphs_guidance_scales_through() -> None:
    node = _denoise(cfg_scale=4.0, audio_cfg_scale=6.0, stg_scale=0.5, modality_scale=2.0, guidance_rescale=0.3)
    guidance = node._resolve_guidance(_context(), distilled=False)
    assert (guidance.cfg_scale, guidance.audio_cfg_scale, guidance.stg_scale, guidance.modality_scale) == (
        4.0,
        6.0,
        0.5,
        2.0,
    )
    assert guidance.rescale == 0.3


def test_guidance_without_a_negative_prompt_is_refused_before_a_model_loads() -> None:
    node = _denoise(negative_conditioning=None)
    context = _context()
    context.conditioning.load.return_value = SimpleNamespace(
        conditionings=[
            LTX2ConditioningInfo(
                video_embeds=torch.zeros(1, 4, 8),
                audio_embeds=torch.zeros(1, 4, 6),
                attention_mask=torch.ones(1, 4, dtype=torch.int64),
            )
        ]
    )
    with pytest.raises(ValueError, match="negative conditioning"):
        node.invoke(context)


def test_an_image_conditioning_from_another_canvas_is_refused_with_both_sizes() -> None:
    node = _denoise(
        width=1248,
        height=704,
        video_conditioning=LTX2VideoConditioningField(latents_name="latents", width=768, height=512),
    )
    with pytest.raises(ValueError, match="768x512"):
        node._load_image_latents(_context())


@pytest.mark.parametrize("num_frames", [120, 122])
def test_a_frame_count_off_the_vae_grid_is_refused(num_frames: int) -> None:
    with pytest.raises(ValueError, match="8n \\+ 1"):
        _denoise(num_frames=num_frames).invoke(_context())


def test_the_working_memory_estimate_grows_with_the_sequence() -> None:
    """Under-reserving packs VRAM with weights and kills the first forward; the estimate has to
    follow the row count rather than being a constant."""
    small = LTX2DenoiseInvocation._estimate_working_memory(320, 9)
    large = LTX2DenoiseInvocation._estimate_working_memory(13728, 126)
    assert large > small
    # Measured on a W7900 at these shapes: 0.46 GiB and 2.81 GiB.
    assert small > 0.46 * 2**30
    assert large > 2.81 * 2**30


def test_the_model_loader_refuses_a_single_file_transformer_with_no_component_folder() -> None:
    """A checkpoint carries the transformer alone; without a folder there is no VAE to decode with,
    and the failure would otherwise surface minutes later inside a loader."""
    node = LTX2ModelLoaderInvocation(
        id="loader",
        model=_identifier(),
        text_encoder_model=_identifier("encoder"),
    )
    context = _context()
    context.models.exists.return_value = True
    context.models.get_config.return_value = SimpleNamespace(
        base=BaseModelType.LTX2, type=ModelType.Main, name="LTX-2.5 Dev", format=ModelFormat.Checkpoint
    )
    with pytest.raises(ValueError, match="Components field"):
        node.invoke(context)


def test_the_model_loader_refuses_a_components_only_folder_as_the_model() -> None:
    node = LTX2ModelLoaderInvocation(id="loader", model=_identifier(), text_encoder_model=_identifier("encoder"))
    context = _context()
    context.models.exists.return_value = True
    context.models.get_config.return_value = Main_Diffusers_LTX2_Config.model_construct(
        base=BaseModelType.LTX2, type=ModelType.Main, name="LTX-2.5 Components", components_only=True
    )
    with pytest.raises(ValueError, match="components-only"):
        node.invoke(context)


def test_the_model_loader_refuses_a_model_from_another_architecture() -> None:
    node = LTX2ModelLoaderInvocation(id="loader", model=_identifier(), text_encoder_model=_identifier("encoder"))
    context = _context()
    context.models.exists.return_value = True
    context.models.get_config.return_value = SimpleNamespace(
        base=BaseModelType.Wan, type=ModelType.Main, name="Wan", format=ModelFormat.Diffusers
    )
    with pytest.raises(ValueError, match="Model field needs an LTX-2 main model"):
        node.invoke(context)


@pytest.mark.parametrize(
    ("source", "preset", "expected"),
    [((1920, 1080), "704p", (1248, 704)), ((1080, 1920), "512p", (512, 896)), ((1024, 1024), "768p", (768, 768))],
)
def test_the_ideal_dimensions_pin_the_short_edge(source, preset, expected) -> None:
    node = LTX2IdealDimensionsInvocation(id="dims", width=source[0], height=source[1], target_resolution=preset)
    output = node.invoke(MagicMock())
    assert (output.width, output.height) == expected
    # One pass: the base canvas is the canvas, so a workflow can wire one pair of numbers.
    assert (output.base_width, output.base_height) == expected
    assert output.two_stage is False


@pytest.mark.parametrize(
    ("source", "preset", "final", "base"),
    [
        ((1920, 1080), "1024p", (1792, 1024), (896, 512)),
        ((1920, 1080), "1536p", (2752, 1536), (1376, 768)),
        ((1080, 1920), "1024p", (1024, 1792), (512, 896)),
    ],
)
def test_a_two_stage_preset_resolves_on_the_64_grid_and_names_its_base_pass(source, preset, final, base) -> None:
    """The node's choice of grid is what makes the base canvas expressible at all: a two-stage canvas
    off the 64 grid halves onto something the VAE cannot encode, and only a live run would notice."""
    node = LTX2IdealDimensionsInvocation(id="dims", width=source[0], height=source[1], target_resolution=preset)
    output = node.invoke(MagicMock())

    assert (output.width, output.height) == final
    assert (output.base_width, output.base_height) == base
    assert output.two_stage is True
    assert (output.base_width * 2, output.base_height * 2) == final
    assert output.base_width % 32 == 0 and output.base_height % 32 == 0


def test_fitting_an_image_to_the_canvas_crops_rather_than_stretches() -> None:
    """A stretched first frame teaches the model the wrong geometry for the whole clip."""
    source = Image.new("RGB", (1000, 500), "red")
    source.paste(Image.new("RGB", (100, 100), "blue"), (450, 200))

    fitted = fit_to_canvas(source, 704, 704)

    assert fitted.size == (704, 704)
    # A 2:1 source cover-cropped to a square keeps the centre, so the blue patch survives.
    assert fitted.getpixel((352, 352)) == (0, 0, 255)


def test_the_crf_round_trip_returns_a_recompressed_frame_of_the_same_size() -> None:
    """The conditioning image has to carry codec artefacts; the round trip goes through ffmpeg,
    whose two-process pipe is easy to get subtly wrong."""
    torch.manual_seed(0)
    noise = (torch.rand(96, 128, 3) * 255).byte().numpy()
    source = Image.fromarray(noise, "RGB")

    recompressed = recompress_h264(source, 18)

    assert recompressed.size == source.size
    assert recompressed.tobytes() != source.tobytes()
    assert recompress_h264(source, 0) is source


def test_an_odd_sized_image_is_cropped_to_what_h264_can_encode() -> None:
    source = Image.new("RGB", (65, 33), "green")
    assert recompress_h264(source, 18).size == (64, 32)


def _latents_to_video(**kwargs) -> LTX2LatentsToVideoInvocation:
    defaults = {
        "video_latents": LatentsField(latents_name="video"),
        "audio_latents": LatentsField(latents_name="audio"),
        "vae": VAEField(vae=_identifier("vae")),
        "audio_vae": VAEField(vae=_identifier("audio_vae")),
        "vocoder": LTX2VocoderField(vocoder=_identifier("vocoder")),
    }
    return LTX2LatentsToVideoInvocation(id="l2v", **{**defaults, **kwargs})


def test_audio_latents_without_their_decoders_name_both_missing_models() -> None:
    """The graph always wires all three, so this catches a hand-built workflow before it decodes a
    whole clip and then finds it has nothing to turn the soundtrack into."""
    node = _latents_to_video(audio_vae=None, vocoder=None)

    with pytest.raises(ValueError, match="Audio VAE and Vocoder"):
        node._decode_audio_to_wav(_context(), 5.0)


@pytest.mark.parametrize(
    ("decoded_seconds", "clip_seconds"),
    [
        # The audio VAE's causal decoder drops its first few mel frames, so the soundtrack comes
        # back a fraction short of the clip: the normal case, and the one the pad exists for.
        (4.97, 5.0),
        # And a grid that overshoots has to be cut, or the mux would run past the last frame.
        (5.2, 5.0),
        (5.0, 5.0),
    ],
)
def test_the_soundtrack_is_written_at_exactly_the_clip_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, decoded_seconds: float, clip_seconds: float
) -> None:
    sample_rate = 48000
    decoded = torch.zeros(2, int(decoded_seconds * sample_rate))
    decoded[:, : decoded.shape[1] // 2] = 0.5

    monkeypatch.setattr(ltx2_latents_to_video, "decode_audio_latents", lambda *_a, **_k: decoded)
    context = _context()
    context.tensors.load.return_value = torch.zeros(1, 126, 128)
    context.models.load.return_value.model_on_device.return_value.__enter__.return_value = (
        None,
        SimpleNamespace(config=SimpleNamespace(output_sampling_rate=sample_rate)),
    )

    wav_path = _latents_to_video()._decode_audio_to_wav(context, clip_seconds)
    try:
        with wave.open(str(wav_path)) as handle:
            assert handle.getnchannels() == 2
            assert handle.getframerate() == sample_rate
            assert handle.getnframes() == int(round(clip_seconds * sample_rate))
    finally:
        wav_path.unlink(missing_ok=True)


def test_a_mono_soundtrack_is_refused_rather_than_written_as_half_a_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ltx2_latents_to_video, "decode_audio_latents", lambda *_a, **_k: torch.zeros(1, 48000))
    context = _context()
    context.tensors.load.return_value = torch.zeros(1, 126, 128)
    context.models.load.return_value.model_on_device.return_value.__enter__.return_value = (
        None,
        SimpleNamespace(config=SimpleNamespace(output_sampling_rate=48000)),
    )

    with pytest.raises(ValueError, match="expected stereo"):
        _latents_to_video()._decode_audio_to_wav(context, 1.0)


def test_unpacked_audio_latents_are_refused_before_a_model_is_locked() -> None:
    context = _context()
    context.tensors.load.return_value = torch.zeros(1, 8, 126, 16)

    with pytest.raises(ValueError, match=r"packed \[1, L, 128\]"):
        _latents_to_video()._decode_audio_to_wav(context, 5.0)


def test_a_refine_pass_without_the_base_passs_audio_is_refused() -> None:
    """The two modalities are denoised jointly off one pair of timesteps, so the refine pass takes
    both of stage one's outputs or neither -- wiring only the video would leave the audio to be
    generated from scratch beside an almost-finished clip."""
    node = _denoise(latents=LatentsField(latents_name="upscaled"))

    with pytest.raises(ValueError, match="audio latents as well as its video"):
        node.invoke(_context())


def test_audio_latents_without_video_latents_are_refused() -> None:
    node = _denoise(audio_latents=LatentsField(latents_name="audio"))

    with pytest.raises(ValueError, match="takes both or neither"):
        node.invoke(_context())


def test_the_upscaler_refuses_latents_that_are_not_one_ltx2_clip() -> None:
    node = LTX2LatentUpsampleInvocation(
        id="upsample",
        video_latents=LatentsField(latents_name="latents"),
        latent_upsampler=LTX2LatentUpsamplerField(latent_upsampler=_identifier("upsampler")),
        vae=VAEField(vae=_identifier("vae")),
    )
    context = _context()
    context.tensors.load.return_value = torch.zeros(1, 16, 4, 8, 8)

    with pytest.raises(ValueError, match="expects one 5D clip"):
        node.invoke(context)


def test_the_upscaler_hands_the_network_raw_latents_and_returns_normalized_ones(monkeypatch) -> None:
    """The scale conversion is the whole reason the VAE is wired into this node: the upscaler was
    trained on the VAE's own latent scale while the transformer reads normalized latents. Swapping
    the two conversions, or dropping either, leaves a wildly mis-scaled latent that only shows up as
    garbage after the refine pass -- by which time the base pass has already run."""
    mean = torch.full((1, 128, 1, 1, 1), 3.0)
    std = torch.full((1, 128, 1, 1, 1), 2.0)
    scaling_factor = 0.5
    seen: dict[str, torch.Tensor] = {}

    class StubUpsampler(torch.nn.Module):
        # Carries the same submodule names as the real `LTX2LatentUpsamplerModel`, as real modules:
        # the node hooks them for cancellation, so a stub of bare lists would let a typo'd or
        # renamed attribute pass here and fail only against the released checkpoint.
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.initial_conv = torch.nn.Identity()
            self.res_blocks = torch.nn.ModuleList([torch.nn.Identity()])
            self.upsampler = torch.nn.Identity()
            self.post_upsample_res_blocks = torch.nn.ModuleList([torch.nn.Identity()])
            self.final_conv = torch.nn.Identity()

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            seen["input"] = hidden_states.detach().clone()
            return hidden_states.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)

    upsampler = StubUpsampler()
    vae = SimpleNamespace(latents_mean=mean, latents_std=std, config=SimpleNamespace(scaling_factor=scaling_factor))

    context = _context()
    normalized_in = torch.randn(1, 128, 2, 4, 6)
    context.tensors.load.return_value = normalized_in

    def save(tensor: torch.Tensor) -> str:
        seen["saved"] = tensor.clone()
        return "saved"

    context.tensors.save.side_effect = save

    def load(identifier):
        if identifier.key == "vae":
            return SimpleNamespace(model=vae, config=SimpleNamespace(base=BaseModelType.LTX2))
        loaded = MagicMock()
        loaded.model_on_device.return_value.__enter__.return_value = (None, upsampler)
        return loaded

    context.models.load.side_effect = load
    monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))

    node = LTX2LatentUpsampleInvocation(
        id="upsample",
        video_latents=LatentsField(latents_name="latents"),
        latent_upsampler=LTX2LatentUpsamplerField(latent_upsampler=_identifier("upsampler")),
        vae=VAEField(vae=_identifier("vae")),
    )
    output = node.invoke(context)

    # In: denormalized to the VAE's own scale. Out: back on the transformer's.
    torch.testing.assert_close(seen["input"], normalized_in * std / scaling_factor + mean)
    torch.testing.assert_close(
        seen["saved"],
        (seen["input"].repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2) - mean) * scaling_factor / std,
    )
    # Pixel geometry, not `size()[3] * 8`: 4x6 latents doubled is 8x12, at 32 px per latent.
    assert (output.width, output.height, output.num_frames) == (12 * 32, 8 * 32, (2 - 1) * 8 + 1)


def test_the_refine_pass_forwards_the_noise_level_the_node_was_given(monkeypatch) -> None:
    """`noise_scale` is the only knob this feature adds, and nothing else reaches the schedule it
    controls without a 22B transformer resident. Hard-coding it at the call site is otherwise free."""
    import invokeai.app.invocations.ltx2.ltx2_denoise as denoise_module

    captured: dict[str, float] = {}

    def fake_build_refine_state(**kwargs):
        captured["noise_scale"] = kwargs["noise_scale"]
        raise _StopAfterState

    monkeypatch.setattr(denoise_module, "build_refine_state", fake_build_refine_state)
    node = _denoise(
        latents=LatentsField(latents_name="upscaled"),
        audio_latents=LatentsField(latents_name="audio"),
        noise_scale=0.42,
        cfg_scale=1.0,
        audio_cfg_scale=1.0,
        stg_scale=0.0,
        modality_scale=1.0,
    )

    context = _context()
    context.conditioning.load.return_value = SimpleNamespace(
        conditionings=[
            LTX2ConditioningInfo(
                video_embeds=torch.zeros(1, 4, 8),
                audio_embeds=torch.zeros(1, 4, 6),
                attention_mask=torch.ones(1, 4, dtype=torch.int64),
            )
        ]
    )

    with pytest.raises(_StopAfterState):
        node.invoke(context)

    assert captured["noise_scale"] == 0.42


# ---------------------------------------------------------------------------
# Whole-modality conditioning: a clip's soundtrack, or its picture


def _audio_conditioning(**kwargs) -> LTX2AudioConditioningInvocation:
    defaults = {
        "video": VideoField(video_name="clip.mp4"),
        "audio_vae": VAEField(vae=_identifier("audio_vae")),
        "vocoder": LTX2VocoderField(vocoder=_identifier("vocoder")),
    }
    return LTX2AudioConditioningInvocation(id="audio_cond", **{**defaults, **kwargs})


def _video_conditioning(**kwargs) -> LTX2VideoConditioningInvocation:
    defaults = {"video": VideoField(video_name="clip.mp4"), "vae": VAEField(vae=_identifier("vae"))}
    return LTX2VideoConditioningInvocation(id="video_cond", **{**defaults, **kwargs})


def _ltx2_vae_context(base: BaseModelType = BaseModelType.LTX2) -> MagicMock:
    context = _context()
    context.models.load.return_value.config.base = base
    context.models.load.return_value.model_on_device.return_value.__enter__.return_value = (None, MagicMock())
    return context


def test_a_clip_with_no_soundtrack_says_so_instead_of_conditioning_on_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silent clip would encode to a valid, uniformly quiet latent -- the model would hold it
    clean and generate a picture scored against nothing."""
    monkeypatch.setattr(ltx2_audio_conditioning, "extract_audio_pcm", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="no audio track"):
        _audio_conditioning().invoke(_ltx2_vae_context())


@pytest.mark.parametrize("node", ["audio", "video"])
def test_a_conditioning_node_refuses_a_vae_from_another_architecture(
    monkeypatch: pytest.MonkeyPatch, node: str
) -> None:
    monkeypatch.setattr(ltx2_audio_conditioning, "extract_audio_pcm", lambda *_a, **_k: (np.zeros((2, 48000)), 48000))
    invocation = _audio_conditioning() if node == "audio" else _video_conditioning()

    with pytest.raises(ValueError, match="Expected an LTX-2"):
        invocation.invoke(_ltx2_vae_context(BaseModelType.Wan))


@pytest.mark.parametrize(
    ("audio_latents", "fps", "expected_frames"),
    [
        # 25 audio latents per second: 100 latents is four seconds, which at 24 fps is 96 frames
        # and snaps DOWN to 89 -- the clip is never asked to cover more picture than it has sound.
        (100, 24.0, 89),
        (100, 30.0, 113),
        # A rate that divides evenly still snaps down rather than to the nearest.
        (75, 24.0, 65),
    ],
)
def test_the_soundtracks_length_decides_the_frame_count(
    monkeypatch: pytest.MonkeyPatch, audio_latents: int, fps: float, expected_frames: int
) -> None:
    monkeypatch.setattr(ltx2_audio_conditioning, "extract_audio_pcm", lambda *_a, **_k: (np.zeros((2, 48000)), 48000))
    monkeypatch.setattr(
        ltx2_audio_conditioning,
        "encode_audio_latents",
        lambda *_a, **_k: torch.zeros(1, audio_latents, LTX2_AUDIO_LATENT_CHANNELS * LTX2_AUDIO_LATENT_MEL_BINS),
    )
    context = _ltx2_vae_context()
    context.tensors.save.return_value = "audio_conditioning"

    output = _audio_conditioning(fps=fps).invoke(context)

    assert output.num_frames == expected_frames
    validate_num_frames(output.num_frames)
    assert output.fps == fps
    # The saved soundtrack is trimmed to the span the frame count covers, so the field reports that
    # count rather than everything the encoder produced -- see the seam test below.
    assert output.audio_conditioning.num_audio_latents == audio_latent_count(expected_frames, fps)
    # The output field carries the clip back to the decode node, which muxes the original
    # recording in place of the generated soundtrack.
    assert output.audio_conditioning.source_video_name == "clip.mp4"


@pytest.mark.parametrize(
    ("audio_latents", "fps"),
    [(100, 24.0), (100, 30.0), (75, 24.0), (125, 25.0), (251, 24.0), (1000, 60.0), (63, 16.0)],
)
def test_the_soundtrack_this_node_saves_is_the_one_the_denoise_asks_for(
    monkeypatch: pytest.MonkeyPatch, audio_latents: int, fps: float
) -> None:
    """The seam between the two halves of audio-to-video, which neither side can check alone.

    This node decides the frame count; `build_denoise_state` then sizes the audio stream from that
    count and refuses a tensor of any other length. Because the count is snapped DOWN, it spans
    slightly less than the recording -- so saving the whole encode makes the two disagree on every
    real soundtrack, and the run dies after the text encoder has already paid for itself.
    """
    monkeypatch.setattr(ltx2_audio_conditioning, "extract_audio_pcm", lambda *_a, **_k: (np.zeros((2, 48000)), 48000))
    monkeypatch.setattr(
        ltx2_audio_conditioning,
        "encode_audio_latents",
        lambda *_a, **_k: torch.zeros(1, audio_latents, LTX2_AUDIO_LATENT_CHANNELS * LTX2_AUDIO_LATENT_MEL_BINS),
    )
    saved: dict[str, torch.Tensor] = {}
    context = _ltx2_vae_context()

    def capture(tensor: torch.Tensor) -> str:
        saved["latents"] = tensor
        return "audio"

    context.tensors.save.side_effect = capture

    output = _audio_conditioning(fps=fps).invoke(context)

    # The real denoise state, built exactly as the graph builds it: the node's own frame count and
    # frame rate, and the tensor it actually saved.
    state = build_denoise_state(
        num_frames=output.num_frames,
        height=512,
        width=768,
        fps=output.fps,
        seed=1,
        distilled=False,
        num_steps=2,
        frozen_audio_latents=saved["latents"],
    )

    assert state.audio_conditioning_mask is not None
    assert state.audio_conditioning_mask.shape == (1, state.audio_latents_count)
    assert state.audio_latents.shape[1] == state.audio_latents_count
    # And the node reports what it saved, so the field is not a second, divergent source of truth.
    assert output.audio_conditioning.num_audio_latents == saved["latents"].shape[1]


def test_a_soundtrack_under_one_frame_group_is_refused_with_its_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ltx2_audio_conditioning, "extract_audio_pcm", lambda *_a, **_k: (np.zeros((2, 4800)), 48000))
    monkeypatch.setattr(
        ltx2_audio_conditioning,
        "encode_audio_latents",
        lambda *_a, **_k: torch.zeros(1, 5, LTX2_AUDIO_LATENT_CHANNELS * LTX2_AUDIO_LATENT_MEL_BINS),
    )

    with pytest.raises(ValueError, match="0.20s long"):
        _audio_conditioning().invoke(_ltx2_vae_context())


def test_the_conditioning_clip_drops_its_ragged_tail_rather_than_padding_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Padding would invent picture for the model to score a soundtrack against; the frames past
    the last whole group are simply not part of the generation."""
    frames = [np.full((512, 768, 3), index % 256, dtype=np.uint8) for index in range(20)]
    encoded: dict[str, torch.Tensor] = {}

    monkeypatch.setattr(ltx2_video_conditioning, "read_canvas_frames", lambda *_a, **_k: _canvas_clip(list(frames)))

    def encode(pixels):
        encoded["pixels"] = pixels
        latent_frames = latent_frame_count(17)
        return SimpleNamespace(
            latent_dist=SimpleNamespace(
                mode=lambda: torch.zeros(1, LTX2_LATENT_CHANNELS, latent_frames, 512 // 32, 768 // 32)
            )
        )

    # The encode is tiled, so the stub carries the compression ratios and the tiling attributes
    # `scoped_ltx2_tiling` snapshots and restores -- a VAE missing them is one this node cannot run.
    vae = SimpleNamespace(
        config=SimpleNamespace(scaling_factor=1.0),
        enable_tiling=lambda **_kwargs: None,
        encode=encode,
        latents_mean=torch.zeros(LTX2_LATENT_CHANNELS),
        latents_std=torch.ones(LTX2_LATENT_CHANNELS),
        buffers=lambda: iter([]),
        encoder=torch.nn.Identity(),
        parameters=lambda: iter([torch.zeros(1)]),
        spatial_compression_ratio=32,
        temporal_compression_ratio=8,
        use_framewise_encoding=False,
        use_framewise_decoding=False,
        use_tiling=False,
    )
    context = _ltx2_vae_context()
    context.models.load.return_value.model_on_device.return_value.__enter__.return_value = (None, vae)
    context.tensors.save.return_value = "video_conditioning"

    output = _video_conditioning(width=768, height=512).invoke(context)

    # 20 frames is two whole groups plus three: 17 frames run, three are dropped.
    assert output.num_frames == 17
    assert encoded["pixels"].shape == (1, 3, 17, 512, 768)
    # And the frames arrive in [-1, 1] with the VAE's own scaling, not raw bytes. The conversion
    # runs in the VAE's dtype to avoid a second copy of the clip, so the order of operations
    # matters: `x / 127.5 - 1` cancels at mid-grey in bf16 and loses the value entirely.
    sources = torch.tensor([index % 256 for index in range(17)], dtype=torch.float32)
    expected = sources.div(127.5).sub(1.0)
    assert torch.allclose(encoded["pixels"][0, 0, :, 0, 0].float(), expected, atol=2e-3)
    assert output.video_conditioning.width == 768
    assert output.video_conditioning.height == 512


def test_a_conditioning_clip_under_one_frame_group_is_refused_with_its_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ltx2_video_conditioning,
        "read_canvas_frames",
        lambda *_a, **_k: _canvas_clip([np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]),
    )

    with pytest.raises(ValueError, match="decoded to 5 frame"):
        _video_conditioning().invoke(_ltx2_vae_context())


@pytest.mark.parametrize(
    "conditioning",
    [
        {
            "audio_conditioning": LTX2AudioConditioningField(
                latents_name="audio", num_audio_latents=100, num_frames=89, fps=24.0, source_video_name="clip.mp4"
            )
        },
        {
            "full_video_conditioning": LTX2FullVideoConditioningField(
                latents_name="video", width=1248, height=704, num_frames=89, fps=24.0, source_video_name="clip.mp4"
            )
        },
    ],
)
def test_a_held_modality_cannot_be_combined_with_a_refine_pass(conditioning: dict) -> None:
    """The refine pass re-noises every token, so the held stream would have to be re-encoded at the
    second canvas. Refused before the transformer loads rather than silently dropped."""
    node = _denoise(
        latents=LatentsField(latents_name="upscaled"),
        audio_latents=LatentsField(latents_name="audio"),
        num_frames=89,
        **conditioning,
    )

    with pytest.raises(ValueError, match="does not run a refine pass"):
        node.invoke(_context())


@pytest.mark.parametrize(
    ("conditioning", "expected"),
    [
        (
            {
                "audio_conditioning": LTX2AudioConditioningField(
                    latents_name="audio", num_audio_latents=93, num_frames=89, fps=30.0, source_video_name="clip.mp4"
                )
            },
            "fps 30.0 vs 24.0",
        ),
        (
            {
                "full_video_conditioning": LTX2FullVideoConditioningField(
                    latents_name="video", width=768, height=704, num_frames=89, fps=24.0, source_video_name="clip.mp4"
                )
            },
            "width 768 vs 1248",
        ),
    ],
)
def test_a_conditioning_prepared_for_a_different_run_is_named_before_the_transformer_loads(
    conditioning: dict, expected: str
) -> None:
    """`fps` is the one the latent shapes cannot catch -- no tensor encodes it -- and it sets the
    audio stream's length, so a mismatch would quietly generate a soundtrack for the wrong
    duration. The canvas and frame count are checked here too, where the message can name them."""
    node = _denoise(num_frames=89, fps=24.0, width=1248, height=704, **conditioning)

    with pytest.raises(ValueError, match=expected):
        node.invoke(_context())


def _keyframe_field(frame_index: int, **kwargs) -> LTX2VideoConditioningField:
    defaults = {"latents_name": "keyframe", "width": 1248, "height": 704, "frame_index": frame_index}
    return LTX2VideoConditioningField(**{**defaults, **kwargs})


@pytest.mark.parametrize(
    ("num_frames", "frame_index", "expected"),
    # 121 frames is 16 latent frames; 9 frames is 2.
    [(121, -1, 15), (121, 15, 15), (121, 1, 1), (121, -15, 1), (9, -1, 1)],
)
def test_a_keyframes_index_is_resolved_against_the_clips_own_length(
    num_frames: int, frame_index: int, expected: int
) -> None:
    """Negative indices count from the end, and they are resolved HERE rather than at the encode:
    the conditioning node does not know how long the clip is, so resolving there would silently
    land the frame at the wrong instant the moment someone changed the frame count."""
    node = _denoise(num_frames=num_frames, keyframe_conditioning=_keyframe_field(frame_index))
    context = _context()
    context.tensors.load.return_value = torch.zeros(1, LTX2_LATENT_CHANNELS, 1, 22, 39)

    assert node._resolve_keyframe(context)[1] == expected


def test_a_keyframe_that_resolves_to_the_first_frame_names_the_input_that_owns_it() -> None:
    """Index 0 is a different mechanism -- overwriting the grid rather than appending -- so this is
    a wiring mistake, and the message has to say which input to use instead."""
    node = _denoise(num_frames=121, keyframe_conditioning=_keyframe_field(0))

    with pytest.raises(ValueError, match="Image Conditioning"):
        node._resolve_keyframe(_context())


@pytest.mark.parametrize("frame_index", [16, 40, -17, -200])
def test_a_keyframe_outside_the_clip_is_refused_with_both_indices(frame_index: int) -> None:
    node = _denoise(num_frames=121, keyframe_conditioning=_keyframe_field(frame_index))

    with pytest.raises(ValueError, match="outside a 121-frame clip"):
        node._resolve_keyframe(_context())


def test_a_keyframe_encoded_for_another_canvas_is_refused_before_the_transformer_loads() -> None:
    node = _denoise(num_frames=121, keyframe_conditioning=_keyframe_field(-1, width=768, height=512))

    with pytest.raises(ValueError, match="keyframe conditioning was prepared"):
        node._resolve_keyframe(_context())


@pytest.mark.parametrize("refine", [False, True])
def test_the_resolved_keyframe_reaches_the_state_the_run_is_built_from(monkeypatch, refine: bool) -> None:
    """Resolving the index correctly is worth nothing if it is not handed on, and the resolution is
    only reachable through a private helper -- so this crosses the seam between them, for the base
    pass and the refine pass alike. A two-stage run that dropped the keyframe would end somewhere
    else than a single-stage one."""
    import invokeai.app.invocations.ltx2.ltx2_denoise as denoise_module

    captured: dict[str, object] = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        raise _StopAfterState

    monkeypatch.setattr(denoise_module, "build_refine_state" if refine else "build_denoise_state", fake_build)
    node = _denoise(
        num_frames=121,
        keyframe_conditioning=_keyframe_field(-1, strength=0.75),
        **(
            {"latents": LatentsField(latents_name="upscaled"), "audio_latents": LatentsField(latents_name="audio")}
            if refine
            else {}
        ),
        cfg_scale=1.0,
        audio_cfg_scale=1.0,
        stg_scale=0.0,
        modality_scale=1.0,
    )
    context = _context()
    context.tensors.load.return_value = torch.zeros(1, LTX2_LATENT_CHANNELS, 1, 22, 39)
    context.conditioning.load.return_value = SimpleNamespace(
        conditionings=[
            LTX2ConditioningInfo(
                video_embeds=torch.zeros(1, 4, 8),
                audio_embeds=torch.zeros(1, 4, 6),
                attention_mask=torch.ones(1, 4, dtype=torch.int64),
            )
        ]
    )

    with pytest.raises(_StopAfterState):
        node.invoke(context)

    assert captured["keyframe_latent_index"] == 15
    assert captured["keyframe_strength"] == 0.75
    assert captured["keyframe_latents"] is not None


@pytest.mark.parametrize("refine", [False, True])
def test_the_held_soundtrack_opening_reaches_both_passes(monkeypatch: pytest.MonkeyPatch, refine: bool) -> None:
    """A continuation's held sound has to survive the whole run. Stage two re-noises every row, so a
    prefix handed only to stage one is gone by the time the join crossfades -- which is exactly the
    seam this exists to remove, and a two-stage run would have it back with nothing failing."""
    import invokeai.app.invocations.ltx2.ltx2_denoise as denoise_module

    captured: dict[str, object] = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        raise _StopAfterState

    monkeypatch.setattr(denoise_module, "build_refine_state" if refine else "build_denoise_state", fake_build)
    prefix = torch.arange(18 * 128, dtype=torch.float32).reshape(1, 18, 128)
    node = _denoise(
        num_frames=121,
        audio_prefix_conditioning=LTX2AudioConditioningField(
            latents_name="held_opening",
            num_audio_latents=18,
            num_frames=17,
            fps=24.0,
            source_video_name="clip.mp4",
        ),
        **(
            {"latents": LatentsField(latents_name="upscaled"), "audio_latents": LatentsField(latents_name="audio")}
            if refine
            else {}
        ),
        cfg_scale=1.0,
        audio_cfg_scale=1.0,
        stg_scale=0.0,
        modality_scale=1.0,
    )
    context = _context()
    context.tensors.load.side_effect = lambda name: (
        prefix if name == "held_opening" else torch.zeros(1, LTX2_LATENT_CHANNELS, 16, 22, 39)
    )
    context.conditioning.load.return_value = SimpleNamespace(
        conditionings=[
            LTX2ConditioningInfo(
                video_embeds=torch.zeros(1, 4, 8),
                audio_embeds=torch.zeros(1, 4, 6),
                attention_mask=torch.ones(1, 4, dtype=torch.int64),
            )
        ]
    )

    with pytest.raises(_StopAfterState):
        node.invoke(context)

    assert torch.equal(captured["audio_prefix_latents"], prefix)


def _canvas_clip(frames: list[np.ndarray], source_frames: int | None = None) -> CanvasClip:
    """What the reader hands back: the fitted frames, and how long the clip they came from is."""
    return CanvasClip(frames, len(frames) if source_frames is None else source_frames)


def test_a_held_opening_prepared_at_another_frame_rate_is_refused() -> None:
    """The rate is the one thing no tensor shape records. A prefix sized at 24 fps and held at 30
    covers a different stretch of time than the picture it belongs to, so the join crossfades real
    source audio into the new material -- and the row count stays well inside what the clip has
    room for, so nothing else would notice."""
    node = _denoise(
        num_frames=121,
        fps=30.0,
        audio_prefix_conditioning=LTX2AudioConditioningField(
            latents_name="held_opening",
            num_audio_latents=18,
            num_frames=17,
            fps=24.0,
            source_video_name="clip.mp4",
        ),
    )

    with pytest.raises(ValueError, match="Extend Conditioning - LTX-2 was prepared"):
        node.invoke(_context())


def _extend_conditioning(**kwargs) -> LTX2ExtendConditioningInvocation:
    defaults = {"video": VideoField(video_name="clip.mp4"), "vae": VAEField(vae=_identifier("vae"))}
    return LTX2ExtendConditioningInvocation(id="extend_cond", **{**defaults, **kwargs})


def _stub_video_vae(context: MagicMock, latent_frames: int, height: int, width: int) -> dict:
    encoded: dict = {}

    def encode(pixels):
        encoded["pixels"] = pixels
        return SimpleNamespace(
            latent_dist=SimpleNamespace(
                mode=lambda: torch.zeros(1, LTX2_LATENT_CHANNELS, latent_frames, height // 32, width // 32)
            )
        )

    vae = SimpleNamespace(
        buffers=lambda: iter([]),
        config=SimpleNamespace(scaling_factor=1.0),
        enable_tiling=lambda **_kwargs: None,
        encode=encode,
        encoder=torch.nn.Identity(),
        latents_mean=torch.zeros(LTX2_LATENT_CHANNELS),
        latents_std=torch.ones(LTX2_LATENT_CHANNELS),
        parameters=lambda: iter([torch.zeros(1)]),
        spatial_compression_ratio=32,
        temporal_compression_ratio=8,
        use_framewise_decoding=False,
        use_framewise_encoding=False,
        use_tiling=False,
    )
    context.models.load.return_value.model_on_device.return_value.__enter__.return_value = (None, vae)
    return encoded


def test_an_extension_conditions_on_the_tail_of_the_clip_not_its_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point is to continue from where the clip ENDED. Reading the head would anchor the
    continuation to the wrong moment, and nothing downstream could tell."""
    captured: dict = {}

    def fake_read(_path, **kwargs):
        captured.update(kwargs)
        return _canvas_clip([np.full((512, 768, 3), index, dtype=np.uint8) for index in range(kwargs["cap"])])

    monkeypatch.setattr(ltx2_extend_conditioning, "read_canvas_frames", fake_read)
    context = _ltx2_vae_context()
    _stub_video_vae(context, latent_frames=3, height=512, width=768)
    context.tensors.save.return_value = "extend_conditioning"

    output = _extend_conditioning(width=768, height=512, context_frames=17).invoke(context)

    assert captured["tail"] is True
    assert captured["cap"] == 17
    assert output.context_frames == 17
    # Held from the first frame onward, which is what makes it a leading anchor rather than a keyframe.
    assert output.video_conditioning.frame_index == 0


@pytest.mark.parametrize(("requested", "expected"), [(17, 17), (20, 17), (9, 9), (100, 97)])
def test_the_context_length_is_snapped_down_to_whole_frame_groups(
    monkeypatch: pytest.MonkeyPatch, requested: int, expected: int
) -> None:
    """The VAE encodes 8k + 1 frames, so a ragged request would have its tail dropped after the
    read. Snapping first means the read asks for what it can actually use."""
    captured: dict = {}

    def fake_read(_path, **kwargs):
        captured.update(kwargs)
        return _canvas_clip([np.zeros((512, 768, 3), dtype=np.uint8) for _ in range(kwargs["cap"])])

    monkeypatch.setattr(ltx2_extend_conditioning, "read_canvas_frames", fake_read)
    context = _ltx2_vae_context()
    _stub_video_vae(context, latent_frames=(expected - 1) // 8 + 1, height=512, width=768)
    context.tensors.save.return_value = "extend_conditioning"

    output = _extend_conditioning(width=768, height=512, context_frames=requested).invoke(context)

    assert output.context_frames == expected
    # Snapped BEFORE the read, so the decode is asked for what the VAE can actually use rather than
    # for frames that are then thrown away.
    assert captured["cap"] == expected


def test_a_source_shorter_than_the_context_keeps_its_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip with fewer frames than asked for still has to contribute its last whole group.
    Trimming the wrong end anchors the continuation to where the clip BEGAN -- which still encodes,
    still validates, and is simply the wrong moment."""
    frames = [np.full((512, 768, 3), index, dtype=np.uint8) for index in range(12)]
    monkeypatch.setattr(
        ltx2_extend_conditioning, "read_canvas_frames", lambda _path, **_kwargs: _canvas_clip(list(frames))
    )
    context = _ltx2_vae_context()
    encoded = _stub_video_vae(context, latent_frames=2, height=512, width=768)
    context.tensors.save.return_value = "extend_conditioning"

    output = _extend_conditioning(width=768, height=512, context_frames=17).invoke(context)

    assert output.context_frames == 9
    # Frames 3..11 of the source, not 0..8.
    held = encoded["pixels"][0, 0, :, 0, 0].float()
    assert held.tolist() == pytest.approx([(index / 127.5) - 1.0 for index in range(3, 12)], abs=2e-3)


def test_a_clip_too_short_to_continue_from_says_how_much_is_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ltx2_extend_conditioning,
        "read_canvas_frames",
        lambda _path, **_kwargs: _canvas_clip([np.zeros((512, 768, 3), dtype=np.uint8) for _ in range(5)]),
    )

    with pytest.raises(ValueError, match="at least 9 frames of the source"):
        _extend_conditioning(width=768, height=512).invoke(_ltx2_vae_context())


# Ten seconds at 24 fps. The audio stubs hand the node a track of exactly this length, so its trim
# to the picture's span is a no-op unless a test deliberately makes the decode longer.
SOURCE_CLIP_FRAMES = 240


def _stub_extend_audio(
    monkeypatch: pytest.MonkeyPatch, context: MagicMock, *, samples: np.ndarray | None, rows: int = 64
) -> dict:
    """The soundtrack side of the extend node, with the mel front end and VAE stubbed out."""
    seen: dict = {}

    monkeypatch.setattr(
        ltx2_extend_conditioning,
        "extract_audio_pcm",
        lambda _path, **_kwargs: None if samples is None else (samples, 16000),
    )

    def fake_encode(_audio_vae, _vocoder, waveform, *, sample_rate):
        seen["waveform"] = waveform.clone()
        seen["sample_rate"] = sample_rate
        # One row per index, so a trim is visible in the values rather than only in the count.
        return torch.arange(rows, dtype=torch.float32).reshape(1, rows, 1).expand(1, rows, 128).contiguous()

    monkeypatch.setattr(ltx2_extend_conditioning, "encode_audio_latents", fake_encode)

    saved: list[torch.Tensor] = []

    def save(tensor):
        saved.append(tensor)
        return f"saved_{len(saved)}"

    context.tensors.save.side_effect = save
    seen["saved"] = saved
    return seen


def _extend_with_audio(**kwargs) -> LTX2ExtendConditioningInvocation:
    return _extend_conditioning(
        width=768,
        height=512,
        audio_vae=VAEField(vae=_identifier("audio_vae")),
        vocoder=LTX2VocoderField(vocoder=_identifier("vocoder")),
        **kwargs,
    )


def test_an_extension_holds_the_closing_sound_of_the_clip_it_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    """The join crossfades the held frames out of both halves. The picture survives that because it
    is held on both sides; the soundtrack only does if it is held too. Left generated, the blend
    fades invented audio in against the source's real audio and the new soundtrack starts an overlap
    early -- audible as a seam at the junction."""
    monkeypatch.setattr(
        ltx2_extend_conditioning,
        "read_canvas_frames",
        lambda _path, **kwargs: _canvas_clip(
            [np.zeros((512, 768, 3), dtype=np.uint8) for _ in range(kwargs["cap"])], SOURCE_CLIP_FRAMES
        ),
    )
    context = _ltx2_vae_context()
    _stub_video_vae(context, latent_frames=3, height=512, width=768)
    # Ten seconds of sound, each sample naming its own index.
    samples = np.arange(160000, dtype=np.float32).reshape(1, -1)
    seen = _stub_extend_audio(monkeypatch, context, samples=samples)

    output = _extend_with_audio(context_frames=17, fps=24.0).invoke(context)

    assert output.audio_conditioning is not None
    # The clip's END: the same 17 frames the picture holds, not its opening.
    wanted = round(17 / 24.0 * 16000)
    assert seen["waveform"].shape[1] == wanted
    assert seen["waveform"][0, 0].item() == pytest.approx(160000 - wanted)
    # Trimmed with the same function the denoise sizes the stream from, so the two cannot disagree.
    held = audio_latent_count(17, 24.0)
    assert output.audio_conditioning.num_audio_latents == held
    assert output.audio_conditioning.num_frames == 17
    assert output.audio_conditioning.fps == 24.0
    rows = seen["saved"][0]
    assert rows.shape[1] == held
    # The FIRST rows of the encode, which are the earliest of the held span -- the opening of the
    # continuation. Taking them from the end would hold the wrong instant.
    assert rows[0, :, 0].tolist() == list(range(held))


def test_the_held_sound_is_cut_from_the_span_the_picture_occupies_not_the_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decoded AAC track runs past the last frame by the codec's end padding -- 688 samples
    (14.3 ms) on the muxes the trim node emits -- and its content is front-aligned. Taking the tail
    of the untrimmed decode would hold audio that starts 14 ms after the picture does and ends in
    padding, and the join would blend it against the source's own correctly-aligned tail: comb
    filtering across the whole overlap, which is the artifact this is here to remove."""
    monkeypatch.setattr(
        ltx2_extend_conditioning,
        "read_canvas_frames",
        lambda _path, **kwargs: _canvas_clip(
            [np.zeros((512, 768, 3), dtype=np.uint8) for _ in range(kwargs["cap"])], SOURCE_CLIP_FRAMES
        ),
    )
    context = _ltx2_vae_context()
    _stub_video_vae(context, latent_frames=3, height=512, width=768)
    # The picture spans 240 frames at 24 fps = 160000 samples; the decode carries 688 more.
    picture = SOURCE_CLIP_FRAMES // 24 * 16000
    seen = _stub_extend_audio(monkeypatch, context, samples=np.arange(picture + 688, dtype=np.float32).reshape(1, -1))

    _extend_with_audio(context_frames=17, fps=24.0).invoke(context)

    wanted = round(17 / 24.0 * 16000)
    assert seen["waveform"].shape[1] == wanted
    # Ends where the PICTURE ends, not where the decode does.
    assert seen["waveform"][0, -1].item() == pytest.approx(picture - 1)
    assert seen["waveform"][0, 0].item() == pytest.approx(picture - wanted)


def test_an_extension_of_a_silent_clip_still_makes_its_own_soundtrack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing to hold is not a failure: the continuation invents the whole soundtrack, as it did
    before any of this existed."""
    monkeypatch.setattr(
        ltx2_extend_conditioning,
        "read_canvas_frames",
        lambda _path, **kwargs: _canvas_clip(
            [np.zeros((512, 768, 3), dtype=np.uint8) for _ in range(kwargs["cap"])], SOURCE_CLIP_FRAMES
        ),
    )
    context = _ltx2_vae_context()
    _stub_video_vae(context, latent_frames=3, height=512, width=768)
    _stub_extend_audio(monkeypatch, context, samples=None)

    assert _extend_with_audio(context_frames=17).invoke(context).audio_conditioning is None


def test_a_track_that_ends_early_holds_silence_anchored_to_the_picture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uploaded footage often has sound that stops before the picture does. Taking the tail of the
    SHORT track would hold an earlier instant than the frames held beside it -- a clip whose audio
    stops two seconds early would hold sound from two seconds before those frames, and the join
    would blend it against the source's real soundtrack. The held span is padded to the picture
    instead, so what is held is the silence the clip actually has there."""
    monkeypatch.setattr(
        ltx2_extend_conditioning,
        "read_canvas_frames",
        lambda _path, **kwargs: _canvas_clip(
            [np.zeros((512, 768, 3), dtype=np.uint8) for _ in range(kwargs["cap"])], SOURCE_CLIP_FRAMES
        ),
    )
    context = _ltx2_vae_context()
    _stub_video_vae(context, latent_frames=3, height=512, width=768)
    # Sound for the first half second of a ten-second clip, every sample naming its own index.
    seen = _stub_extend_audio(monkeypatch, context, samples=np.arange(1, 8001, dtype=np.float32).reshape(1, -1))

    output = _extend_with_audio(context_frames=17, fps=24.0).invoke(context)

    assert output.audio_conditioning is not None
    wanted = round(17 / 24.0 * 16000)
    assert seen["waveform"].shape[1] == wanted
    # Entirely silence: the picture's last 0.7s is long after the track ran out. Non-zero values
    # here would mean the slice had drifted back to where the sound actually was.
    assert seen["waveform"].abs().max().item() == 0.0


def test_an_extension_without_the_audio_models_holds_only_the_picture(monkeypatch: pytest.MonkeyPatch) -> None:
    """The audio models are optional inputs, so a graph that does not wire them must still run."""
    monkeypatch.setattr(
        ltx2_extend_conditioning,
        "read_canvas_frames",
        lambda _path, **kwargs: _canvas_clip(
            [np.zeros((512, 768, 3), dtype=np.uint8) for _ in range(kwargs["cap"])], SOURCE_CLIP_FRAMES
        ),
    )
    context = _ltx2_vae_context()
    _stub_video_vae(context, latent_frames=3, height=512, width=768)
    context.tensors.save.return_value = "extend_conditioning"

    output = _extend_conditioning(width=768, height=512, context_frames=17).invoke(context)

    assert output.audio_conditioning is None
    assert output.context_frames == 17


def test_the_lora_loader_wires_into_the_graph_the_panel_compiles() -> None:
    """The panel splices the LoRA collection loader between the model loader and both denoise
    passes. Every one of those edges has to be one the queue accepts, and `Graph.validate_self()`
    runs at enqueue -- after the models have loaded -- so a type mismatch here surfaces as a failed
    run rather than a failed compile. Numeric coercion is asymmetric and has bitten this stack
    twice, which is why the check is against the backend's own compatibility function.
    """
    from invokeai.app.invocations.baseinvocation import InvocationRegistry
    from invokeai.app.services.shared.graph import are_connection_types_compatible

    invocations = InvocationRegistry.get_invocations_map()
    loader = invocations["ltx2_lora_collection_loader"]
    model_loader_out = invocations["ltx2_model_loader"].get_output_annotation()
    denoise = invocations["ltx2_denoise"]

    edges = [
        (model_loader_out, "transformer", loader, "transformer"),
        (loader.get_output_annotation(), "transformer", denoise, "transformer"),
        (model_loader_out, "transformer", denoise, "transformer"),
    ]
    for source_cls, source_field, target_cls, target_field in edges:
        source = source_cls.model_fields[source_field].annotation
        target = target_cls.model_fields[target_field].annotation
        assert are_connection_types_compatible(source, target), (
            f"the queue refuses {source_cls.__name__}.{source_field} -> {target_cls.__name__}.{target_field}"
        )


def test_a_context_longer_than_the_model_can_generate_is_refused_before_any_decoding() -> None:
    """A tail read cannot stop early, so `context_frames` sizes a buffer of SOURCE-resolution frames
    held before any of them are fitted to the canvas. Unbounded, asking for 1001 frames of a 1080p
    clip reserves ~6 GiB of host memory and encodes all of it before the denoise rejects the anchor
    for exceeding the generation's length. Nothing past the model's own maximum is usable anyway."""
    with pytest.raises(ValidationError):
        _extend_conditioning(width=768, height=512, context_frames=LTX2_MAX_EXTEND_CONTEXT_FRAMES + 1)

    # The boundary itself is allowed, so the cap does not quietly exclude a usable length.
    assert _extend_conditioning(context_frames=LTX2_MAX_EXTEND_CONTEXT_FRAMES).context_frames == (
        LTX2_MAX_EXTEND_CONTEXT_FRAMES
    )


def test_an_extension_refuses_a_vae_from_another_architecture() -> None:
    with pytest.raises(ValueError, match="Expected an LTX-2"):
        _extend_conditioning().invoke(_ltx2_vae_context(BaseModelType.Wan))


def _full_video_field(num_frames: int = 9, **kwargs) -> LTX2FullVideoConditioningField:
    defaults = {
        "latents_name": "video",
        "width": 768,
        "height": 512,
        "num_frames": num_frames,
        "fps": 24.0,
        "source_video_name": "clip.mp4",
    }
    return LTX2FullVideoConditioningField(**{**defaults, **kwargs})


def test_video_to_audio_writes_the_users_own_frames_at_their_own_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """The picture was the given half, so decoding the held latents would hand back a cover-cropped
    VAE round trip of footage the user already has. Their frames are written instead -- at the
    resolution they shot, not the canvas the model needed."""
    source = [np.full((360, 640, 3), index, dtype=np.uint8) for index in range(12)]
    monkeypatch.setattr(ltx2_latents_to_video, "iter_video_frames", lambda *_a, **_k: iter(source))

    node = _latents_to_video(source_video=_full_video_field(num_frames=9))
    frames, height, width = node._source_frames(_context())
    written = list(frames)

    # Trimmed to what was generated, and untouched otherwise.
    assert (height, width) == (360, 640)
    assert len(written) == 9
    assert [int(frame[0, 0, 0]) for frame in written] == list(range(9))


def test_a_source_clip_that_shrank_since_it_was_encoded_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ltx2_latents_to_video,
        "iter_video_frames",
        lambda *_a, **_k: iter([np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(4)]),
    )

    frames, _height, _width = _latents_to_video(source_video=_full_video_field(num_frames=9))._source_frames(_context())

    with pytest.raises(ValueError, match="now decodes to 4 frame"):
        list(frames)


def test_the_source_clip_is_streamed_rather_than_collected(monkeypatch: pytest.MonkeyPatch) -> None:
    """These are full-resolution frames -- a 4K clip of 241 is 5.6 GiB -- and none of it is budgeted
    by the model cache. Only the first frame may be held, to report the geometry the writer needs."""
    decoded = 0

    def counting_frames(*_a, **_k):
        nonlocal decoded
        for index in range(12):
            decoded += 1
            yield np.full((360, 640, 3), index, dtype=np.uint8)

    monkeypatch.setattr(ltx2_latents_to_video, "iter_video_frames", counting_frames)
    node = _latents_to_video(source_video=_full_video_field(num_frames=9))
    frames, _height, _width = node._source_frames(_context())

    # Geometry is known without draining the clip.
    assert decoded == 1
    next(frames)
    assert decoded == 1  # the first frame was the one already read
    next(frames)
    assert decoded == 2


def test_the_held_latents_are_not_decoded_when_the_source_clip_is_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point is to skip that work: decoding is a second tiled VAE pass over a clip whose
    pixels are already on disk."""
    monkeypatch.setattr(
        ltx2_latents_to_video,
        "iter_video_frames",
        lambda *_a, **_k: iter([np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(9)]),
    )
    node = _latents_to_video(source_video=_full_video_field(num_frames=9))
    monkeypatch.setattr(
        type(node), "_decode_video", lambda *_a, **_k: pytest.fail("the held latents were decoded anyway")
    )
    monkeypatch.setattr(type(node), "_decode_audio_to_wav", lambda *_a, **_k: None)
    context = _context()
    context.util.is_canceled.return_value = False
    context.videos.save.side_effect = _StopAfterState

    with pytest.raises(_StopAfterState):
        node.invoke(context)


def test_a_last_frame_encode_wired_into_the_first_frame_slot_is_refused() -> None:
    """The two slots are different mechanisms -- one overwrites the opening grid tokens, the other
    appends -- so this slot cannot honour a non-zero index. Holding it at frame 0 anyway would do
    the opposite of what the field's own description promises."""
    node = _denoise(video_conditioning=_keyframe_field(-1))

    with pytest.raises(ValueError, match="Keyframe Conditioning instead"):
        node._load_image_latents(_context())


def test_the_reservation_is_sized_from_what_the_patches_weigh() -> None:
    """The reservation is subtracted from VRAM *before* any weight streams in, so it is not a
    ceiling -- every byte over-reserved is a byte of transformer that stays in host RAM and is
    streamed each forward. A per-LoRA constant big enough for the rank-450 distilled accelerator
    (8.3 GiB over 1660 layers) would charge a rank-16 style LoRA the same, and two of them would
    exceed a 24 GB card's whole budget on their own."""
    node_type = LTX2DenoiseInvocation
    rows, audio = 13728, 300
    small, large = 8 * 1024**2, 9 * 1024**3

    bare = node_type._estimate_working_memory(rows, audio)

    # Proportional to the patch, not to the count.
    assert node_type._estimate_working_memory(rows, audio, small) - bare == small + 1024**3
    assert node_type._estimate_working_memory(rows, audio, large) - bare == large + 1024**3
    # A small LoRA must not be charged like a large one -- the defect a flat constant has.
    assert (
        node_type._estimate_working_memory(rows, audio, large)
        > node_type._estimate_working_memory(rows, audio, small) + 8 * 1024**3
    )


def test_an_auto_schedule_with_a_lora_applied_says_so() -> None:
    """`auto` follows the transformer's VARIANT, which names the checkpoint -- and a LoRA does not
    change it. A step-distillation LoRA on a Dev checkpoint therefore resolves to the guided
    ~30-step schedule and samples the LoRA's 8 steps on it. The panel sets `schedule` explicitly to
    avoid this; a hand-built graph (or the shipped workflow, whose default is `auto`) cannot be told
    any other way, and the resulting clip looks like a broken model rather than a wiring mistake."""
    node = _denoise(
        num_frames=121,
        transformer=LTX2TransformerField(
            transformer=_identifier("transformer"),
            loras=[LoRAField(lora=_identifier("distilled-lora"), weight=1.0)],
            variant="ltx2_dev",
        ),
    )
    context = _context()

    assert node._resolve_distilled(context) is False
    assert "Distilled" in " ".join(str(call) for call in context.logger.info.call_args_list)

    # Silent where it would be wrong: on a distilled checkpoint `auto` already resolves correctly,
    # so telling the user to set Schedule to 'Distilled' would be advice against the truth.
    on_distilled = _context()
    distilled_node = _denoise(
        num_frames=121,
        transformer=LTX2TransformerField(
            transformer=_identifier("transformer"),
            loras=[LoRAField(lora=_identifier("style-lora"), weight=1.0)],
            variant="ltx2_distilled",
        ),
    )

    assert distilled_node._resolve_distilled(on_distilled) is True
    assert on_distilled.logger.info.call_count == 0

    # And silent when the schedule was named explicitly, or when there is nothing patched.
    quiet = _context()
    _denoise(num_frames=121, schedule="distilled")._resolve_distilled(quiet)

    assert quiet.logger.info.call_count == 0


def test_a_directly_patched_lora_reserves_nothing_for_itself() -> None:
    """Only the sidecar path leaves patch tensors on the device; the direct path returns each patch
    to the CPU as it applies it. The caller passes zero for that case, and charging for it anyway
    would cost a bf16 run -- the shipped Dev starter -- gigabytes of residency for nothing."""
    rows, audio = 13728, 300

    assert LTX2DenoiseInvocation._estimate_working_memory(
        rows, audio, 0
    ) == LTX2DenoiseInvocation._estimate_working_memory(rows, audio)


def test_the_reservation_covers_the_keyframe_rows_the_transformer_attends_over(monkeypatch) -> None:
    """A held keyframe adds rows to every forward -- 858 on top of 13728 at 1248x704 x121 -- and the
    working-memory fit is tightest exactly where running out costs the most."""
    import invokeai.app.invocations.ltx2.ltx2_denoise as denoise_module

    captured: list[int] = []
    # The reservation is arithmetic over the sequence, not a hardware question, but `invoke` asks the
    # chosen device for its dtype before it gets there. Left unpinned that reaches the real
    # accelerator: on a macOS runner MPS reports available and then fails every allocation.
    monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    node = _denoise(num_frames=121, keyframe_conditioning=_keyframe_field(-1), cfg_scale=1.0, audio_cfg_scale=1.0)
    monkeypatch.setattr(
        type(node),
        "_estimate_working_memory",
        lambda _self, sequence, _audio, _loras=0: captured.append(sequence) or (_ for _ in ()).throw(_StopAfterState()),
    )
    monkeypatch.setattr(denoise_module, "build_denoise_state", _denoise_state_stub(keyframe_rows=858))
    context = _context()
    context.tensors.load.return_value = torch.zeros(1, LTX2_LATENT_CHANNELS, 1, 22, 39)
    context.conditioning.load.return_value = SimpleNamespace(
        conditionings=[
            LTX2ConditioningInfo(
                video_embeds=torch.zeros(1, 4, 8),
                audio_embeds=torch.zeros(1, 4, 6),
                attention_mask=torch.ones(1, 4, dtype=torch.int64),
            )
        ]
    )

    with pytest.raises(_StopAfterState):
        node.invoke(context)

    assert captured == [13728 + 858]


def _denoise_state_stub(keyframe_rows: int):
    """A state whose sequence already carries the appended rows, as the real one would."""

    def build(**_kwargs):
        state = SimpleNamespace(
            video_latents=torch.zeros(1, 13728 + keyframe_rows, LTX2_LATENT_CHANNELS),
            audio_latents_count=126,
        )
        return state

    return build


class _StopAfterState(Exception):
    """Ends the invocation once the state has been built, before any model is loaded."""
