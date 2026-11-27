"""The LTX-2.5 loaders over the released files (slow lane: needs the weights on disk).

Every component's pinned config and key map were derived from the released files' headers; this is
the check that the real tensors land in the real diffusers modules and that the Gemma-4 text tower
built by ``Gemma4TextModel`` runs. Skipped unless the ``DeepBeepMeep/LTX-2`` snapshot is in the
Hugging Face cache (``INVOKEAI_LTX2_SNAPSHOT`` overrides the path).
"""

import gc
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from invokeai.backend.ltx2 import checkpoint_layout as layout
from invokeai.backend.model_manager.configs.gemma4_encoder import Gemma4Encoder_Gemma4Encoder_LTX2_Config
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_LTX2_Config, Main_Diffusers_LTX2_Config
from invokeai.backend.model_manager.load.model_loaders.ltx2 import (
    LTX2CheckpointModel,
    LTX2FolderModel,
    LTX2Gemma4EncoderModel,
)
from invokeai.backend.model_manager.taxonomy import LTX2VariantType, SubModelType
from invokeai.backend.util.devices import TorchDevice

pytestmark = pytest.mark.slow


def _snapshot() -> Path | None:
    override = os.environ.get("INVOKEAI_LTX2_SNAPSHOT")
    if override:
        return Path(override)
    hub = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub" / "models--DeepBeepMeep--LTX-2"
    )
    snapshots = sorted((hub / "snapshots").glob("*")) if hub.exists() else []
    return snapshots[-1] if snapshots else None


SNAPSHOT = _snapshot()
COMPONENTS = {
    layout.ROLE_VIDEO_VAE: "ltx-2.5-22b_video_vae_bf16.safetensors",
    layout.ROLE_AUDIO_VAE: "ltx-2.5-22b_audio_vae_bf16.safetensors",
    layout.ROLE_VOCODER: "ltx-2.5-22b_vocoder_bf16.safetensors",
    layout.ROLE_TEXT_PROJECTION: "ltx-2.5-22b_text_embedding_projection_bf16.safetensors",
    layout.ROLE_VIDEO_CONNECTOR: "ltx-2.5-22b_video_embeddings_connector_bf16.safetensors",
    layout.ROLE_AUDIO_CONNECTOR: "ltx-2.5-22b_audio_embeddings_connector_bf16.safetensors",
    layout.ROLE_SPATIAL_UPSAMPLER: "ltx-2.5-spatial-upscaler-x2-1.0_bf16.safetensors",
    layout.ROLE_TEMPORAL_UPSAMPLER: "ltx-2.5-temporal-upscaler-x2-1.0_bf16.safetensors",
}
requires_weights = pytest.mark.skipif(
    SNAPSHOT is None or not all((SNAPSHOT / name).exists() for name in COMPONENTS.values()),
    reason="the LTX-2.5 release files are not in the Hugging Face cache",
)


def _loader(cls, device: torch.device | None = None):
    loader = object.__new__(cls)
    loader._ram_cache = SimpleNamespace(make_room=lambda _n: None)
    loader._logger = SimpleNamespace(info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None)
    loader._torch_device = device or torch.device("cpu")
    loader._apply_fp8_layerwise_casting = lambda model, _c, _s: model
    return loader


def _no_meta(model: torch.nn.Module) -> None:
    assert not any(p.is_meta for p in model.parameters()) and not any(b.is_meta for b in model.buffers())


@requires_weights
@pytest.mark.parametrize(
    ("submodel", "expected_class"),
    [
        (SubModelType.VAE, "AutoencoderKLLTX2Video"),
        (SubModelType.AudioVAE, "AutoencoderKLLTX2Audio"),
        (SubModelType.Vocoder, "LTX2VocoderWithBWE"),
        (SubModelType.Connectors, "LTX2TextConnectors"),
        (SubModelType.LatentUpsampler, "LTX2LatentUpsamplerModel"),
        (SubModelType.TemporalLatentUpsampler, "LTX2LatentUpsamplerModel"),
    ],
)
def test_every_released_component_lands_in_its_diffusers_module(submodel, expected_class) -> None:
    config = Main_Diffusers_LTX2_Config.model_construct(
        path=str(SNAPSHOT), components=dict(COMPONENTS), components_only=True, variant=LTX2VariantType.Dev
    )
    model = _loader(LTX2FolderModel)._load_model(config, submodel)
    assert type(model).__name__ == expected_class
    _no_meta(model)


@requires_weights
def test_the_int8_transformer_builds_with_its_layers_packed() -> None:
    path = SNAPSHOT / "ltx-2.5-22b-dev_diffusion_model_int8_convrot.safetensors"
    if not path.exists():
        pytest.skip("int8 dev transformer not downloaded")
    from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear

    config = Main_Checkpoint_LTX2_Config.model_construct(
        path=str(path), variant=LTX2VariantType.Dev, generation="2.5", fp8_storage=None
    )
    model = _loader(LTX2CheckpointModel)._load_model(config, SubModelType.Transformer)
    _no_meta(model)
    assert isinstance(model.transformer_blocks[0].attn1.to_q, Int8ConvrotLinear)
    assert model.config.use_keyframes_abs_pos_embedding is True
    assert len(model.transformer_blocks) == 48


@requires_weights
def test_the_gemma4_tower_runs_and_yields_all_forty_nine_hidden_states() -> None:
    root = SNAPSHOT / "gemma4-12b-ltx-v1"
    weight = "gemma4-12b-ltx-v1_int8_convrot.safetensors"
    if not (root / weight).exists():
        pytest.skip("int8 Gemma-4 encoder not downloaded")
    device = TorchDevice.choose_torch_device()
    if device.type == "cpu":
        pytest.skip("needs an accelerator for a 12B forward")
    config = Gemma4Encoder_Gemma4Encoder_LTX2_Config.model_construct(path=str(root), subfolder="", weight_file=weight)
    loader = _loader(LTX2Gemma4EncoderModel)
    tokenizer = loader._load_model(config, SubModelType.Tokenizer)
    model = loader._load_model(config, SubModelType.TextEncoder)
    _no_meta(model)

    model.to(device)
    tokenizer.padding_side = "left"
    batch = tokenizer(
        ["A cat on a windowsill, purring."], padding="max_length", max_length=64, truncation=True, return_tensors="pt"
    )
    with torch.inference_mode():
        out = model(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            output_hidden_states=True,
        )
    assert len(out.hidden_states) == 49
    stacked = torch.stack(out.hidden_states, dim=-1)
    assert stacked.shape == (1, 64, 3840, 49)
    assert torch.isfinite(stacked).all()
    # The tower is doing work: the last layer is not a rescaled copy of the embedding output.
    assert not torch.allclose(out.hidden_states[-1].float(), out.hidden_states[0].float(), atol=1.0)


@pytest.mark.skipif(
    SNAPSHOT is None or not (SNAPSHOT / "gemma4-12b-ltx-v1" / "tokenizer.json").exists(),
    reason="the Gemma-4 tokenizer is not in the Hugging Face cache",
)
def test_the_released_tokenizer_keeps_the_words_the_mistral_fix_would_shatter() -> None:
    """transformers advises the Mistral regex fix for this folder. Taking it prepends the Tekken
    split regex to Gemma's pre-tokenizer, which fragments words this vocabulary has whole -- every
    prompt would then encode differently from what the reference pipelines feed the tower. The
    synthetic folder in the loader suite pins the warnings; this pins the tokenization itself."""
    root = SNAPSHOT / "gemma4-12b-ltx-v1"
    config = Gemma4Encoder_Gemma4Encoder_LTX2_Config.model_construct(
        path=str(root), subfolder="", weight_file="gemma4-12b-ltx-v1_bf16.safetensors"
    )
    tokenizer = _loader(LTX2Gemma4EncoderModel)._load_model(config, SubModelType.Tokenizer)

    tokens = tokenizer.convert_ids_to_tokens(
        tokenizer("worst quality, inconsistent motion, blurry, jittery, distorted").input_ids
    )
    # Exactly the words the Mistral split shatters ("in"+"consistent", "bl"+"urry", "dist"+"orted").
    assert {"\u2581inconsistent", "\u2581blurry", "\u2581distorted"} <= set(tokens)


@requires_weights
def test_the_mirror_s_nvfp4_transformer_is_refused_for_naming_no_layer() -> None:
    """WanGP's nvfp4 repack carries no marker and no header entry for its 1176 packed layers, so the
    block-scale layout cannot be known; the nvfp4 reader refuses it by name. Pinned so a future
    release that does name its layers is noticed here, not by a user."""
    path = SNAPSHOT / "ltx-2.5-22b-distilled_diffusion_model_nvfp4.safetensors"
    if not path.exists():
        pytest.skip("nvfp4 distilled transformer not downloaded")
    config = Main_Checkpoint_LTX2_Config.model_construct(
        path=str(path), variant=LTX2VariantType.Distilled, generation="2.5", fp8_storage=None
    )
    with pytest.raises(ValueError, match="nvfp4"):
        _loader(LTX2CheckpointModel)._load_model(config, SubModelType.Transformer)


# --- Generation over the released weights ---------------------------------------------------------
#
# These run the real recipe end to end on an accelerator. They install the wide-head SDPA guard
# themselves: it is normally installed by application startup, which a test process never runs, and
# without it Gemma-4's 512-wide full-attention layers return wrong values on ROCm -- the tower's
# hidden states go non-finite part-way through the stack, and every generation that follows is NaN.

_PROMPT = "A ginger cat sits on a windowsill, purring, as rain patters on the glass."


def _accelerator() -> torch.device:
    device = TorchDevice.choose_torch_device()
    if device.type == "cpu":
        pytest.skip("needs an accelerator")
    from invokeai.backend.util.attention import install_rocm_sdpa_guard

    install_rocm_sdpa_guard()
    return device


def _folder_config() -> Main_Diffusers_LTX2_Config:
    return Main_Diffusers_LTX2_Config.model_construct(
        path=str(SNAPSHOT), components=dict(COMPONENTS), components_only=True, variant=LTX2VariantType.Dev
    )


def _encode_prompts(device: torch.device, prompts: list[str]):
    """The full two-stage prompt encode, one model resident at a time."""
    from invokeai.backend.ltx2.text_conditioning import apply_connectors, encode_hidden_states

    weight = "gemma4-12b-ltx-v1_int8_convrot.safetensors"
    root = SNAPSHOT / "gemma4-12b-ltx-v1"
    if not (root / weight).exists():
        pytest.skip("the int8 Gemma-4 encoder is not downloaded")

    config = Gemma4Encoder_Gemma4Encoder_LTX2_Config.model_construct(
        path=str(SNAPSHOT), subfolder="gemma4-12b-ltx-v1", weight_file=weight
    )
    loader = _loader(LTX2Gemma4EncoderModel, device)
    tokenizer = loader._load_model(config, SubModelType.Tokenizer)
    encoder = loader._load_model(config, SubModelType.TextEncoder).to(device).eval()
    states = [
        tuple(t.cpu() for t in encode_hidden_states(encoder, tokenizer, p, max_sequence_length=1024, device=device))
        for p in prompts
    ]
    del encoder
    gc.collect()
    TorchDevice.empty_cache()

    connectors = _loader(LTX2FolderModel, device)._load_model(_folder_config(), SubModelType.Connectors)
    connectors = connectors.to(device).eval()
    conditionings = [apply_connectors(connectors, *state, device=device) for state in states]
    del connectors
    gc.collect()
    TorchDevice.empty_cache()
    return states, conditionings


@requires_weights
def test_the_prompt_encode_stays_finite_at_the_padded_length_the_connectors_need() -> None:
    """Every prompt is padded to 1024 tokens, which is the length the connectors' registers are
    defined against. The tower's full-attention layers are 512 wide, and on a build whose fused
    attention kernels are wrong at that width the states go non-finite around layer 30 -- finite at
    a shorter padding, so only the real length catches it."""
    device = _accelerator()
    states, conditionings = _encode_prompts(device, [_PROMPT])

    hidden, mask = states[0]
    assert hidden.shape == (1, 1024, 3840 * 49)
    assert torch.isfinite(hidden).all()
    assert int(mask.sum()) < 1024

    conditioning = conditionings[0]
    assert conditioning.video_embeds.shape == (1, 1024, 4096)
    assert conditioning.audio_embeds.shape == (1, 1024, 2048)
    assert torch.isfinite(conditioning.video_embeds).all()
    assert torch.isfinite(conditioning.audio_embeds).all()


@requires_weights
def test_the_distilled_checkpoint_generates_a_clip_with_a_soundtrack() -> None:
    """The whole recipe: eight ancestral steps, then both decoders. A wrong sigma schedule, a
    mis-packed sequence or a mis-ordered guidance combine all land here as noise or NaN."""
    from invokeai.backend.ltx2.denoise import build_denoise_state, denoise
    from invokeai.backend.ltx2.guidance import LTX2Guidance
    from invokeai.backend.ltx2.packing import unpack_video_latents
    from invokeai.backend.ltx2.video_decoding import decode_audio_latents, decode_video_latents

    transformer_path = SNAPSHOT / "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
    if not transformer_path.exists():
        pytest.skip("the int8 distilled transformer is not downloaded")

    device = _accelerator()
    _, conditionings = _encode_prompts(device, [_PROMPT])

    width, height, num_frames = 512, 320, 9
    state = build_denoise_state(
        num_frames=num_frames, height=height, width=width, fps=24.0, seed=42, distilled=True, num_steps=8
    )
    assert state.eta == 1.0, "LTX-2.5 samples its distilled schedule ancestrally"

    transformer = _loader(LTX2CheckpointModel, device)._load_model(
        Main_Checkpoint_LTX2_Config.model_construct(
            path=str(transformer_path), variant=LTX2VariantType.Distilled, generation="2.5", fp8_storage=None
        ),
        SubModelType.Transformer,
    )
    transformer = transformer.to(device).eval()
    video, audio = denoise(
        transformer=transformer,
        state=state,
        positive=conditionings[0],
        negative=None,
        guidance=LTX2Guidance(cfg_scale=1.0, audio_cfg_scale=1.0, stg_scale=0.0, modality_scale=1.0, rescale=0.0),
        fps=24.0,
        dtype=torch.bfloat16,
        device=device,
    )
    del transformer
    gc.collect()
    TorchDevice.empty_cache()

    assert torch.isfinite(video).all() and torch.isfinite(audio).all()

    vae = _loader(LTX2FolderModel, device)._load_model(_folder_config(), SubModelType.VAE).to(device).eval()
    clip = decode_video_latents(
        vae,
        unpack_video_latents(video, state.latent_frames, state.latent_height, state.latent_width),
        tile_size=512,
        temporal_tile=16,
    )
    del vae
    gc.collect()
    TorchDevice.empty_cache()

    assert clip.shape == (3, num_frames, height, width)
    # A clip, not a flat field: the released model at this size returns a photographic image.
    assert 0.1 < float(clip.mean()) < 0.9
    assert float(clip.std()) > 0.1

    audio_vae = _loader(LTX2FolderModel, device)._load_model(_folder_config(), SubModelType.AudioVAE)
    vocoder = _loader(LTX2FolderModel, device)._load_model(_folder_config(), SubModelType.Vocoder)
    waveform = decode_audio_latents(audio_vae.to(device).eval(), vocoder.to(device).eval(), audio)

    assert waveform.shape[0] == 2
    # The audio VAE's causal decoder drops its first few mel frames, so the soundtrack comes back
    # a fraction of a second short of the clip and the latents-to-video node pads it out.
    nominal = num_frames / 24.0 * vocoder.config.output_sampling_rate
    assert 0.85 * nominal <= waveform.shape[1] <= nominal
    assert float(waveform.abs().max()) <= 1.0
    assert float(waveform.pow(2).mean().sqrt()) > 1e-3, "the soundtrack is silent"


@requires_weights
def test_an_image_conditioning_survives_the_round_trip_to_the_first_decoded_frame() -> None:
    """The conditioning mask is what every later conditioning feature is built on, so the anchor
    has to come back out of the decoder as the frame that went in."""
    from invokeai.backend.ltx2.image_conditioning import encode_image_latents, fit_to_canvas, recompress_h264
    from invokeai.backend.ltx2.video_decoding import decode_video_latents

    device = _accelerator()
    width, height = 512, 320
    source = fit_to_canvas(
        recompress_h264(Image.effect_mandelbrot((1024, 768), (-2.5, -1.5, 1.5, 1.5), 60).convert("RGB"), 18),
        height,
        width,
    )

    vae = _loader(LTX2FolderModel, device)._load_model(_folder_config(), SubModelType.VAE).to(device).eval()
    latents = encode_image_latents(vae, source, device=device)
    assert latents.shape == (1, 128, 1, height // 32, width // 32)

    decoded = decode_video_latents(vae, latents, tile_size=512, temporal_tile=16)
    assert decoded.shape == (3, 1, height, width)

    original = torch.from_numpy(np.asarray(source, dtype=np.float32) / 255.0).permute(2, 0, 1)
    psnr = 10 * torch.log10(1.0 / torch.mean((decoded[:, 0] - original) ** 2))
    assert float(psnr) > 25.0, f"the VAE round trip lost the conditioning frame ({float(psnr):.1f} dB)"


@requires_weights
def test_a_recording_encodes_to_the_audio_rate_the_transformer_reads_and_survives_the_round_trip() -> None:
    """The conditioning front end against the released weights.

    The release ships no standalone log-mel module: the transform is lifted off the vocoder's own
    bandwidth-extension stage and re-strided to the audio VAE's hop. Two things have to hold for
    that to be the right transform, and neither can be checked without the real filters.

    First the *rate*: LTX-2 reads 25 audio latents per second everywhere else, and at the vocoder's
    own hop (half the VAE's) this would produce 50 -- a clip conditioned at double speed.

    Second the *domain*: generation decodes a latent to a mel and hands that mel to the vocoder, so
    the mel this module computes has to be the one the VAE decoded. That is checked here directly,
    by running the released chain forwards and this module backwards over the same signal: any
    difference of filterbank, window or rate convention shows up as the two mels disagreeing, where
    a shape check alone would pass. A mel basis left at its constructed zeros, for instance, has
    exactly the right shape and represents silence.
    """
    from invokeai.backend.ltx2.audio_conditioning import build_mel_transform, encode_audio_latents
    from invokeai.backend.ltx2.constants import LTX2_AUDIO_LATENTS_PER_SECOND
    from invokeai.backend.ltx2.packing import denormalize_audio_latents, unpack_audio_latents
    from invokeai.backend.util.audio_resample import resample_sinc

    device = _accelerator()
    audio_vae = _loader(LTX2FolderModel, device)._load_model(_folder_config(), SubModelType.AudioVAE)
    vocoder = _loader(LTX2FolderModel, device)._load_model(_folder_config(), SubModelType.Vocoder)
    audio_vae = audio_vae.to(device).eval()
    vocoder = vocoder.to(device).eval()

    vae_rate = int(audio_vae.config.sample_rate)
    hop = int(audio_vae.config.mel_hop_length)
    out_rate = int(vocoder.config.output_sampling_rate)
    # The filterbank is fitted for the rate the vocoder is fed, which is the VAE's own -- the
    # higher rate is what it synthesises. If these ever diverge, the transform is being applied to
    # a signal it was not fitted for and everything below is measuring the wrong thing.
    assert int(vocoder.config.input_sampling_rate) == vae_rate

    # A 3 Hz amplitude-modulated tone: the loudness contour a soundtrack conditions a picture on.
    seconds = 2.0
    time = torch.arange(int(vae_rate * seconds), dtype=torch.float32) / vae_rate
    tone = torch.sin(2 * torch.pi * 300 * time) * (0.5 + 0.5 * torch.sin(2 * torch.pi * 3 * time)) * 0.5
    latents = encode_audio_latents(audio_vae, vocoder, torch.stack([tone, tone]), sample_rate=vae_rate)

    assert latents.shape == (1, int(seconds * LTX2_AUDIO_LATENTS_PER_SECOND), 128)
    assert torch.isfinite(latents).all()
    # Denormalizing and unpacking has to land on the VAE's own 8-channel, 16-bin grid rather than a
    # transposed or half-length one.
    unpacked = unpack_audio_latents(
        denormalize_audio_latents(latents, audio_vae.latents_mean.cpu(), audio_vae.latents_std.cpu())
    )
    assert unpacked.shape == (1, 8, int(seconds * LTX2_AUDIO_LATENTS_PER_SECOND), 16)

    # Forwards through the release: a latent becomes the VAE's own mel, which the vocoder speaks.
    with torch.inference_mode():
        released = denormalize_audio_latents(
            torch.randn(1, 75, 128, generator=torch.Generator().manual_seed(5)).cumsum(1).div(8.0).to(device),
            audio_vae.latents_mean,
            audio_vae.latents_std,
        )
        vae_mel = audio_vae.decode(
            unpack_audio_latents(released).to(next(iter(audio_vae.parameters())).dtype), return_dict=False
        )[0].float()
        waveform = vocoder(vae_mel.to(next(iter(vocoder.parameters())).dtype))[0].float().cpu()

    # Backwards through this module, over that waveform.
    transform = build_mel_transform(vocoder, hop_length=hop)
    audio = resample_sinc(waveform, out_rate, vae_rate)
    padding = -audio.shape[-1] % hop
    audio = torch.nn.functional.pad(audio, (0, padding)) if padding else audio
    mine, *_ = transform(audio[None].to(device=device, dtype=transform.mel_basis.dtype).flatten(0, 1))
    mine = mine.unflatten(0, (1, audio.shape[0])).transpose(2, 3).float()

    assert mine.shape == vae_mel.shape, "the re-analysed mel is not the shape the VAE decodes to"

    frames = min(vae_mel.shape[2], mine.shape[2])
    pair = [side[..., :frames, :].flatten() for side in (vae_mel, mine)]
    centered = [side - side.mean() for side in pair]
    agreement = float((centered[0] * centered[1]).sum() / (centered[0].norm() * centered[1].norm()).clamp_min(1e-8))

    # The residual is the vocoder's own phase reconstruction and the 48 k -> 16 k resample, not a
    # difference of domain; a mismatched filterbank or rate lands far below this.
    assert agreement > 0.9, f"this module's mel is not the one the VAE decoded ({agreement:.3f})"

    del audio_vae, vocoder
    gc.collect()
    TorchDevice.empty_cache()
