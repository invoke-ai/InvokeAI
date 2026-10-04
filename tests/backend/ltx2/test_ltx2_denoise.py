"""The LTX-2 denoising loop, over a transformer stub that records what it was asked."""

from types import SimpleNamespace

import pytest
import torch

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.backend.ltx2.constants import (
    LTX2_ANCESTRAL_NOISE_SEED_OFFSET,
    LTX2_AUDIO_LATENT_CHANNELS,
    LTX2_AUDIO_LATENT_MEL_BINS,
    LTX2_LATENT_CHANNELS,
    LTX2_REFINE_NOISE_SEED_OFFSET,
    LTX2_STAGE_2_NOISE_SCALE,
)
from invokeai.backend.ltx2.denoise import (
    _step_noise,
    build_denoise_state,
    build_refine_state,
    denoise,
    preview_latent_frame,
)
from invokeai.backend.ltx2.guidance import LTX2Guidance
from invokeai.backend.ltx2.packing import (
    audio_latent_count,
    pack_video_latents,
    unpack_video_latents,
    video_latent_shape,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import LTX2ConditioningInfo

WIDTH, HEIGHT, FRAMES = 64, 64, 9
LATENT = (2, 2, 2)  # frames, height, width at 32x/8x compression
ROWS = LATENT[0] * LATENT[1] * LATENT[2]
OFF = {"cfg_scale": 1.0, "audio_cfg_scale": 1.0, "stg_scale": 0.0, "modality_scale": 1.0, "rescale": 0.0}


def _conditioning(seed: int) -> LTX2ConditioningInfo:
    generator = torch.Generator().manual_seed(seed)
    return LTX2ConditioningInfo(
        video_embeds=torch.randn(1, 4, 8, generator=generator),
        audio_embeds=torch.randn(1, 4, 6, generator=generator),
        attention_mask=torch.ones(1, 4, dtype=torch.int64),
    )


class TransformerStub(torch.nn.Module):
    """Returns a velocity that denoises toward a fixed target, and records every call."""

    def __init__(self, target: torch.Tensor | None = None, audio_target: torch.Tensor | None = None) -> None:
        super().__init__()
        self.config = SimpleNamespace(timestep_scale_multiplier=1000, patch_size=1, patch_size_t=1)
        self.rope = SimpleNamespace(
            prepare_video_coords=lambda b, f, h, w, device, fps: torch.zeros(b, 3, f * h * w, 2, device=device)
        )
        self.audio_rope = SimpleNamespace(
            prepare_audio_coords=lambda b, n, device: torch.zeros(b, 1, n, 2, device=device)
        )
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity()])
        self.calls: list[dict] = []
        self.target = target
        self.audio_target = audio_target

    def forward(self, **kwargs):
        # Run the block the cancel hook is attached to, the way the real forward does.
        self.transformer_blocks[0](kwargs["hidden_states"])
        self.calls.append(kwargs)
        video, audio = kwargs["hidden_states"].float(), kwargs["audio_hidden_states"].float()
        # x0 = x - sigma * v, so this velocity predicts the target at any sigma.
        sigma = float(kwargs["sigma"][0]) / 1000.0
        target = self.target
        if target is not None and target.shape[1] < video.shape[1]:
            # Keyframe tokens ride on the end of the sequence. The target is extended with a value
            # they are NOT already at, so they drift unless something holds them -- extending with
            # their own value would give them zero velocity and quietly excuse the mask from
            # working at all.
            target = torch.cat([target, torch.full_like(video[:, target.shape[1] :], -3.0)], dim=1)
        video_velocity = torch.zeros_like(video) if target is None else (video - target) / max(sigma, 1e-6)
        audio_velocity = (
            torch.zeros_like(audio) if self.audio_target is None else (audio - self.audio_target) / max(sigma, 1e-6)
        )
        return video_velocity, audio_velocity


def _denoise(transformer, state, guidance, negative=None, **kwargs):
    return denoise(
        transformer=transformer,
        state=state,
        positive=_conditioning(1),
        negative=negative,
        guidance=guidance,
        fps=24.0,
        dtype=torch.float32,
        device=torch.device("cpu"),
        **kwargs,
    )


def _state(**kwargs):
    return build_denoise_state(
        num_frames=FRAMES, height=HEIGHT, width=WIDTH, fps=24.0, seed=7, distilled=False, num_steps=3, **kwargs
    )


def test_every_pass_runs_once_per_step_with_its_own_conditioning_and_flags() -> None:
    transformer = TransformerStub()
    guidance = LTX2Guidance(cfg_scale=3.0, audio_cfg_scale=7.0, stg_scale=1.0, modality_scale=3.0, rescale=0.0)
    negative = _conditioning(2)

    _denoise(transformer, _state(), guidance, negative=negative)

    assert len(transformer.calls) == 3 * 4
    first_step = transformer.calls[:4]
    assert [call["spatio_temporal_guidance_blocks"] for call in first_step] == [None, None, [28], None]
    assert [call["isolate_modalities"] for call in first_step] == [False, False, False, True]
    # Only the unconditional pass sees the negative prompt.
    assert torch.equal(first_step[1]["encoder_hidden_states"], negative.video_embeds)
    for index in (0, 2, 3):
        assert torch.equal(first_step[index]["encoder_hidden_states"], _conditioning(1).video_embeds)


def test_an_unguided_run_costs_one_forward_per_step() -> None:
    transformer = TransformerStub()
    _denoise(transformer, _state(), LTX2Guidance(**OFF))
    assert len(transformer.calls) == 3


def test_classifier_free_guidance_without_negative_conditioning_is_refused() -> None:
    with pytest.raises(ValueError, match="negative conditioning"):
        _denoise(TransformerStub(), _state(), LTX2Guidance())


def test_the_timestep_is_the_sigma_on_the_transformers_own_scale() -> None:
    transformer = TransformerStub()
    state = _state()
    _denoise(transformer, state, LTX2Guidance(**OFF))

    for index, call in enumerate(transformer.calls):
        assert float(call["timestep"][0]) == pytest.approx(float(state.sigmas[index]) * 1000)
        assert torch.equal(call["timestep"], call["audio_timestep"])
        assert torch.equal(call["timestep"], call["sigma"])


def test_a_conditioned_token_is_presented_as_clean_and_never_drifts() -> None:
    """The one mechanism behind every kind of conditioning: the anchor's timestep is zero at every
    forward, and its value is the encode it came from at the end of the run."""
    image_latents = torch.randn(1, LTX2_LATENT_CHANNELS, 1, LATENT[1], LATENT[2])
    state = _state(image_latents=image_latents)
    transformer = TransformerStub(target=torch.randn(1, ROWS, LTX2_LATENT_CHANNELS))

    video, _ = _denoise(transformer, state, LTX2Guidance(**OFF))

    anchor_rows = LATENT[1] * LATENT[2]
    for call in transformer.calls:
        assert call["timestep"].shape == (1, ROWS)
        assert torch.equal(call["timestep"][0, :anchor_rows], torch.zeros(anchor_rows))
        assert (call["timestep"][0, anchor_rows:] > 0).all()

    unpacked = unpack_video_latents(video, *LATENT)
    assert torch.allclose(unpacked[:, :, :1], image_latents, atol=1e-5)


def test_a_partial_conditioning_strength_holds_the_anchor_partway() -> None:
    image_latents = torch.randn(1, LTX2_LATENT_CHANNELS, 1, LATENT[1], LATENT[2])
    state = _state(image_latents=image_latents, conditioning_strength=0.5)
    transformer = TransformerStub(target=torch.randn(1, ROWS, LTX2_LATENT_CHANNELS))

    video, _ = _denoise(transformer, state, LTX2Guidance(**OFF))

    anchor_rows = LATENT[1] * LATENT[2]
    half = float(state.sigmas[0]) * 1000 * 0.5
    assert transformer.calls[0]["timestep"][0, :anchor_rows].tolist() == pytest.approx([half] * anchor_rows)
    assert not torch.allclose(unpack_video_latents(video, *LATENT)[:, :, :1], image_latents, atol=1e-3)


@pytest.mark.parametrize("strength", [0.0, -0.5, 1.5])
def test_a_conditioning_strength_outside_the_unit_interval_is_refused(strength: float) -> None:
    with pytest.raises(ValueError, match="strength"):
        _state(
            image_latents=torch.randn(1, LTX2_LATENT_CHANNELS, 1, LATENT[1], LATENT[2]), conditioning_strength=strength
        )


def test_an_image_conditioning_encoded_for_another_canvas_is_refused() -> None:
    with pytest.raises(ValueError, match="needs"):
        _state(image_latents=torch.randn(1, LTX2_LATENT_CHANNELS, 1, 4, 4))


def test_the_ancestral_branch_renoises_and_puts_the_anchor_back() -> None:
    """The distilled schedule injects noise at every step; without the restore the anchor would be
    noised along with everything else."""
    image_latents = torch.randn(1, LTX2_LATENT_CHANNELS, 1, LATENT[1], LATENT[2])
    state = build_denoise_state(
        num_frames=FRAMES,
        height=HEIGHT,
        width=WIDTH,
        fps=24.0,
        seed=7,
        distilled=True,
        num_steps=8,
        image_latents=image_latents,
    )
    assert state.eta == 1.0

    transformer = TransformerStub(target=torch.randn(1, ROWS, LTX2_LATENT_CHANNELS))
    video, _ = _denoise(transformer, state, LTX2Guidance(**OFF))

    # At every forward, not just at the end: the final step lands on x0 whatever came before it, so
    # an output-only check would pass with the anchor re-noised at each of the seven steps between.
    anchor_rows = LATENT[1] * LATENT[2]
    packed_anchor = pack_video_latents(image_latents)
    for call in transformer.calls:
        assert torch.allclose(call["hidden_states"].float()[:, :anchor_rows], packed_anchor, atol=1e-5)
    assert torch.allclose(unpack_video_latents(video, *LATENT)[:, :, :1], image_latents, atol=1e-5)


AUDIO_ROWS = audio_latent_count(FRAMES, 24.0)
AUDIO_WIDTH = LTX2_AUDIO_LATENT_CHANNELS * LTX2_AUDIO_LATENT_MEL_BINS


def _soundtrack(seed: int = 3) -> torch.Tensor:
    return torch.randn(1, AUDIO_ROWS, AUDIO_WIDTH, generator=torch.Generator().manual_seed(seed))


def _clip(seed: int = 4) -> torch.Tensor:
    return torch.randn(1, LTX2_LATENT_CHANNELS, *LATENT, generator=torch.Generator().manual_seed(seed))


def test_a_held_soundtrack_is_clean_at_every_forward_while_the_picture_is_sampled() -> None:
    """Audio-to-video is the first-frame mechanism with every audio row set instead of one video
    row. The model has to be HANDED the recording at every forward -- a soundtrack that only
    reappeared in the output would have been a different one all the way through the run, which is
    what the picture was actually scored against."""
    soundtrack = _soundtrack()
    # An audio target the model would pull toward if the rows were not held, so "unchanged" is a
    # statement about the mask rather than about a stub that predicts no motion.
    transformer = TransformerStub(
        target=torch.randn(1, ROWS, LTX2_LATENT_CHANNELS), audio_target=torch.zeros(1, AUDIO_ROWS, AUDIO_WIDTH)
    )

    video, audio = _denoise(transformer, _state(frozen_audio_latents=soundtrack), LTX2Guidance(**OFF))

    for call in transformer.calls:
        assert call["audio_timestep"].shape == (1, AUDIO_ROWS)
        assert torch.equal(call["audio_timestep"][0], torch.zeros(AUDIO_ROWS))
        assert torch.allclose(call["audio_hidden_states"].float(), soundtrack, atol=1e-5)
        # The picture is what is being generated, so its rows are noised as usual.
        assert (call["timestep"] > 0).all()

    assert torch.allclose(audio, soundtrack, atol=1e-5)
    assert not torch.allclose(video, torch.zeros_like(video))


def test_a_held_picture_is_clean_at_every_forward_while_the_soundtrack_is_sampled() -> None:
    clip = _clip()
    packed = pack_video_latents(clip)
    # The soundtrack is what is being generated here, so the stub has to actually move it --
    # otherwise "the audio was sampled" is a statement about a stub that predicts no motion.
    transformer = TransformerStub(
        target=torch.zeros(1, ROWS, LTX2_LATENT_CHANNELS),
        audio_target=torch.full((1, AUDIO_ROWS, AUDIO_WIDTH), 5.0),
    )

    video, audio = _denoise(transformer, _state(frozen_video_latents=clip), LTX2Guidance(**OFF))

    for call in transformer.calls:
        assert call["timestep"].shape == (1, ROWS)
        assert torch.equal(call["timestep"][0], torch.zeros(ROWS))
        assert torch.allclose(call["hidden_states"].float(), packed, atol=1e-5)
        assert (call["audio_timestep"] > 0).all()

    assert torch.allclose(unpack_video_latents(video, *LATENT), clip, atol=1e-5)
    assert torch.allclose(audio, torch.full((1, AUDIO_ROWS, AUDIO_WIDTH), 5.0), atol=1e-4)


def test_the_ancestral_branch_puts_a_held_soundtrack_back() -> None:
    """The distilled schedule re-noises every row at every step; without the restore the held
    soundtrack would drift away from the recording it was encoded from."""
    soundtrack = _soundtrack()
    state = build_denoise_state(
        num_frames=FRAMES,
        height=HEIGHT,
        width=WIDTH,
        fps=24.0,
        seed=7,
        distilled=True,
        num_steps=8,
        frozen_audio_latents=soundtrack,
    )

    transformer = TransformerStub(audio_target=torch.zeros(1, AUDIO_ROWS, AUDIO_WIDTH))
    _, audio = _denoise(transformer, state, LTX2Guidance(**OFF))

    # Every forward, not just the output: the last step lands on x0 whatever happened before it,
    # so an output-only check would pass with the recording re-noised at every step in between.
    for call in transformer.calls:
        assert torch.allclose(call["audio_hidden_states"].float(), soundtrack, atol=1e-5)
    assert torch.allclose(audio, soundtrack, atol=1e-5)


def test_a_first_frame_still_anchors_a_soundtrack_conditioned_run() -> None:
    """The two hold different streams, so they compose: a picture that starts from a given frame
    and follows a given soundtrack."""
    image_latents = torch.randn(1, LTX2_LATENT_CHANNELS, 1, LATENT[1], LATENT[2])
    state = _state(image_latents=image_latents, frozen_audio_latents=_soundtrack())

    video, audio = _denoise(
        TransformerStub(target=torch.randn(1, ROWS, LTX2_LATENT_CHANNELS)), state, LTX2Guidance(**OFF)
    )

    assert torch.allclose(unpack_video_latents(video, *LATENT)[:, :, :1], image_latents, atol=1e-5)
    assert torch.allclose(audio, state.clean_audio_latents, atol=1e-5)


def test_holding_both_modalities_leaves_nothing_to_generate_and_is_refused() -> None:
    with pytest.raises(ValueError, match="nothing for the model to generate"):
        _state(frozen_audio_latents=_soundtrack(), frozen_video_latents=_clip())


def test_a_first_frame_beside_a_whole_clip_conditioning_is_refused() -> None:
    """Both write frame 0. Silently overwriting one with the other would drop a picture the graph
    paid an encode for."""
    with pytest.raises(ValueError, match="first frame cannot be combined"):
        _state(
            image_latents=torch.randn(1, LTX2_LATENT_CHANNELS, 1, LATENT[1], LATENT[2]), frozen_video_latents=_clip()
        )


def test_a_soundtrack_of_the_wrong_length_names_the_frame_count_it_implies() -> None:
    """The frame count must be derived from the soundtrack; a mismatch means the graph set it
    separately, and the message has to say which way to fix it."""
    with pytest.raises(ValueError, match="Derive the frame count from the soundtrack"):
        _state(frozen_audio_latents=torch.randn(1, AUDIO_ROWS + 1, AUDIO_WIDTH))


def test_a_conditioning_clip_encoded_for_another_canvas_is_refused() -> None:
    with pytest.raises(ValueError, match="conditioning clip is"):
        _state(frozen_video_latents=torch.randn(1, LTX2_LATENT_CHANNELS, LATENT[0], LATENT[1] + 1, LATENT[2]))


def _keyframe(seed: int = 8) -> torch.Tensor:
    return torch.randn(1, LTX2_LATENT_CHANNELS, 1, LATENT[1], LATENT[2], generator=torch.Generator().manual_seed(seed))


def test_a_keyframe_rides_on_the_end_of_the_sequence_and_is_trimmed_off_the_result() -> None:
    """A last frame cannot overwrite grid tokens the way the first frame does -- the generation
    needs that position too -- so it is appended, and the model sees a longer sequence than the clip
    it returns. If the extra rows survived into the output, unpacking would reshape garbage."""
    keyframe = _keyframe()
    state = _state(keyframe_latents=keyframe, keyframe_latent_index=LATENT[0] - 1)
    transformer = TransformerStub(target=torch.randn(1, ROWS, LTX2_LATENT_CHANNELS))
    extra = LATENT[1] * LATENT[2]

    assert state.keyframe_tokens == extra
    assert state.video_latents.shape[1] == ROWS + extra

    video, _ = _denoise(transformer, state, LTX2Guidance(**OFF))

    # What the model was handed, versus what came back.
    for call in transformer.calls:
        assert call["hidden_states"].shape[1] == ROWS + extra
        assert call["video_coords"].shape[2] == ROWS + extra
    assert video.shape[1] == ROWS
    unpack_video_latents(video, *LATENT)  # would raise if the trim were wrong


def test_a_keyframe_is_presented_clean_at_every_forward_and_keeps_its_value() -> None:
    keyframe = _keyframe()
    state = _state(keyframe_latents=keyframe, keyframe_latent_index=LATENT[0] - 1)
    transformer = TransformerStub(target=torch.randn(1, ROWS, LTX2_LATENT_CHANNELS))
    packed = pack_video_latents(keyframe)

    _denoise(transformer, state, LTX2Guidance(**OFF))

    for call in transformer.calls:
        held = call["hidden_states"].float()[:, ROWS:]
        assert torch.allclose(held, packed, atol=1e-5)
        assert torch.equal(call["timestep"][0, ROWS:], torch.zeros(state.keyframe_tokens))
        assert (call["timestep"][0, :ROWS] > 0).all()


def test_a_keyframes_coordinates_place_it_at_its_own_instant_not_at_frame_zero() -> None:
    """The coordinates are the only thing saying *when* the held frame is; without them the model
    reads it as more of frame 0 and interpolates toward it from the start."""
    index = LATENT[0] - 1
    state = _state(keyframe_latents=_keyframe(), keyframe_latent_index=index)
    transformer = TransformerStub()

    _denoise(transformer, state, LTX2Guidance(**OFF))

    coords = transformer.calls[0]["video_coords"]
    keyframe_start = coords[0, 0, ROWS:, 0]
    # (index - 1) * 8 + 1 pixel frames in, expressed in seconds at the run's own frame rate.
    assert torch.allclose(keyframe_start, torch.full_like(keyframe_start, ((index - 1) * 8 + 1) / 24.0))


def test_a_first_frame_and_a_keyframe_compose_into_first_to_last_interpolation() -> None:
    """The two use the same mask at different ends of it: one overwrites grid tokens, one appends.
    Both have to survive the run, or interpolation silently becomes plain image-to-video."""
    image_latents = torch.randn(1, LTX2_LATENT_CHANNELS, 1, LATENT[1], LATENT[2])
    keyframe = _keyframe()
    state = _state(image_latents=image_latents, keyframe_latents=keyframe, keyframe_latent_index=LATENT[0] - 1)
    transformer = TransformerStub(target=torch.randn(1, ROWS, LTX2_LATENT_CHANNELS))

    video, _ = _denoise(transformer, state, LTX2Guidance(**OFF))

    anchor_rows = LATENT[1] * LATENT[2]
    for call in transformer.calls:
        assert torch.equal(call["timestep"][0, :anchor_rows], torch.zeros(anchor_rows))
        assert torch.equal(call["timestep"][0, ROWS:], torch.zeros(state.keyframe_tokens))
        assert (call["timestep"][0, anchor_rows:ROWS] > 0).all()
    assert torch.allclose(unpack_video_latents(video, *LATENT)[:, :, :1], image_latents, atol=1e-5)


def test_a_partially_held_keyframe_is_noised_to_its_share_of_the_step() -> None:
    state = _state(keyframe_latents=_keyframe(), keyframe_latent_index=LATENT[0] - 1, keyframe_strength=0.5)
    transformer = TransformerStub()

    _denoise(transformer, state, LTX2Guidance(**OFF))

    half = float(state.sigmas[0]) * 1000 * 0.5
    held = transformer.calls[0]["timestep"][0, ROWS:]
    assert held.tolist() == pytest.approx([half] * state.keyframe_tokens)


@pytest.mark.parametrize("index", [0, -1, LATENT[0], LATENT[0] + 5])
def test_a_keyframe_outside_the_clip_or_at_frame_zero_is_refused(index: int) -> None:
    """Index 0 is the first frame's job, and an index past the end names a moment the clip does
    not contain -- both are graph mistakes that would otherwise be a shape failure much later."""
    with pytest.raises(ValueError, match="keyframe"):
        _state(keyframe_latents=_keyframe(), keyframe_latent_index=index)


def test_a_keyframe_encoded_for_another_canvas_is_refused() -> None:
    with pytest.raises(ValueError, match="keyframe was encoded"):
        _state(
            keyframe_latents=torch.randn(1, LTX2_LATENT_CHANNELS, 1, LATENT[1] + 1, LATENT[2]),
            keyframe_latent_index=LATENT[0] - 1,
        )


@pytest.mark.parametrize("distilled", [False, True])
def test_adding_a_keyframe_does_not_reshuffle_the_generation(distilled: bool) -> None:
    """The ancestral branch draws fresh noise every step from one shared stream, sized from the
    sequence -- which the keyframe rows are part of. Drawing for them would shift the audio noise
    and, from the next step on, the picture too, so adding a last frame would not change how a clip
    ends, it would change the clip. The distilled checkpoint is the only schedule that takes that
    branch, which is why it needs its own case here."""
    keyframe = _keyframe()
    common = {
        "num_frames": FRAMES,
        "height": HEIGHT,
        "width": WIDTH,
        "fps": 24.0,
        "seed": 7,
        "distilled": distilled,
        "num_steps": 3,
    }
    plain = build_denoise_state(**common)
    with_keyframe = build_denoise_state(**common, keyframe_latents=keyframe, keyframe_latent_index=LATENT[0] - 1)

    assert with_keyframe.eta == (1.0 if distilled else 0.0)

    generators = [torch.Generator(device="cpu").manual_seed(state.noise_seed) for state in (plain, with_keyframe)]
    for _step in range(3):
        plain_noise = _step_noise(plain, plain.video_latents, plain.audio_latents, generators[0], torch.device("cpu"))
        keyed_noise = _step_noise(
            with_keyframe, with_keyframe.video_latents, with_keyframe.audio_latents, generators[1], torch.device("cpu")
        )
        if not distilled:
            assert plain_noise == (None, None)
            continue
        assert torch.equal(keyed_noise[0][:, :ROWS], plain_noise[0])
        assert torch.equal(keyed_noise[1], plain_noise[1])


def test_a_keyframe_beside_a_whole_clip_conditioning_is_refused() -> None:
    """The frozen branch replaces the entire sequence, taking the appended rows with it and leaving
    the keyframe's coordinates and token count pointing at tokens that are gone."""
    with pytest.raises(ValueError, match="keyframe cannot be combined"):
        _state(
            keyframe_latents=_keyframe(),
            keyframe_latent_index=LATENT[0] - 1,
            frozen_video_latents=torch.randn(1, LTX2_LATENT_CHANNELS, *LATENT),
        )


def test_adding_a_keyframe_does_not_reshuffle_the_noise_the_rest_of_the_clip_would_have_had() -> None:
    """Its noise is drawn last, so the same seed keeps the same generation and a user adding a last
    frame sees their clip end differently rather than change entirely."""
    plain = _state()
    with_keyframe = _state(keyframe_latents=_keyframe(), keyframe_latent_index=LATENT[0] - 1)

    assert torch.equal(with_keyframe.video_latents[:, :ROWS], plain.video_latents)
    assert torch.equal(with_keyframe.audio_latents, plain.audio_latents)


def test_the_refine_pass_is_given_the_keyframe_again_or_it_would_end_somewhere_else() -> None:
    """Appended tokens do not go through the upsampler, and the refine pass re-noises every token
    it does have -- so a last frame held only in stage one is gone by stage two, and a two-stage
    run would quietly mean something different by "last frame" than a single-stage one."""
    keyframe = _keyframe()
    state = build_refine_state(
        video_latents=torch.randn(1, LTX2_LATENT_CHANNELS, *LATENT),
        audio_latents=torch.randn(1, audio_latent_count(FRAMES, 24.0), 128),
        num_frames=FRAMES,
        height=HEIGHT,
        width=WIDTH,
        fps=24.0,
        seed=7,
        distilled=False,
        num_steps=2,
        noise_scale=LTX2_STAGE_2_NOISE_SCALE,
        keyframe_latents=keyframe,
        keyframe_latent_index=LATENT[0] - 1,
    )
    transformer = TransformerStub(target=torch.randn(1, ROWS, LTX2_LATENT_CHANNELS))

    assert state.keyframe_tokens == LATENT[1] * LATENT[2]

    video, _ = _denoise(transformer, state, LTX2Guidance(**OFF))

    packed = pack_video_latents(keyframe)
    for call in transformer.calls:
        assert torch.allclose(call["hidden_states"].float()[:, ROWS:], packed, atol=1e-5)
    assert video.shape[1] == ROWS


@pytest.mark.parametrize("held", [1, 2])
def test_a_leading_anchor_may_span_several_latent_frames_for_an_extension(held: int) -> None:
    """A still frame says where a clip starts but nothing about how it was moving, so an extension
    seeded from one stalls or lurches at the join. Several latent frames of the source carry its
    motion in, held by the same mask over more of the front of the grid."""
    anchor = torch.randn(1, LTX2_LATENT_CHANNELS, held, LATENT[1], LATENT[2])
    state = _state(image_latents=anchor)
    transformer = TransformerStub(target=torch.randn(1, ROWS, LTX2_LATENT_CHANNELS))

    held_rows = held * LATENT[1] * LATENT[2]

    assert state.conditioning_mask is not None
    assert torch.equal(state.conditioning_mask[0, :held_rows], torch.ones(held_rows))
    assert torch.equal(state.conditioning_mask[0, held_rows:], torch.zeros(ROWS - held_rows))

    video, _ = _denoise(transformer, state, LTX2Guidance(**OFF))

    for call in transformer.calls:
        assert torch.equal(call["timestep"][0, :held_rows], torch.zeros(held_rows))
        assert (call["timestep"][0, held_rows:] > 0).all()
    assert torch.allclose(unpack_video_latents(video, *LATENT)[:, :, :held], anchor, atol=1e-5)


def test_an_anchor_longer_than_the_clip_is_refused() -> None:
    with pytest.raises(ValueError, match="image conditioning was encoded"):
        _state(image_latents=torch.randn(1, LTX2_LATENT_CHANNELS, LATENT[0] + 1, LATENT[1], LATENT[2]))


def _audio_prefix(rows: int, seed: int = 11) -> torch.Tensor:
    return torch.randn(1, rows, AUDIO_WIDTH, generator=torch.Generator().manual_seed(seed))


def test_holding_the_soundtracks_opening_leaves_the_rest_to_be_generated() -> None:
    """A continuation replays the source's last moments and the join crossfades exactly those out of
    both halves. The picture survives that because it is held; the soundtrack has to be held over
    the same span or the blend fades invented audio in against the source's real audio, and the new
    soundtrack audibly starts one overlap early."""
    held = 4
    prefix = _audio_prefix(held)
    state = _state(audio_prefix_latents=prefix)
    transformer = TransformerStub(audio_target=torch.full((1, AUDIO_ROWS, AUDIO_WIDTH), 5.0))

    _, audio = _denoise(transformer, state, LTX2Guidance(**OFF))

    for call in transformer.calls:
        # The opening is clean at every forward; everything after it is noised as usual.
        assert torch.equal(call["audio_timestep"][0, :held], torch.zeros(held))
        assert (call["audio_timestep"][0, held:] > 0).all()
        assert torch.allclose(call["audio_hidden_states"].float()[:, :held], prefix, atol=1e-5)

    assert torch.allclose(audio[:, :held], prefix, atol=1e-5)
    # And the rest was genuinely generated, not held.
    assert torch.allclose(audio[:, held:], torch.full((1, AUDIO_ROWS - held, AUDIO_WIDTH), 5.0), atol=1e-4)


def test_the_refine_pass_holds_the_soundtracks_opening_again() -> None:
    """Stage two re-noises every row, so an opening held only in stage one is gone by the join."""
    held = 4
    prefix = _audio_prefix(held)
    state = build_refine_state(
        video_latents=torch.randn(1, LTX2_LATENT_CHANNELS, *LATENT),
        audio_latents=torch.randn(1, audio_latent_count(FRAMES, 24.0), AUDIO_WIDTH),
        num_frames=FRAMES,
        height=HEIGHT,
        width=WIDTH,
        fps=24.0,
        seed=7,
        distilled=False,
        num_steps=2,
        noise_scale=LTX2_STAGE_2_NOISE_SCALE,
        audio_prefix_latents=prefix,
    )
    transformer = TransformerStub(audio_target=torch.full((1, AUDIO_ROWS, AUDIO_WIDTH), 5.0))

    _, audio = _denoise(transformer, state, LTX2Guidance(**OFF))

    assert torch.allclose(audio[:, :held], prefix, atol=1e-5)


def test_holding_the_whole_soundtrack_and_its_opening_at_once_is_refused() -> None:
    with pytest.raises(ValueError, match="held whole and held at its opening"):
        _state(frozen_audio_latents=_soundtrack(), audio_prefix_latents=_audio_prefix(4))


@pytest.mark.parametrize(
    "shape",
    [
        (1, 0, AUDIO_WIDTH),
        (1, AUDIO_ROWS + 1, AUDIO_WIDTH),
        (1, 4, AUDIO_WIDTH + 1),
        # Saved tensors are whatever the workflow pointed at. A prefix of the wrong rank must be
        # named like the others rather than raising a bare IndexError on its missing dimension.
        (AUDIO_ROWS,),
        (1, AUDIO_ROWS),
    ],
)
def test_a_held_opening_that_does_not_fit_the_soundtrack_is_refused(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="held opening of the soundtrack"):
        _state(audio_prefix_latents=torch.zeros(shape))


def test_a_cancel_stops_the_run_inside_a_forward() -> None:
    """A step is a whole transformer forward, so a cancel polled between steps would leave the GPU
    busy for the rest of it; the hook fires per block instead."""
    transformer = TransformerStub()
    with pytest.raises(CanceledException):
        _denoise(transformer, _state(), LTX2Guidance(**OFF), is_canceled=lambda: True)
    assert transformer.calls == []


def test_the_preview_frame_is_the_middle_frame_of_the_unpacked_clip() -> None:
    """The preview slices the packed rows instead of unpacking the clip, which is only valid
    because the rows are frame-major; the expectation is the unpacked tensor it replaces."""
    latents = torch.randn(1, LTX2_LATENT_CHANNELS, 5, 3, 4)
    state = build_denoise_state(num_frames=33, height=96, width=128, fps=24.0, seed=1, distilled=False, num_steps=1)
    packed = pack_video_latents(latents)

    assert (state.latent_frames, state.latent_height, state.latent_width) == (5, 3, 4)
    assert torch.equal(preview_latent_frame(packed, state), latents[:, :, 2])


def test_the_step_callback_reports_progress_and_a_previewable_frame() -> None:
    reported: list[tuple[int, int, tuple[int, ...]]] = []
    state = _state()

    def callback(step: int, total: int, video_x0: torch.Tensor) -> None:
        reported.append((step, total, tuple(preview_latent_frame(video_x0, state).shape)))

    _denoise(TransformerStub(), state, LTX2Guidance(**OFF), step_callback=callback)
    assert reported == [(i, 3, (1, LTX2_LATENT_CHANNELS, LATENT[1], LATENT[2])) for i in (1, 2, 3)]


def test_the_run_is_reproducible_from_its_seed_and_differs_without_it() -> None:
    same = [_denoise(TransformerStub(), _state(), LTX2Guidance(**OFF))[0] for _ in range(2)]
    assert torch.equal(*same)

    other = build_denoise_state(
        num_frames=FRAMES, height=HEIGHT, width=WIDTH, fps=24.0, seed=8, distilled=False, num_steps=3
    )
    assert not torch.equal(_denoise(TransformerStub(), other, LTX2Guidance(**OFF))[0], same[0])


def _refine_inputs(width: int = 1792, height: int = 1024, num_frames: int = 121, fps: float = 24.0):
    # Deliberately not a standard normal: stage one's output is a denoised clip, and latents that
    # were already N(0, 1) would make "the noise this was mixed with" indistinguishable from the
    # latents themselves -- an assertion on its distribution would then hold with no mixing at all.
    frames, latent_height, latent_width = video_latent_shape(num_frames, height, width)
    video = torch.randn(1, LTX2_LATENT_CHANNELS, frames, latent_height, latent_width) * 4.0 + 7.0
    audio = torch.randn(1, audio_latent_count(num_frames, fps), LTX2_LATENT_CHANNELS) * 4.0 + 7.0
    return video, audio


@pytest.mark.parametrize("distilled", [True, False])
def test_the_refine_pass_starts_as_the_forward_process_at_its_first_level(distilled: bool) -> None:
    """A partial schedule expects a sample at its first level, not a clean one: both modalities are
    taken back up the same rectified flow the step walks down."""
    video, audio = _refine_inputs()
    state = build_refine_state(
        video_latents=video,
        audio_latents=audio,
        num_frames=121,
        height=1024,
        width=1792,
        fps=24.0,
        seed=4,
        distilled=distilled,
        num_steps=12,
        noise_scale=LTX2_STAGE_2_NOISE_SCALE,
    )
    sigma = float(state.sigmas[0])

    # Independent of the implementation's draw order: the noise is whatever is left once the clean
    # part is removed, and it has to be a standard normal of the right shape and scale.
    for name, mixed, clean in (
        ("video", state.video_latents, pack_video_latents(video)),
        # Audio is the half of this the docstring argues hardest for: the transformer reads one pair
        # of timesteps for both streams, so clean audio beside sigma-0.91 video would misstate where
        # the audio sits on the trajectory. A pass-through here would leave every other assertion in
        # this file green.
        ("audio", state.audio_latents, audio),
    ):
        noise = (mixed - (1 - sigma) * clean) / sigma
        assert noise.shape == mixed.shape, name
        assert abs(float(noise.std()) - 1.0) < 0.05, f"{name} noise is not unit-variance"
        assert abs(float(noise.mean())) < 0.05, f"{name} noise is not zero-mean"
    # The schedule is float32 and 0.909375 rounds up in it, so the level that truncation keeps
    # can exceed the requested one by an ulp; the comparison that matters is made in float32.
    assert sigma == pytest.approx(LTX2_STAGE_2_NOISE_SCALE, abs=1e-6) or sigma < LTX2_STAGE_2_NOISE_SCALE
    # Nothing is anchored: a first frame is already resolved into what is being refined.
    assert state.conditioning_mask is None
    assert state.clean_video_latents is None


def test_a_refine_pass_is_reproducible_from_its_seed() -> None:
    video, audio = _refine_inputs()
    kwargs = {
        "audio_latents": audio,
        "distilled": True,
        "fps": 24.0,
        "height": 1024,
        "noise_scale": LTX2_STAGE_2_NOISE_SCALE,
        "num_frames": 121,
        "num_steps": 8,
        "video_latents": video,
        "width": 1792,
    }
    first = build_refine_state(seed=11, **kwargs)
    again = build_refine_state(seed=11, **kwargs)
    other = build_refine_state(seed=12, **kwargs)

    assert torch.equal(first.video_latents, again.video_latents)
    assert torch.equal(first.audio_latents, again.audio_latents)
    assert not torch.equal(first.video_latents, other.video_latents)


def test_latents_from_the_wrong_canvas_are_refused_naming_both_shapes() -> None:
    """The upscaler doubles a latent grid exactly, so a mismatch here means the stages were planned
    at canvases that do not line up -- silently reshaping would denoise the wrong geometry."""
    video, audio = _refine_inputs(width=1792, height=1024)

    with pytest.raises(ValueError, match="1248x704"):
        build_refine_state(
            video_latents=video,
            audio_latents=audio,
            num_frames=121,
            height=704,
            width=1248,
            fps=24.0,
            seed=1,
            distilled=True,
            num_steps=8,
            noise_scale=LTX2_STAGE_2_NOISE_SCALE,
        )


def test_audio_latents_from_a_different_clip_length_are_refused() -> None:
    video, _ = _refine_inputs()
    _, mismatched = _refine_inputs(num_frames=49)

    with pytest.raises(ValueError, match="one fps and frame count"):
        build_refine_state(
            video_latents=video,
            audio_latents=mismatched,
            num_frames=121,
            height=1024,
            width=1792,
            fps=24.0,
            seed=1,
            distilled=True,
            num_steps=8,
            noise_scale=LTX2_STAGE_2_NOISE_SCALE,
        )


def test_the_refine_pass_draws_from_a_different_stream_than_the_base_pass() -> None:
    """The forward process assumes the noise mixed in is independent of what it is mixed into. Both
    stages seed from the request's seed, so without an offset the refine's draw would begin with
    exactly the values the base pass's own initial noise came from -- the numbers the clip was grown
    out of, mixed back into it. The module already carries that argument for the ancestral loop."""
    video, audio = _refine_inputs(width=896, height=512)
    base = build_denoise_state(num_frames=121, height=512, width=896, fps=24.0, seed=7, distilled=True, num_steps=8)
    refine = build_refine_state(
        video_latents=video,
        audio_latents=audio,
        num_frames=121,
        height=512,
        width=896,
        fps=24.0,
        seed=7,
        distilled=True,
        num_steps=8,
        noise_scale=LTX2_STAGE_2_NOISE_SCALE,
    )
    sigma = float(refine.sigmas[0])
    drawn = (refine.video_latents - (1 - sigma) * pack_video_latents(video)) / sigma

    # Same shape here (no upscale in this fixture), so a shared stream would be an exact match.
    assert drawn.shape == base.video_latents.shape
    assert not torch.allclose(drawn, base.video_latents, atol=1e-4)
    # All four streams in a two-stage run must be distinct, not merely the two initial ones: an
    # offset equal to the ancestral one would make the refine's re-noise repeat the base pass's
    # ancestral draws, which is the collision the ancestral offset exists to prevent.
    assert len({7, 7 + LTX2_ANCESTRAL_NOISE_SEED_OFFSET, 7 + LTX2_REFINE_NOISE_SEED_OFFSET, refine.noise_seed}) == 4


def test_the_refine_pass_re_anchors_a_conditioned_first_frame() -> None:
    """The refine pass re-noises every token, frame 0 included, so at the released entry level about
    nine tenths of an anchored frame's signal is replaced. Nothing restores it without a mask, and
    the base pass's encode is half this canvas -- so two-stage image-to-video would regenerate the
    first frame from the prompt alone."""
    video, audio = _refine_inputs(width=896, height=512)
    _, latent_height, latent_width = video_latent_shape(121, 512, 896)
    anchor = torch.randn(1, LTX2_LATENT_CHANNELS, 1, latent_height, latent_width)
    common = {
        "audio_latents": audio,
        "distilled": True,
        "fps": 24.0,
        "height": 512,
        "noise_scale": LTX2_STAGE_2_NOISE_SCALE,
        "num_frames": 121,
        "num_steps": 8,
        "seed": 3,
        "video_latents": video,
        "width": 896,
    }
    anchored = build_refine_state(image_latents=anchor, **common)
    free = build_refine_state(**common)

    assert free.conditioning_mask is None and free.clean_video_latents is None
    assert anchored.conditioning_mask is not None and anchored.clean_video_latents is not None
    # Frame 0's tokens are held at the anchor; everything after it is free.
    per_frame = anchored.conditioning_mask[0].reshape(-1, latent_height * latent_width)
    assert torch.all(per_frame[0] == 1.0)
    assert torch.all(per_frame[1:] == 0.0)
    first_span = anchored.video_latents[0, : latent_height * latent_width]
    torch.testing.assert_close(first_span, anchored.clean_video_latents[0, : latent_height * latent_width])


def test_a_refine_anchor_encoded_at_the_base_canvas_is_refused_by_name() -> None:
    video, audio = _refine_inputs(width=896, height=512)
    _, base_h, base_w = video_latent_shape(121, 256, 448)

    with pytest.raises(ValueError, match="refine canvas"):
        build_refine_state(
            video_latents=video,
            audio_latents=audio,
            num_frames=121,
            height=512,
            width=896,
            fps=24.0,
            seed=3,
            distilled=True,
            num_steps=8,
            noise_scale=LTX2_STAGE_2_NOISE_SCALE,
            image_latents=torch.randn(1, LTX2_LATENT_CHANNELS, 1, base_h, base_w),
        )
