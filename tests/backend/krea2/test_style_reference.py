import pytest
import torch

from invokeai.backend.krea2.style_reference import (
    KREA2_NUM_BLOCKS,
    Krea2StyleReferenceMode,
    Krea2StyleReferenceSettings,
    Krea2StyleReferenceState,
    _adain_to_stats,
    _token_mean_std,
    apply_style_reference,
    build_rope_scale_vector,
    capture_style_reference,
    lerp_scales,
    parse_block_spec,
    resolve_effective_settings,
)

# Krea-2's real RoPE layout: (temporal, height, width), summing to the 128-dim head.
KREA2_AXES = (32, 48, 48)


def _state(image_seq_len: int = 6, axes: tuple[int, ...] = (2, 3, 3), **overrides) -> Krea2StyleReferenceState:
    # The test tensors use head_dim 8, so the axes have to sum to 8 (Krea-2's real layout is KREA2_AXES).
    return Krea2StyleReferenceState(
        settings=resolve_effective_settings(Krea2StyleReferenceSettings(**overrides)),
        image_seq_len=image_seq_len,
        axes_dims_rope=axes,
    )


# --- block spec ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("7-27", frozenset(range(7, 28))),
        ("5", frozenset({5})),
        ("7-9,3", frozenset({3, 7, 8, 9})),
        (" 7 - 9 ; 3 ", frozenset({3, 7, 8, 9})),
    ],
)
def test_parse_block_spec_accepts_ranges_lists_and_singletons(spec: str, expected: frozenset[int]) -> None:
    assert parse_block_spec(spec, KREA2_NUM_BLOCKS) == expected


@pytest.mark.parametrize("spec", ["", "   ", ",,"])
def test_parse_block_spec_rejects_specs_that_select_nothing(spec: str) -> None:
    with pytest.raises(ValueError, match="selects no blocks"):
        parse_block_spec(spec, KREA2_NUM_BLOCKS)


def test_parse_block_spec_rejects_blocks_the_transformer_does_not_have() -> None:
    # Upstream silently accepts out-of-range indices and then styles nothing. Failing here surfaces the
    # typo at graph time instead of as a mysteriously unstyled image.
    with pytest.raises(ValueError, match=r"selects blocks \[28\]"):
        parse_block_spec("7-28", KREA2_NUM_BLOCKS)


def test_parse_block_spec_rejects_a_reversed_range() -> None:
    with pytest.raises(ValueError, match="end is before start"):
        parse_block_spec("27-7", KREA2_NUM_BLOCKS)


# --- style_strength modulation -------------------------------------------------------------------


def test_style_strength_of_one_leaves_every_parameter_at_its_configured_value() -> None:
    settings = Krea2StyleReferenceSettings()
    effective = resolve_effective_settings(settings)

    assert effective.high_scale_start == pytest.approx(settings.high_scale_start)
    assert effective.low_scale_end == pytest.approx(settings.low_scale_end)
    assert effective.adain_strength == pytest.approx(settings.adain_strength)
    assert effective.attention_mix == pytest.approx(1.0)


def test_style_strength_of_zero_neutralizes_every_modulated_parameter() -> None:
    # Not just the attention mix: upstream also pulls the frequency scales back to 1.0 (i.e. no scaling)
    # and zeroes the AdaIN, so a strength of 0 is a true bypass.
    effective = resolve_effective_settings(Krea2StyleReferenceSettings(style_strength=0.0))

    assert effective.high_scale_start == pytest.approx(1.0)
    assert effective.low_scale_end == pytest.approx(1.0)
    assert effective.adain_strength == pytest.approx(0.0)
    assert effective.attention_mix == pytest.approx(0.0)


def test_style_strength_saturates_each_factor_at_its_own_ceiling() -> None:
    # attention_mix clamps at 1.0, the AdaIN multiplier at 1.25 and the high-scale multiplier at 1.5.
    effective = resolve_effective_settings(Krea2StyleReferenceSettings(style_strength=2.0, adain_strength=0.5))

    assert effective.attention_mix == pytest.approx(1.0)
    assert effective.adain_strength == pytest.approx(0.5 * 1.25)
    assert effective.high_scale_start == pytest.approx(1.0 + (1.04 - 1.0) * 1.5)
    # low_scale_end is *not* capped.
    assert effective.low_scale_end == pytest.approx(1.0 + (1.10 - 1.0) * 2.0)


# --- RoPE frequency scale vector -----------------------------------------------------------------


def test_rope_scale_vector_length_matches_the_head_dim() -> None:
    vector = build_rope_scale_vector(KREA2_AXES, 1.04, 1.10, 2.5, torch.device("cpu"), torch.float32)
    assert vector.shape == (sum(KREA2_AXES),)


def test_rope_scale_vector_holds_the_temporal_axis_flat_at_the_low_scale() -> None:
    # Every Krea-2 token sits at t=0, so axis 0's rotation is the identity and has no frequency structure
    # to shape.
    vector = build_rope_scale_vector(KREA2_AXES, 1.04, 1.10, 2.5, torch.device("cpu"), torch.float32)
    assert torch.allclose(vector[: KREA2_AXES[0]], torch.full((KREA2_AXES[0],), 1.10))


def test_rope_scale_vector_runs_from_high_to_low_across_each_spatial_axis() -> None:
    high, low = 1.04, 1.10
    vector = build_rope_scale_vector(KREA2_AXES, high, low, 2.5, torch.device("cpu"), torch.float32)

    height_axis = vector[KREA2_AXES[0] : KREA2_AXES[0] + KREA2_AXES[1]]
    width_axis = vector[KREA2_AXES[0] + KREA2_AXES[1] :]
    for axis in (height_axis, width_axis):
        # get_1d_rotary_pos_embed puts the highest frequency first, so the curve starts at `high`.
        assert axis[0].item() == pytest.approx(high)
        assert axis[-1].item() == pytest.approx(low)


def test_rope_scale_vector_repeats_each_frequency_across_its_pair() -> None:
    # Krea2RotaryPosEmbed uses repeat_interleave_real=True, so each frequency occupies two consecutive
    # dims. The scale vector has to line up with that or it shifts the bands it is meant to attenuate.
    vector = build_rope_scale_vector(KREA2_AXES, 1.04, 0.0, 2.5, torch.device("cpu"), torch.float32)
    assert torch.equal(vector[0::2], vector[1::2])


def test_rope_scale_vector_kills_the_highest_bands_when_high_scale_reaches_zero() -> None:
    # This is the mechanism that stops reference *content* leaking in as the schedule progresses.
    vector = build_rope_scale_vector(KREA2_AXES, 0.0, 1.0, 2.5, torch.device("cpu"), torch.float32)
    assert vector[KREA2_AXES[0]].item() == pytest.approx(0.0)


def test_lerp_scales_walks_from_the_start_values_to_the_end_values() -> None:
    settings = resolve_effective_settings(Krea2StyleReferenceSettings())

    assert lerp_scales(settings, 0.0) == pytest.approx((settings.high_scale_start, settings.low_scale_start))
    assert lerp_scales(settings, 1.0) == pytest.approx((settings.high_scale_end, settings.low_scale_end))


# --- AdaIN ---------------------------------------------------------------------------------------


def test_adain_at_full_strength_adopts_the_reference_statistics() -> None:
    torch.manual_seed(0)
    target = torch.randn(1, 4, 32, 8) * 3.0 + 5.0
    style = torch.randn(1, 4, 32, 8) * 0.5 - 2.0
    style_mean, style_std = _token_mean_std(style)

    result = _adain_to_stats(target, style_mean, style_std, 1.0)
    result_mean, result_std = _token_mean_std(result)

    assert torch.allclose(result_mean, style_mean, atol=1e-5)
    assert torch.allclose(result_std, style_std, atol=1e-5)


def test_adain_at_zero_strength_is_the_identity() -> None:
    torch.manual_seed(0)
    target = torch.randn(1, 4, 32, 8)
    style_mean, style_std = _token_mean_std(torch.randn(1, 4, 32, 8))
    assert torch.equal(_adain_to_stats(target, style_mean, style_std, 0.0), target)


# --- capture / inject ----------------------------------------------------------------------------


def test_capture_before_head_expansion_matches_capture_after() -> None:
    """The load-bearing test for the 4x memory saving.

    Capturing at 12 KV heads instead of 48 is only sound because ``repeat_interleave`` duplicates whole
    heads, so per-``(head, dim)`` token statistics are identical within a group. If that ever stops
    holding, the captured cache silently diverges from upstream.
    """
    torch.manual_seed(0)
    query = torch.randn(1, 8, 10, 8)
    key = torch.randn(1, 2, 10, 8)
    value = torch.randn(1, 2, 10, 8)
    repeats = 4

    pre = _state(image_seq_len=6)
    pre.begin_capture()
    capture_style_reference(pre, 0, query, key, value)

    post = _state(image_seq_len=6)
    post.begin_capture()
    capture_style_reference(
        post, 0, query, key.repeat_interleave(repeats, dim=1), value.repeat_interleave(repeats, dim=1)
    )

    pre_cache, post_cache = pre.get(0), post.get(0)
    assert torch.equal(pre_cache.reference_key.repeat_interleave(repeats, dim=1), post_cache.reference_key)
    assert torch.equal(pre_cache.reference_value.repeat_interleave(repeats, dim=1), post_cache.reference_value)
    assert torch.allclose(pre_cache.key_mean.repeat_interleave(repeats, dim=1), post_cache.key_mean)
    assert torch.allclose(pre_cache.key_std.repeat_interleave(repeats, dim=1), post_cache.key_std)
    assert torch.equal(pre_cache.query_mean, post_cache.query_mean)


def test_capture_only_keeps_the_image_tokens() -> None:
    torch.manual_seed(0)
    query = torch.randn(1, 8, 10, 8)
    key = torch.randn(1, 2, 10, 8)
    value = torch.randn(1, 2, 10, 8)

    state = _state(image_seq_len=6)
    state.begin_capture()
    capture_style_reference(state, 3, query, key, value)

    cache = state.get(3)
    assert cache.reference_key.shape == (1, 2, 6, 8)
    # Krea-2 concatenates [text, image], so the image tokens are the tail of the sequence.
    assert torch.equal(cache.reference_key, key[:, :, 4:, :])


def test_capture_rejects_an_image_seq_len_longer_than_the_sequence() -> None:
    state = _state(image_seq_len=99)
    state.begin_capture()
    with pytest.raises(ValueError, match="exceeds the transformer sequence length"):
        capture_style_reference(state, 0, torch.randn(1, 8, 10, 8), torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8))


def test_inject_before_capture_fails_loudly() -> None:
    state = _state()
    with pytest.raises(RuntimeError, match="before any reference pass was captured"):
        state.begin_inject(0.0)


def test_inject_on_an_uncaptured_block_fails_loudly() -> None:
    state = _state(image_seq_len=6)
    state.begin_capture()
    capture_style_reference(state, 7, torch.randn(1, 8, 10, 8), torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8))
    state.begin_inject(0.0)

    with pytest.raises(RuntimeError, match="block 8 was not captured"):
        apply_style_reference(state, 8, torch.randn(1, 8, 10, 8), torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8))


def test_inject_appends_the_reference_image_tokens_to_the_keys_and_values() -> None:
    torch.manual_seed(0)
    state = _state(image_seq_len=6)
    state.begin_capture()
    capture_style_reference(state, 0, torch.randn(1, 8, 10, 8), torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8))
    state.begin_inject(0.5)

    injection = apply_style_reference(
        state, 0, torch.randn(1, 8, 10, 8), torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8)
    )

    assert injection.query.shape == (1, 8, 10, 8)
    assert injection.key.shape == (1, 2, 16, 8)
    assert injection.value.shape == (1, 2, 16, 8)


def test_inject_with_the_default_value_mode_passes_the_reference_values_through_untouched() -> None:
    # value_mode="target_adain_plus_ref" with ref_value_mix=1.0 discards the AdaIN'd blend entirely. This
    # is why value_adain_strength has no effect at the recommended settings.
    torch.manual_seed(0)
    reference_value = torch.randn(1, 2, 6, 8)
    state = _state(image_seq_len=6)
    state.begin_capture()
    capture_style_reference(state, 0, torch.randn(1, 8, 6, 8), torch.randn(1, 2, 6, 8), reference_value)
    state.begin_inject(0.0)

    injection = apply_style_reference(
        state, 0, torch.randn(1, 8, 6, 8), torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 8)
    )

    assert torch.equal(injection.value[:, :, 6:, :], reference_value)


def test_inject_does_not_mutate_the_caller_tensors() -> None:
    # The AdaIN writes into the image-token slice; doing that in place would corrupt the value tensor the
    # processor still needs for the unstyled branch.
    torch.manual_seed(0)
    state = _state(image_seq_len=6)
    state.begin_capture()
    capture_style_reference(state, 0, torch.randn(1, 8, 10, 8), torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8))
    state.begin_inject(0.0)

    query = torch.randn(1, 8, 10, 8)
    key = torch.randn(1, 2, 10, 8)
    value = torch.randn(1, 2, 10, 8)
    original = (query.clone(), key.clone(), value.clone())
    apply_style_reference(state, 0, query, key, value)

    assert torch.equal(query, original[0])
    assert torch.equal(key, original[1])
    assert torch.equal(value, original[2])


# --- shared state lifecycle ----------------------------------------------------------------------


def test_pad_attention_mask_widens_the_key_axis_and_allows_the_reference() -> None:
    state = _state(image_seq_len=6)
    mask = torch.zeros(10, 10, dtype=torch.bool)

    padded = state.pad_attention_mask(mask)

    assert padded is not None
    assert padded.shape == (10, 16)
    assert torch.equal(padded[:, :10], mask)
    assert bool(padded[:, 10:].all())


def test_pad_attention_mask_is_cached_per_source_mask() -> None:
    # Rebuilding this for each of the ~21 styled blocks would be a real cost at high resolution.
    state = _state(image_seq_len=6)
    mask = torch.zeros(10, 10, dtype=torch.bool)
    assert state.pad_attention_mask(mask) is state.pad_attention_mask(mask)


def test_pad_attention_mask_passes_none_through() -> None:
    assert _state().pad_attention_mask(None) is None


def test_clear_releases_the_captured_cache() -> None:
    # The processors stay installed on the *cached* transformer, so anything retained here survives the
    # invocation. At 2560x1440 that would be ~1.7 GiB of leaked VRAM.
    state = _state(image_seq_len=6)
    state.begin_capture()
    capture_style_reference(state, 0, torch.randn(1, 8, 10, 8), torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8))
    state.begin_inject(0.0)
    state.pad_attention_mask(torch.zeros(10, 10, dtype=torch.bool))

    state.clear()

    assert state.mode is Krea2StyleReferenceMode.OFF
    assert state._cache == {}
    assert state._scale_vector is None
    assert state._padded_masks == []
