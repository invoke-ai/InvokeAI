"""Which passes LTX-2 guidance runs, and how their predictions combine."""

import pytest
import torch

from invokeai.backend.ltx2.guidance import PASS_COND, PASS_MODALITY, PASS_STG, PASS_UNCOND, LTX2Guidance

OFF = {"cfg_scale": 1.0, "audio_cfg_scale": 1.0, "stg_scale": 0.0, "modality_scale": 1.0, "rescale": 0.0}


def test_all_guidance_off_costs_one_forward() -> None:
    assert LTX2Guidance(**OFF).passes == (PASS_COND,)
    assert not LTX2Guidance(**OFF).needs_negative_conditioning


def test_an_audio_only_cfg_scale_still_buys_the_unconditional_pass() -> None:
    """One forward produces both modalities, so a pass either modality wants is a pass for both."""
    guidance = LTX2Guidance(**{**OFF, "audio_cfg_scale": 7.0})
    assert guidance.passes == (PASS_COND, PASS_UNCOND)
    assert guidance.needs_negative_conditioning


def test_each_term_adds_exactly_one_pass() -> None:
    assert LTX2Guidance(**{**OFF, "stg_scale": 1.0}).passes == (PASS_COND, PASS_STG)
    assert LTX2Guidance(**{**OFF, "modality_scale": 3.0}).passes == (PASS_COND, PASS_MODALITY)
    assert LTX2Guidance().passes == (PASS_COND, PASS_UNCOND, PASS_STG, PASS_MODALITY)


def test_spatio_temporal_guidance_needs_blocks_to_perturb() -> None:
    """A scale with no block list would pay for a forward identical to the conditional one."""
    assert LTX2Guidance(**{**OFF, "stg_scale": 1.0}, stg_blocks=()).passes == (PASS_COND,)


def test_the_combine_is_the_reference_delta_formulation() -> None:
    torch.manual_seed(0)
    preds = {
        PASS_COND: torch.randn(1, 8, 4),
        PASS_UNCOND: torch.randn(1, 8, 4),
        PASS_STG: torch.randn(1, 8, 4),
        PASS_MODALITY: torch.randn(1, 8, 4),
    }
    guidance = LTX2Guidance(cfg_scale=3.0, audio_cfg_scale=7.0, stg_scale=1.0, modality_scale=3.0, rescale=0.0)

    cond = preds[PASS_COND]
    expected = (
        cond + 2.0 * (cond - preds[PASS_UNCOND]) + 1.0 * (cond - preds[PASS_STG]) + 2.0 * (cond - preds[PASS_MODALITY])
    )
    assert torch.allclose(guidance.combine_video(preds), expected, atol=1e-6)

    # The audio stream differs only in its CFG scale.
    audio_expected = expected + 4.0 * (cond - preds[PASS_UNCOND])
    assert torch.allclose(guidance.combine_audio(preds), audio_expected, atol=1e-6)


def test_an_inert_scale_contributes_nothing_when_the_other_modality_bought_the_pass() -> None:
    preds = {PASS_COND: torch.randn(1, 8, 4), PASS_UNCOND: torch.randn(1, 8, 4)}
    guidance = LTX2Guidance(**{**OFF, "audio_cfg_scale": 7.0})
    assert torch.equal(guidance.combine_video(preds), preds[PASS_COND])


def test_the_rescale_pulls_the_guided_prediction_back_toward_the_conditional_scale() -> None:
    """Guidance deltas inflate the prediction's spread; the rescale is what stops the clip blowing
    out. At rescale 1 the result carries the conditional prediction's standard deviation exactly."""
    torch.manual_seed(0)
    preds = {PASS_COND: torch.randn(1, 512, 16), PASS_UNCOND: torch.randn(1, 512, 16) * 0.2}

    unrescaled = LTX2Guidance(**{**OFF, "cfg_scale": 5.0}).combine_video(preds)
    rescaled = LTX2Guidance(**{**OFF, "cfg_scale": 5.0, "rescale": 1.0}).combine_video(preds)

    assert float(unrescaled.std()) > float(preds[PASS_COND].std())
    assert float(rescaled.std()) == pytest.approx(float(preds[PASS_COND].std()), rel=1e-5)


def test_an_unguided_step_returns_the_conditional_prediction_untouched() -> None:
    """What the distilled checkpoint runs on every step. Rescaling a prediction against its own
    standard deviation is a no-op, so it has to be skipped rather than computed."""
    preds = {PASS_COND: torch.randn(1, 8, 4)}
    guidance = LTX2Guidance(**{**OFF, "rescale": 0.7})
    assert guidance.combine_video(preds) is preds[PASS_COND]
    assert guidance.combine_audio(preds) is preds[PASS_COND]
