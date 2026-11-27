"""The decode nodes' up-front tiling rule: tile when an untiled decode would claim more than a share of the VAE's GPU.

Waiting for an out-of-memory error does not work where the driver pages into system memory instead of failing
(Windows ROCm always, NVIDIA's sysmem fallback by default), so the rule has to hold on the estimate alone.
"""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.backend.util.vae_working_memory import VAE_PRETILE_VRAM_FRACTION, should_pretile_vae_decode

TOTAL = 16 * 2**30


@pytest.mark.parametrize("fraction", [0.7, VAE_PRETILE_VRAM_FRACTION], ids=["anima", "image-and-video-vaes"])
def test_a_gpu_decode_flips_at_the_share_of_its_devices_memory(fraction):
    device = torch.device("cuda", 1)
    boundary = fraction * TOTAL
    with patch("torch.cuda.get_device_properties", return_value=MagicMock(total_memory=TOTAL)) as props:
        assert should_pretile_vae_decode(device, math.floor(boundary), fraction) is False
        assert should_pretile_vae_decode(device, math.ceil(boundary) + 1, fraction) is True
    props.assert_called_with(device)  # the VAE's own device, not whichever is current


@pytest.mark.parametrize("device_type", ["cpu", "mps"])
def test_a_decode_off_the_gpu_is_never_tiled_on_these_grounds(device_type):
    """A cpu_only VAE runs in system RAM, and MPS shares it: device totals are not the constraint there."""
    assert should_pretile_vae_decode(torch.device(device_type), 10**15, VAE_PRETILE_VRAM_FRACTION) is False


def test_1024px_stays_untiled_on_an_8gb_card_and_1536px_tiles_on_16gb():
    """The share is chosen so common sizes keep the exact single-pass decode: a Qwen-Image VAE decode at 1024px
    (~5.7 GiB) on an 8 GiB card stays untiled; a FLUX VAE decode at 1536px on ROCm (~15.8 GiB) tiles on 16 GiB."""
    with patch("torch.cuda.get_device_properties", return_value=MagicMock(total_memory=8 * 2**30)):
        assert not should_pretile_vae_decode(torch.device("cuda"), 1024 * 1024 * 2 * 2900, VAE_PRETILE_VRAM_FRACTION)
    with patch("torch.cuda.get_device_properties", return_value=MagicMock(total_memory=TOTAL)):
        assert should_pretile_vae_decode(torch.device("cuda"), 1536 * 1536 * 2 * 3600, VAE_PRETILE_VRAM_FRACTION)


def test_the_pretile_gate_uses_the_memory_the_device_will_keep_resident(monkeypatch):
    """The gate must measure against what the device keeps resident, not the card's nameplate total.

    This is the platform the feature was written for: the docstring justifies up-front tiling with
    "on Windows, drivers page an allocation that does not fit into system memory instead of failing
    it (always for ROCm)". Windows pages once the process passes its WDDM budget, so the gate has to
    compare against that budget rather than `total_memory`.

    A 16 GiB card whose budget is down to 12.5 GiB pages a 13.3 GiB decode (FLUX.1 at 1408px on
    MIOpen). No out-of-memory error is raised there, so the nodes' tiled retry never fires and the
    generation just crawls. Measured against the card total the decode looks fine (13.3 < 14.4 GiB),
    so it is not tiled.
    """
    total_bytes = 16 * 2**30
    budget_bytes = int(12.5 * 2**30)
    estimate = 1408 * 1408 * 2 * 3600  # 13.3 GiB: under 90% of the card, over the budget

    monkeypatch.setattr("torch.cuda.get_device_properties", lambda device: MagicMock(total_memory=total_bytes))
    # This process has allocated nothing yet, so its residency cannot lift the ceiling above the budget.
    monkeypatch.setattr("invokeai.backend.util.wddm.local_video_memory", lambda device: (budget_bytes, 0))
    monkeypatch.setattr("invokeai.backend.util.wddm.paged_bytes", lambda device: 0)

    assert estimate < VAE_PRETILE_VRAM_FRACTION * total_bytes, "test shape must look fine against the card total"
    assert estimate > budget_bytes, "test shape must exceed what Windows would keep resident"

    assert should_pretile_vae_decode(torch.device("cuda", 0), estimate) is True


def test_a_budget_below_our_own_residency_does_not_tile_a_decode_that_fits(monkeypatch):
    """Windows trims the budget as the process itself grows -- measured on an RX 9060 XT, 15.09 GiB until it passes
    about 12 of 16 GiB, then 7.62, i.e. below what it already holds. That is an instruction to trim itself, which the
    cache does for the reservation, so the ceiling is what it holds: comparing against the bare budget tiled a 7.0 GiB
    Z-Image decode against a 6.9 GiB line mid-session and changed output that had been pixel-identical."""
    total_bytes = 16 * 2**30
    estimate = 7 * 2**30

    monkeypatch.setattr("torch.cuda.get_device_properties", lambda device: MagicMock(total_memory=total_bytes))
    monkeypatch.setattr("invokeai.backend.util.wddm.local_video_memory", lambda device: (int(7.6 * 2**30), 12 * 2**30))
    monkeypatch.setattr("invokeai.backend.util.wddm.paged_bytes", lambda device: 0)  # alone: nothing paged out

    assert should_pretile_vae_decode(torch.device("cuda", 0), estimate) is False


def test_a_decode_beyond_both_the_budget_and_our_residency_is_tiled(monkeypatch):
    """The combined state -- something else holds part of the card and Invoke holds the rest -- is the one where the
    budget does carry the foreign pressure: evicting everything of ours reaches our own usage, not the whole card."""
    total_bytes = 16 * 2**30
    estimate = int(8.93 * 2**30)  # a Qwen-Image 1024px decode peak

    monkeypatch.setattr("torch.cuda.get_device_properties", lambda device: MagicMock(total_memory=total_bytes))
    monkeypatch.setattr(
        "invokeai.backend.util.wddm.local_video_memory", lambda device: (int(7.55 * 2**30), int(8.2 * 2**30))
    )
    monkeypatch.setattr("invokeai.backend.util.wddm.paged_bytes", lambda device: 0)

    assert estimate < VAE_PRETILE_VRAM_FRACTION * total_bytes, "the card total alone would let this through"
    assert should_pretile_vae_decode(torch.device("cuda", 0), estimate) is True


QWEN_1024_DECODE = int(8.93 * 2**30)


def _windows_rocm(monkeypatch, budget: float, usage: float | None, paged: float | None) -> None:
    monkeypatch.setattr("torch.cuda.get_device_properties", lambda device: MagicMock(total_memory=16 * 2**30))
    usage_bytes = None if usage is None else int(usage * 2**30)
    monkeypatch.setattr(
        "invokeai.backend.util.wddm.local_video_memory", lambda device: (int(budget * 2**30), usage_bytes)
    )
    paged_bytes = None if paged is None else int(paged * 2**30)
    monkeypatch.setattr("invokeai.backend.util.wddm.paged_bytes", lambda device: paged_bytes)


def test_usage_windows_has_paged_out_does_not_lift_the_ceiling(monkeypatch):
    """Next to an 8 GiB holder the budget steps down to its share of the card while `CurrentUsage` keeps counting what
    Windows paged out (measured: budget 7.58 GiB, usage 9.65 GiB, 7.70 GiB actually in VRAM). Only the resident part
    may override the budget; taking the whole usage priced this decode against a 12 GiB ceiling and let it page."""
    _windows_rocm(monkeypatch, budget=7.6, usage=12.0, paged=4.3)

    assert should_pretile_vae_decode(torch.device("cuda", 0), QWEN_1024_DECODE) is True


def test_usage_beyond_the_card_leaves_the_budget_in_charge(monkeypatch):
    """An over-committed process reports more usage than the card has; that says nothing about residency, and
    clamping it to the card size switched the correction off exactly while Windows was paging."""
    _windows_rocm(monkeypatch, budget=7.6, usage=None, paged=4.3)

    assert should_pretile_vae_decode(torch.device("cuda", 0), QWEN_1024_DECODE) is True


def test_residency_unknown_leaves_the_budget_in_charge(monkeypatch):
    """Without a paged-bytes reading the resident part is unknown, so a budget below our usage is taken as it is."""
    _windows_rocm(monkeypatch, budget=7.6, usage=12.0, paged=None)

    assert should_pretile_vae_decode(torch.device("cuda", 0), 7 * 2**30) is True


def test_the_card_total_stands_when_windows_does_not_answer(monkeypatch):
    """Off Windows ROCm, and whenever the driver cannot answer, the rule is the card's own size."""
    total_bytes = 16 * 2**30

    monkeypatch.setattr("torch.cuda.get_device_properties", lambda device: MagicMock(total_memory=total_bytes))
    monkeypatch.setattr("invokeai.backend.util.wddm.local_video_memory", lambda device: None)

    assert should_pretile_vae_decode(torch.device("cuda", 0), 7 * 2**30) is False
    assert should_pretile_vae_decode(torch.device("cuda", 0), 15 * 2**30) is True
