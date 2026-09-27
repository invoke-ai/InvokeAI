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
    it (always for ROCm)". Windows pages once the process passes its WDDM budget, and another GPU
    process lowers that budget within about a second -- which is exactly why this PR added
    `TorchDevice.cuda_mem_get_info` to cap the model cache's free figure by `wddm.video_memory_budget`.
    The tiling gate consults neither that budget nor free memory, only `total_memory`.

    A 16 GiB card whose budget is down to 12.5 GiB pages a 13.3 GiB decode (FLUX.1 at 1408px on
    MIOpen). No out-of-memory error is raised there, so the nodes' tiled retry never fires and the
    generation just crawls. Measured against the card total the decode looks fine (13.3 < 14.4 GiB),
    so it is not tiled. With no adapter-wide reading available this is the budget's own case; the
    tests below cover what the gate does when it can tell whose memory it is.
    """
    total_bytes = 16 * 2**30
    budget_bytes = int(12.5 * 2**30)
    estimate = 1408 * 1408 * 2 * 3600  # 13.3 GiB: under 90% of the card, over the budget

    monkeypatch.setattr("torch.cuda.get_device_properties", lambda device: MagicMock(total_memory=total_bytes))
    # torch cannot see the shortfall: on Windows ROCm its free figure is the device total minus this
    # process's own live allocations, and this process has allocated nothing yet.
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device: (total_bytes, total_bytes))
    # Patch both binding sites: `devices` imports the name directly, so a fix routing through
    # `TorchDevice.cuda_mem_get_info` and one calling `wddm.video_memory_budget` are both covered.
    monkeypatch.setattr("invokeai.backend.util.devices.video_memory_budget", lambda device: budget_bytes)
    monkeypatch.setattr("invokeai.backend.util.wddm.video_memory_budget", lambda device: budget_bytes)
    # Without this the gate reads the real card here, and any foreign usage below ~1.2 GiB flips the case.
    monkeypatch.setattr("invokeai.backend.util.wddm.other_process_local_bytes", lambda device: None)

    assert estimate < VAE_PRETILE_VRAM_FRACTION * total_bytes, "test shape must look fine against the card total"
    assert estimate > budget_bytes, "test shape must exceed what Windows would keep resident"

    assert should_pretile_vae_decode(torch.device("cuda", 0), estimate) is True


def test_a_budget_trimmed_by_our_own_residency_does_not_tile_a_decode_that_fits(monkeypatch):
    """Windows trims the budget when the process itself grows -- measured on an RX 9060 XT, 15.09 GiB until it passes
    about 12 of 16 GiB, then 7.62. The cache evicts those models for the reservation, so a decode that fits must not
    be tiled on that reading: it tiled a 7.0 GiB Z-Image decode against a 6.9 GiB line mid-session, which changed
    output that had been pixel-identical."""
    total_bytes = 16 * 2**30
    estimate = 7 * 2**30

    monkeypatch.setattr("torch.cuda.get_device_properties", lambda device: MagicMock(total_memory=total_bytes))
    # 12 GiB of our own models resident, and nothing else on the card.
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device: (4 * 2**30, total_bytes))
    monkeypatch.setattr("invokeai.backend.util.wddm.video_memory_budget", lambda device: int(7.6 * 2**30))
    monkeypatch.setattr("invokeai.backend.util.wddm.other_process_local_bytes", lambda device: 0)

    assert should_pretile_vae_decode(torch.device("cuda", 0), estimate) is False


def test_another_process_holding_the_card_tiles_even_while_the_budget_looks_fine(monkeypatch):
    """The case the budget does not answer: measured with another program holding 8 GiB and Invoke idle, the budget
    still read 15.09 GiB although only ~7.7 GiB could be had. Evicting Invoke's own models cannot recover memory it
    does not hold, so a 8.9 GiB Qwen-Image decode has to be tiled here."""
    total_bytes = 16 * 2**30
    estimate = int(8.9 * 2**30)

    monkeypatch.setattr("torch.cuda.get_device_properties", lambda device: MagicMock(total_memory=total_bytes))
    # Invoke holds almost nothing; torch's free figure does not see the other process at all.
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device: (total_bytes - 2**28, total_bytes))
    monkeypatch.setattr("invokeai.backend.util.wddm.video_memory_budget", lambda device: int(15.09 * 2**30))
    monkeypatch.setattr("invokeai.backend.util.wddm.other_process_local_bytes", lambda device: 8 * 2**30)

    assert estimate < VAE_PRETILE_VRAM_FRACTION * total_bytes, "the card total and the budget both look fine"
    assert should_pretile_vae_decode(torch.device("cuda", 0), estimate) is True


def test_the_budget_is_the_fallback_when_the_counters_cannot_answer(monkeypatch):
    """No adapter-wide figure means no way to tell whose memory it is; the budget is then the conservative answer."""
    total_bytes = 16 * 2**30
    estimate = 7 * 2**30

    monkeypatch.setattr("torch.cuda.get_device_properties", lambda device: MagicMock(total_memory=total_bytes))
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device: (4 * 2**30, total_bytes))
    monkeypatch.setattr("invokeai.backend.util.wddm.video_memory_budget", lambda device: int(7.6 * 2**30))
    monkeypatch.setattr("invokeai.backend.util.wddm.other_process_local_bytes", lambda device: None)

    assert should_pretile_vae_decode(torch.device("cuda", 0), estimate) is True
