"""Tiny AutoEncoder denoising previews, the lock they hold, and the linear fallback around them."""

import logging
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from invokeai.app.services.progress_previews.progress_previews_default import MemoryProgressPreviews
from invokeai.app.services.session_processor.session_processor_common import ProgressImage
from invokeai.app.services.shared.invocation_context import InvocationContextData, UtilInterface
from invokeai.app.util import tiny_autoencoder_previews
from invokeai.app.util.step_callback import render_linear_preview
from invokeai.app.util.tiny_autoencoder_previews import TinyAutoencoderPreviews, tiny_autoencoder_starters
from invokeai.backend.architectures import resolve_latent_space
from invokeai.backend.model_manager.load.model_cache.model_cache import MODEL_LOAD_LOCK
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.stable_diffusion.extensions.preview import PipelineIntermediateState
from invokeai.backend.tiny_autoencoder.checkpoint import TAEW2_1, build_taehv_decoder
from invokeai.backend.tiny_autoencoder.taesd import Decoder

LOGGER_NAME = "test.tiny_autoencoder_previews"


class _Store:
    def __init__(self, configs: list[SimpleNamespace]) -> None:
        self.configs = configs
        self.searches = 0

    def search_by_attr(self, model_type: ModelType | None = None, **_filters) -> list[SimpleNamespace]:
        self.searches += 1
        return [config for config in self.configs if model_type is None or config.type == model_type]


def _model_load_lock_state() -> str | None:
    """How MODEL_LOAD_LOCK is held right now. These tests run on one thread, so any holder is the code under test."""
    if MODEL_LOAD_LOCK._writer_active:
        return "write"
    return "read" if MODEL_LOAD_LOCK._readers else None


class _LoadedModel:
    """A cache entry: counts locks, and reports how much of the decoder a lock leaves resident.

    Like the cache's full-load wrapper, a CPU model reports nothing resident: its lock does not move it.
    """

    def __init__(self, decoder: torch.nn.Module, device: str = "cpu", resident_fraction_when_locked: float = 1.0):
        self.decoder = decoder
        self.model = decoder
        self.compute_device = torch.device(device)
        self.weight_bytes = sum(p.numel() * p.element_size() for p in decoder.parameters())
        self._resident_when_locked = int(self.weight_bytes * resident_fraction_when_locked)
        self.locks_taken = 0
        self.locks_held = 0
        self.working_mem_requests: list[int | None] = []
        self.model_load_lock_at_lock: list[str | None] = []

    @property
    def resident_weight_bytes(self) -> int:
        if self.compute_device.type == "cpu" or not self.locks_held:
            return 0
        return self._resident_when_locked

    @contextmanager
    def model_on_device(self, working_mem_bytes: int | None = None):
        self.working_mem_requests.append(working_mem_bytes)
        self.model_load_lock_at_lock.append(_model_load_lock_state())
        self.locks_taken += 1
        self.locks_held += 1
        try:
            yield None, self.decoder
        finally:
            self.locks_held -= 1


class _Load:
    """The model load service. Its cache's `make_room_in_vram` reports `vram_available` beyond the configured
    working-memory reserve, `working_mem_floor`; a larger requested reserve takes the difference from it, as the
    cache's own does."""

    def __init__(
        self,
        loaded: _LoadedModel,
        vram_available: int = 0,
        error: Exception | None = None,
        working_mem_floor: int = 3 * 2**30,
    ) -> None:
        self.loaded = loaded
        self.loads = 0
        self.error = error
        # (call, how MODEL_LOAD_LOCK was held during it), in call order.
        self.model_load_lock_during: list[tuple[str, str | None]] = []

        def make_room_in_vram(_needed: int, working_mem_bytes: int | None = None) -> int:
            self.model_load_lock_during.append(("make_room_in_vram", _model_load_lock_state()))
            return vram_available - max(0, (working_mem_bytes or 0) - working_mem_floor)

        self.ram_cache = SimpleNamespace(make_room_in_vram=make_room_in_vram)

    def load_model(self, config, submodel_type=None, user_id=None) -> _LoadedModel:
        self.model_load_lock_during.append(("load_model", _model_load_lock_state()))
        self.loads += 1
        if self.error is not None:
            raise self.error
        return self.loaded


class _Events:
    def __init__(self) -> None:
        self.emitted: list[dict] = []

    def emit_invocation_progress(self, **kwargs):
        self.emitted.append(kwargs)
        queue_item = kwargs["queue_item"]
        return SimpleNamespace(
            invocation=kwargs["invocation"],
            item_id=queue_item.item_id,
            session_id=queue_item.session_id,
            revision=kwargs.get("revision"),
        )


def _tae_config(base: BaseModelType = BaseModelType.StableDiffusion1, key: str = "tae-key") -> SimpleNamespace:
    return SimpleNamespace(key=key, name="TAESD", type=ModelType.TinyAutoencoder, base=base, latent_channels=4)


def _decoder() -> torch.nn.Module:
    torch.manual_seed(0)
    return Decoder(latent_channels=4).eval()


def _sample(seed: int = 1, height: int = 16, width: int = 24) -> torch.Tensor:
    return torch.randn(1, 4, height, width, generator=torch.Generator().manual_seed(seed))


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _services(
    store: _Store, load: _Load, enabled: bool = True, previews: MemoryProgressPreviews | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        logger=logging.getLogger(LOGGER_NAME),
        model_manager=SimpleNamespace(store=store, load=load),
        configuration=SimpleNamespace(tiny_autoencoder_previews=enabled),
        events=_Events(),
        progress_previews=previews or MemoryProgressPreviews(min_interval_s=0),
    )


def _load(decoder: torch.nn.Module | None = None, **kwargs) -> _Load:
    return _Load(_LoadedModel(decoder or _decoder(), **kwargs))


def _count_calls(module: torch.nn.Module) -> list[int]:
    calls: list[int] = []
    module.register_forward_pre_hook(lambda *_args: calls.append(1))
    return calls


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER_NAME and r.levelno == level]


def _is_linear(frame: tuple, sample: torch.Tensor) -> bool:
    linear, size = render_linear_preview(BaseModelType.StableDiffusion1, sample)
    return frame[1] == size and np.array_equal(np.asarray(frame[0]), np.asarray(linear))


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> _Clock:
    """Fresh process state, and a clock that only moves when a test moves it: a frame costs nothing
    unless the test says what it costs."""
    monkeypatch.setattr(tiny_autoencoder_previews, "_reported", set())
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    clock = _Clock()
    monkeypatch.setattr(tiny_autoencoder_previews, "monotonic", clock)
    return clock


def _invocation(services: SimpleNamespace) -> tuple[UtilInterface, InvocationContextData]:
    queue_item = SimpleNamespace(item_id=1, session_id="session", user_id="owner", queue_id="default")
    data = InvocationContextData(
        queue_item=queue_item,  # type: ignore[arg-type]
        invocation=SimpleNamespace(id="denoise"),  # type: ignore[arg-type]
        source_invocation_id="denoise",
    )
    return UtilInterface(services, data, is_canceled=lambda: False), data  # type: ignore[arg-type]


def _step(util: UtilInterface, sample: torch.Tensor, step: int = 0) -> None:
    state = PipelineIntermediateState(step=step, order=1, total_steps=10, timestep=0, latents=sample)
    util.sd_step_callback(state, BaseModelType.StableDiffusion1)


def _previews(services: SimpleNamespace) -> TinyAutoencoderPreviews:
    return TinyAutoencoderPreviews(services)  # type: ignore[arg-type]


def test_setting_off_renders_linear_previews_without_a_lookup_or_a_lock() -> None:
    store, load = _Store([_tae_config()]), _load()
    services = _services(store, load, enabled=False)
    util, _data = _invocation(services)

    _step(util, _sample(), step=0)
    _step(util, _sample(), step=1)

    assert (store.searches, load.loads, load.loaded.locks_taken) == (0, 0, 0)
    linear = ProgressImage.build(*render_linear_preview(BaseModelType.StableDiffusion1, _sample()))
    assert [call["image"] for call in services.events.emitted] == [linear, linear]


def test_the_tae_is_locked_once_per_invocation_and_released_with_its_resources() -> None:
    store, load = _Store([_tae_config()]), _load()
    loaded = load.loaded
    decodes = _count_calls(loaded.decoder)
    services = _services(store, load)

    for invocation in range(1, 3):
        util, data = _invocation(services)
        for step in range(4):
            _step(util, _sample(step, height=8, width=8), step=step)
            assert loaded.locks_held == 1
        assert (store.searches, load.loads, loaded.locks_taken) == (invocation, invocation, invocation)

        data.invocation_resources.close()
        assert loaded.locks_held == 0

    assert len(decodes) == 8
    assert all((call["image"].width, call["image"].height) == (64, 64) for call in services.events.emitted)


def test_the_tae_lock_reserves_the_working_memory_of_a_decode() -> None:
    load = _load()
    previews = _previews(_services(_Store([_tae_config()]), load))

    previews.render(BaseModelType.StableDiffusion1, _sample(height=16, width=24))
    previews.render(BaseModelType.StableDiffusion1, _sample(height=16, width=24))

    # One lock for the invocation, keeping three live 64-channel float32 activations of the 128 x 192 image and two
    # of convolution workspace free, and the fixed 5 MiB allowance.
    assert load.loaded.working_mem_requests == [(3 + 2) * 64 * (128 * 192) * 4 + 5 * 2**20]


def test_tae_frame_is_full_resolution_and_reports_the_linear_display_size() -> None:
    previews = _previews(_services(_Store([_tae_config()]), _load()))
    sample = _sample()

    image, size = previews.render(BaseModelType.StableDiffusion1, sample)
    linear, linear_size = render_linear_preview(BaseModelType.StableDiffusion1, sample)

    assert size == linear_size == (192, 128)
    assert image.size == (192, 128) and linear.size == (24, 16)
    assert image.mode == "RGB"


def test_capped_tae_frame_still_reports_the_full_generation_size() -> None:
    decoder = _decoder()
    decoder.max_preview_edge = 96
    previews = _previews(_services(_Store([_tae_config()]), _load(decoder)))

    image, size = previews.render(BaseModelType.StableDiffusion1, _sample(height=32, width=48))

    assert max(image.size) <= 96
    assert size == (48 * 8, 32 * 8)


def test_sibling_architecture_shares_the_tae_of_its_latent_space() -> None:
    load = _load()
    previews = _previews(_services(_Store([_tae_config(BaseModelType.StableDiffusion1)]), load))

    image, _size = previews.render(BaseModelType.StableDiffusion2, _sample())

    assert load.loads == 1
    assert image.size == (192, 128)


def test_missing_tae_renders_linear_suggests_the_starter_and_logs_info_once_per_process(
    caplog: pytest.LogCaptureFixture,
) -> None:
    load = _load()
    services = _services(_Store([_tae_config(BaseModelType.StableDiffusionXL)]), load)  # a different space
    sample = _sample()

    frames = [_previews(services).render(BaseModelType.StableDiffusion1, sample) for _ in range(2)]

    assert all(_is_linear(frame, sample) for frame in frames)
    assert (load.loads, load.loaded.locks_taken) == (0, 0)
    [info] = _messages(caplog, logging.INFO)
    assert "sd-1" in info and "starter model" in info
    assert len(_messages(caplog, logging.DEBUG)) == 1


def test_family_without_any_tae_is_told_so_rather_than_sent_to_the_starters(
    caplog: pytest.LogCaptureFixture,
) -> None:
    previews = _previews(_services(_Store([_tae_config()]), _load()))
    cogview4_sample = torch.randn(1, 16, 8, 8, generator=torch.Generator().manual_seed(0))

    previews.render(BaseModelType.CogView4, cogview4_sample)

    [info] = _messages(caplog, logging.INFO)
    assert "has no Tiny AutoEncoder" in info and "starter" not in info


def test_tae_whose_file_is_gone_warns_once_per_process_without_a_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    load = _load()
    load.error = FileNotFoundError("Files for model 'TAESD' not found at /models/taesd.safetensors")
    services = _services(_Store([_tae_config()]), load)
    sample = _sample()

    frames = [_previews(services).render(BaseModelType.StableDiffusion1, sample) for _ in range(2)]

    assert all(_is_linear(frame, sample) for frame in frames)
    warnings = [r for r in caplog.records if r.name == LOGGER_NAME and r.levelno == logging.WARNING]
    assert len(warnings) == 1 and warnings[0].exc_info is None
    assert "TAESD" in warnings[0].getMessage()


def test_decode_failure_releases_the_lock_and_renders_linear_for_the_rest_of_the_invocation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    decoder = _decoder()
    attempts: list[int] = []

    def fail(*_args) -> None:
        attempts.append(1)
        raise torch.OutOfMemoryError("CUDA out of memory")

    decoder.register_forward_pre_hook(fail)
    load = _load(decoder)
    previews = _previews(_services(_Store([_tae_config()]), load))
    sample = _sample()

    frames = [previews.render(BaseModelType.StableDiffusion1, sample) for _ in range(3)]

    assert len(attempts) == 1
    assert load.loaded.locks_held == 0
    assert all(_is_linear(frame, sample) for frame in frames)
    warnings = [r for r in caplog.records if r.name == LOGGER_NAME and r.levelno == logging.WARNING]
    assert len(warnings) == 1 and warnings[0].exc_info is not None


def test_tae_left_partly_in_ram_by_its_lock_is_released_and_renders_linear(caplog: pytest.LogCaptureFixture) -> None:
    decoder = _decoder()
    decodes = _count_calls(decoder)
    # The device is only read by the residency checks; nothing is decoded on it.
    loaded = _LoadedModel(decoder, device="cuda", resident_fraction_when_locked=0.5)
    load = _Load(loaded, vram_available=loaded.weight_bytes)
    previews = _previews(_services(_Store([_tae_config()]), load))
    sample = _sample()

    frames = [previews.render(BaseModelType.StableDiffusion1, sample) for _ in range(2)]

    assert all(_is_linear(frame, sample) for frame in frames)
    assert decodes == []
    assert (load.loads, load.loaded.locks_taken, load.loaded.locks_held) == (1, 1, 0)
    assert len(_messages(caplog, logging.INFO)) == 1


def test_the_fit_check_offloads_under_the_model_load_read_lock_and_loads_and_locks_outside_it() -> None:
    """The offload is a VRAM move: a construction on another worker would strand the weights it registers on `meta`
    unless it holds the read lock, as every VRAM move does. The load (a writer on a cache miss) and the lock take
    MODEL_LOAD_LOCK themselves, and the lock is not reentrant: held around either, it would deadlock."""
    # Locked half-resident, so the invocation stops before decoding on a device it does not have.
    loaded = _LoadedModel(_decoder(), device="cuda", resident_fraction_when_locked=0.5)
    load = _Load(loaded, vram_available=loaded.weight_bytes)
    previews = _previews(_services(_Store([_tae_config()]), load))

    previews.render(BaseModelType.StableDiffusion1, _sample())

    assert load.model_load_lock_during == [("load_model", None), ("make_room_in_vram", "read")]
    assert loaded.model_load_lock_at_lock == [None]
    assert _model_load_lock_state() is None


def test_tae_that_would_not_fit_in_vram_is_never_locked(caplog: pytest.LogCaptureFixture) -> None:
    # The device is only read by the residency decision; nothing is decoded on it.
    loaded = _LoadedModel(_decoder(), device="cuda")
    load = _Load(loaded, vram_available=loaded.weight_bytes // 2)
    previews = _previews(_services(_Store([_tae_config()]), load))
    sample = _sample()

    frames = [previews.render(BaseModelType.StableDiffusion1, sample) for _ in range(2)]

    assert all(_is_linear(frame, sample) for frame in frames)
    assert (load.loads, loaded.locks_taken) == (1, 0)
    (message,) = _messages(caplog, logging.INFO)
    assert "Decode previews on CPU" in message


def test_tae_whose_decode_would_not_fit_next_to_it_is_never_locked(caplog: pytest.LogCaptureFixture) -> None:
    """With `device_working_mem_gb` below the decode's reservation, the room the lock would keep decides, not the
    weights alone: locking anyway would partially load the TAE, with the cache's warning, on every generation."""
    loaded = _LoadedModel(_decoder(), device="cuda")  # only read by the residency decision
    load = _Load(loaded, vram_available=loaded.weight_bytes + 2**20, working_mem_floor=2**20)
    previews = _previews(_services(_Store([_tae_config()]), load))
    sample = _sample()

    assert _is_linear(previews.render(BaseModelType.StableDiffusion1, sample), sample)
    assert loaded.locks_taken == 0
    assert len(_messages(caplog, logging.INFO)) == 1


def test_a_tae_that_does_not_fit_is_reported_once_per_process_and_another_tae_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def generation(config: SimpleNamespace) -> None:
        loaded = _LoadedModel(_decoder(), device="cuda")  # only read by the residency decision
        load = _Load(loaded, vram_available=0)
        _previews(_services(_Store([config]), load)).render(BaseModelType.StableDiffusion1, _sample())

    generation(_tae_config())
    generation(_tae_config())
    assert len(_messages(caplog, logging.INFO)) == 1
    assert len(_messages(caplog, logging.DEBUG)) == 1

    generation(_tae_config(key="another-tae"))
    assert len(_messages(caplog, logging.INFO)) == 2


def test_a_decode_failure_warns_with_its_traceback_once_per_process_and_a_new_failure_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def generation(error: Exception, key: str = "tae-key") -> None:
        decoder = _decoder()

        def fail(*_args) -> None:
            raise error

        decoder.register_forward_pre_hook(fail)
        services = _services(_Store([_tae_config(key=key)]), _load(decoder))
        _previews(services).render(BaseModelType.StableDiffusion1, _sample())

    def warnings() -> list[logging.LogRecord]:
        return [r for r in caplog.records if r.name == LOGGER_NAME and r.levelno == logging.WARNING]

    generation(torch.OutOfMemoryError("CUDA out of memory"))
    generation(torch.OutOfMemoryError("CUDA out of memory"))
    assert [w.exc_info is not None for w in warnings()] == [True]
    [repeat] = [r for r in caplog.records if r.name == LOGGER_NAME and r.levelno == logging.DEBUG]
    assert repeat.exc_info is None

    generation(RuntimeError("MPS backend failure"))
    assert [w.exc_info is not None for w in warnings()] == [True, True]

    generation(torch.OutOfMemoryError("CUDA out of memory"), key="another-tae")  # another TAE for the same space
    assert [w.exc_info is not None for w in warnings()] == [True, True, True]


def test_close_releases_the_lock_and_later_frames_render_linear() -> None:
    load = _load()
    previews = _previews(_services(_Store([_tae_config()]), load))
    sample = _sample()
    previews.render(BaseModelType.StableDiffusion1, sample)

    previews.close()
    previews.close()

    assert load.loaded.locks_held == 0
    assert _is_linear(previews.render(BaseModelType.StableDiffusion1, sample), sample)
    assert load.loaded.locks_taken == 1


def test_tae_frames_are_spaced_by_their_render_cost_and_the_final_frame_still_renders(clock: _Clock) -> None:
    decoder = _decoder()
    decodes = _count_calls(decoder)
    frame_cost = 0.05
    decoder.register_forward_pre_hook(lambda *_args: setattr(clock, "now", clock.now + frame_cost))
    services = _services(_Store([_tae_config()]), _load(decoder), previews=MemoryProgressPreviews(clock=clock))
    util, _data = _invocation(services)

    def step_after(seconds: float, step: int) -> None:
        clock.now += seconds
        _step(util, _sample(step), step=step)

    step_after(0.0, 0)  # first frame: a CPU decode is counted, without the lookup, load and lock
    step_after(0.2, 1)  # 0.2 s after the last frame, inside k x 0.05 s = 1 s: not rendered
    step_after(0.85, 2)  # 1.05 s after it: rendered
    step_after(0.3, 3)  # not rendered
    step_after(0.01, 10)  # the final frame passes any interval

    assert [call["percentage"] for call in services.events.emitted] == [0.0, 0.2, 1.0]
    assert len(decodes) == 3


def test_linear_previews_keep_the_stores_interval(clock: _Clock) -> None:
    services = _services(_Store([]), _load(), enabled=False, previews=MemoryProgressPreviews(clock=clock))
    util, _data = _invocation(services)

    for step in range(4):
        clock.now += 0.15
        _step(util, _sample(step), step=step)
    clock.now += 0.05
    _step(util, _sample(), step=4)  # inside 0.1 s: dropped

    assert [call["percentage"] for call in services.events.emitted] == [0.0, 0.1, 0.2, 0.3]


def test_a_cpu_tae_is_used_although_its_lock_leaves_nothing_resident() -> None:
    load = _load()
    previews = _previews(_services(_Store([_tae_config()]), load))

    image, _size = previews.render(BaseModelType.StableDiffusion1, _sample())

    assert load.loaded.locks_held == 1 and load.loaded.resident_weight_bytes == 0
    assert image.size == (192, 128)


def test_a_cpu_tae_renders_where_the_same_tae_on_the_gpu_is_refused_for_vram() -> None:
    """No VRAM is free next to the running model. A TAE with "Decode previews on CPU" gets the cache's CPU execution device, so
    it needs none, and its decode is reserved at the 512-pixel CPU cap, not the decoder's own."""
    sample = _sample(height=100, width=100)  # an 800-pixel image
    gpu = _LoadedModel(_decoder(), device="cuda")  # only read by the residency decision
    gpu_previews = _previews(_services(_Store([_tae_config()]), _Load(gpu, vram_available=0)))
    cpu = _LoadedModel(_decoder(), device="cpu")
    cpu_previews = _previews(_services(_Store([_tae_config()]), _Load(cpu, vram_available=0)))

    assert _is_linear(gpu_previews.render(BaseModelType.StableDiffusion1, sample), sample)
    image, size = cpu_previews.render(BaseModelType.StableDiffusion1, sample)

    assert (gpu.locks_taken, cpu.locks_taken) == (0, 1)
    assert image.size == (512, 512) and size == (800, 800)
    assert cpu.working_mem_requests == [(3 + 2) * 64 * (512 * 512) * 4 + 5 * 2**20]


def test_a_cpu_tae_is_paced_from_its_first_frame_without_the_step_still_queued_on_the_latents_device(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Latents on another device (`meta` stands in for the GPU) with a 5 s denoise step still queued: the CPU decode's
    copy of them waits for it, and that wait is not the preview's cost. The first frame, which also looks the TAE up,
    already sets the interval, so the frame after it is paced."""
    decode_cost, queued_step = 0.3, 5.0
    queued = [0.0]

    def synchronize(device: torch.device) -> None:
        if device.type == "meta":
            clock.now += queued[0]
            queued[0] = 0.0

    def decode(_decoder, _sample, device=None):
        clock.now += queued[0] + decode_cost  # the copy to the CPU waits for whatever is still queued
        queued[0] = 0.0
        return Image.new("RGB", (8, 8))

    monkeypatch.setattr(tiny_autoencoder_previews.TorchDevice, "synchronize", synchronize)
    monkeypatch.setattr(tiny_autoencoder_previews, "decode_preview", decode)
    previews = _previews(_services(_Store([_tae_config()]), _load()))
    sample = torch.empty(1, 4, 16, 24, device="meta")
    intervals = []
    for _frame in range(3):
        queued[0] = queued_step
        with previews.measure_frame():
            previews.render(BaseModelType.StableDiffusion1, sample)
        intervals.append(previews.min_frame_interval_s)

    expected = tiny_autoencoder_previews.RENDER_COST_INTERVAL_FACTOR * decode_cost
    assert intervals == [pytest.approx(expected)] * 3


def test_a_gpu_tae_is_paced_only_from_its_second_frame(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first GPU frame also loads the TAE and makes the first cuDNN call, so its cost is not representative."""
    frame_costs = iter([1.0, 0.05, 0.05])

    def decode(_decoder, _sample, device=None):
        clock.now += next(frame_costs)
        return Image.new("RGB", (8, 8))

    monkeypatch.setattr(tiny_autoencoder_previews.TorchDevice, "synchronize", lambda _device: None)
    monkeypatch.setattr(tiny_autoencoder_previews, "decode_preview", decode)
    loaded = _LoadedModel(_decoder(), device="cuda")  # only read by the residency decision; nothing runs on it
    previews = _previews(_services(_Store([_tae_config()]), _Load(loaded, vram_available=2**40)))
    intervals = []
    for _frame in range(2):
        with previews.measure_frame():
            previews.render(BaseModelType.StableDiffusion1, _sample())
        intervals.append(previews.min_frame_interval_s)

    assert intervals == [None, pytest.approx(tiny_autoencoder_previews.RENDER_COST_INTERVAL_FACTOR * 0.05)]


def test_a_gpu_tae_above_its_cap_is_reserved_at_its_own_cap_not_the_cpu_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """An 800-pixel preview fits TAESD's own 1024-pixel cap; only a CPU decode is limited to 512."""
    monkeypatch.setattr(tiny_autoencoder_previews, "decode_preview", lambda *_a, **_k: Image.new("RGB", (8, 8)))
    loaded = _LoadedModel(_decoder(), device="cuda")  # only read by the residency decision; nothing runs on it
    previews = _previews(_services(_Store([_tae_config()]), _Load(loaded, vram_available=2**40)))

    previews.render(BaseModelType.StableDiffusion1, _sample(height=100, width=100))

    assert loaded.working_mem_requests == [(3 + 2) * 64 * (800 * 800) * 4 + 5 * 2**20]


def _slow_linear_frames(monkeypatch: pytest.MonkeyPatch, clock: _Clock, seconds: float) -> None:
    """A linear frame whose `.cpu()` waits out a long denoise step still queued on the GPU."""
    render = tiny_autoencoder_previews.render_linear_preview

    def slow(base_model, sample):
        clock.now += seconds
        return render(base_model, sample)

    monkeypatch.setattr(tiny_autoencoder_previews, "render_linear_preview", slow)


def test_slow_linear_frames_without_an_installed_tae_keep_the_stores_interval(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    _slow_linear_frames(monkeypatch, clock, seconds=1.0)
    services = _services(_Store([]), _load(), previews=MemoryProgressPreviews(clock=clock))
    util, _data = _invocation(services)

    for step in range(4):
        clock.now += 0.15
        _step(util, _sample(step), step=step)

    assert [call["percentage"] for call in services.events.emitted] == [0.0, 0.1, 0.2, 0.3]


def test_frames_after_the_tae_fell_back_to_linear_return_to_the_stores_interval(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    decoder = _decoder()
    decodes: list[int] = []

    def decode_then_fail(*_args) -> None:
        decodes.append(1)
        clock.now += 0.05
        if len(decodes) == 3:
            raise torch.OutOfMemoryError("CUDA out of memory")

    decoder.register_forward_pre_hook(decode_then_fail)
    _slow_linear_frames(monkeypatch, clock, seconds=1.0)
    syncs: list[torch.device] = []
    monkeypatch.setattr(tiny_autoencoder_previews.TorchDevice, "synchronize", syncs.append)
    services = _services(_Store([_tae_config()]), _load(decoder), previews=MemoryProgressPreviews(clock=clock))
    util, _data = _invocation(services)

    for step, after in enumerate([0.0, 1.1, 1.1, 0.15, 0.15]):
        clock.now += after
        _step(util, _sample(step), step=step)

    # Two TAE frames paced at k x 0.05 s; the third decode fails and renders linearly, and the frames after
    # it are 0.15 s apart again, with each linear frame's 1 s of waiting not counted as a cost.
    assert [call["percentage"] for call in services.events.emitted] == [0.0, 0.1, 0.2, 0.3, 0.4]
    assert len(decodes) == 3
    # The first frame waits for the latents' device, the frames after a TAE frame for the TAE's; once it fell back,
    # there is none to wait for.
    assert len(syncs) == 3


def _wan_window(frames: int = 2) -> torch.Tensor:
    """What the Wan video denoiser reports: the middle latent frame and the one before it."""
    return torch.randn(1, 16, frames, 6, 8, generator=torch.Generator().manual_seed(3))


def test_a_latent_window_renders_linearly_as_its_last_frame() -> None:
    window = _wan_window()

    image, size = render_linear_preview(BaseModelType.Wan, window)
    last, last_size = render_linear_preview(BaseModelType.Wan, window[:, :, -1])

    assert np.array_equal(np.asarray(image), np.asarray(last))
    assert size == last_size == (64, 48)


def test_a_latent_window_is_decoded_by_the_video_tae_of_its_space() -> None:
    torch.manual_seed(0)
    taew2_1 = build_taehv_decoder(TAEW2_1).eval()
    config = SimpleNamespace(
        key="taew2_1", name="TAEW2.1", type=ModelType.TinyAutoencoder, base=BaseModelType.Wan, latent_channels=16
    )
    load = _load(taew2_1)
    previews = _previews(_services(_Store([config]), load))
    window = _wan_window()

    image, size = previews.render(BaseModelType.Wan, window)

    with torch.inference_mode():
        last = taew2_1.decode_video(window.transpose(1, 2))[0, -1]
    expected = last.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    assert np.array_equal(np.asarray(image), expected)
    assert size == (64, 48)
    # The lock keeps the whole window's decode free: two latent frames, four output frames each, of the 48 x 64
    # image, at 2.25 live float32 activations and 2 of workspace per output frame, and the fixed 5 MiB allowance.
    assert load.loaded.working_mem_requests == [(2.25 + 2) * 64 * (48 * 64) * 4 * 8 + 5 * 2**20]


def test_a_missing_tae_names_the_starter_of_its_space_even_under_another_base(caplog: pytest.LogCaptureFixture) -> None:
    """Anima denoises in the Wan 2.1 VAE's space, whose TAE is filed under Wan in the starter list."""
    previews = _previews(_services(_Store([]), _load()))
    anima_sample = torch.randn(1, 16, 8, 8, generator=torch.Generator().manual_seed(0))

    previews.render(BaseModelType.Anima, anima_sample)

    [info] = _messages(caplog, logging.INFO)
    assert "anima" in info
    assert "the TAEW2.1 (Wan 2.1 VAE preview decoder) starter model (listed under wan)" in info


def _space_of(base: BaseModelType, channels: int):
    return resolve_latent_space(base, torch.zeros(1, channels, 2, 2))


@pytest.mark.parametrize(
    ("base", "channels", "starters"),
    [
        (BaseModelType.StableDiffusion2, 4, ["TAESD (SD 1.x/2.x preview decoder)"]),
        (BaseModelType.ZImage, 16, ["TAEF1 (FLUX.1 preview decoder)"]),
        (BaseModelType.Ideogram4, 32, ["TAEF2 (FLUX.2 preview decoder)"]),
        (BaseModelType.QwenImage21, 64, ["TAEQI2.1 (Qwen-Image-2.1 preview decoder)"]),
        (BaseModelType.Anima, 16, ["TAEW2.1 (Wan 2.1 VAE preview decoder)"]),
        (BaseModelType.Wan, 48, ["TAEW2.2 (Wan 2.2 TI2V-5B preview decoder)"]),
        (BaseModelType.LTX2, 128, ["TAELTX2.3 (LTX-2.5 preview decoder)"]),
        (BaseModelType.CogView4, 16, []),
        (BaseModelType.MiniMaxH3, 24, []),
    ],
)
def test_the_starters_that_install_a_tae_for_a_space(base: BaseModelType, channels: int, starters: list[str]) -> None:
    assert [starter.name for starter in tiny_autoencoder_starters(_space_of(base, channels))] == starters
