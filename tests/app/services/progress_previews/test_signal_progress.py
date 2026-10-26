"""`signal_progress` wires the preview store into the progress event path."""

from types import SimpleNamespace

from PIL import Image

from invokeai.app.services.progress_previews.progress_previews_default import MemoryProgressPreviews
from invokeai.app.services.shared.invocation_context import UtilInterface


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FakeEvents:
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
            user_id=queue_item.user_id,
            queue_id=queue_item.queue_id,
        )


def _util(clock: _Clock) -> tuple[UtilInterface, _FakeEvents, MemoryProgressPreviews]:
    events = _FakeEvents()
    previews = MemoryProgressPreviews(clock=clock)
    services = SimpleNamespace(events=events, progress_previews=previews)
    queue_item = SimpleNamespace(item_id=7, session_id="session-7", user_id="owner", queue_id="default")
    data = SimpleNamespace(queue_item=queue_item, invocation=SimpleNamespace(id="node"))
    return UtilInterface(services, data, is_canceled=lambda: False), events, previews  # type: ignore[arg-type]


def test_image_frames_are_revisioned_throttled_and_retained() -> None:
    clock = _Clock()
    util, events, previews = _util(clock)
    image = Image.new("RGB", (8, 8))

    util.signal_progress("Denoising", 0.1, image, (64, 64))
    clock.now += 0.01
    util.signal_progress("Denoising", 0.2, image, (64, 64))  # inside the interval: dropped
    clock.now += 1
    util.signal_progress("Denoising", 0.3, image, (64, 64))

    assert [call["revision"] for call in events.emitted] == [1, 2]
    assert [call["percentage"] for call in events.emitted] == [0.1, 0.3]
    assert all(call["image"] is not None for call in events.emitted)
    retained = previews.get(7)
    assert retained is not None and retained.revision == 2


def test_imageless_progress_is_neither_throttled_nor_revisioned() -> None:
    clock = _Clock()
    util, events, previews = _util(clock)

    util.signal_progress("Loading model")
    util.signal_progress("Loading model", 0.5)

    assert len(events.emitted) == 2
    assert all("revision" not in call and call.get("image") is None for call in events.emitted)
    assert previews.get(7) is None


def test_signal_progress_flushes_a_deferred_empty_cache(monkeypatch) -> None:
    """Every denoise loop reports progress once per step, which is the busy worker's natural
    boundary for performing a process-global empty_cache that a peer deferred. Both the
    imageless and the image-bearing paths must reach the flush."""
    from invokeai.backend.util.devices import TorchDevice

    flushes: list[int] = []
    monkeypatch.setattr(TorchDevice, "flush_deferred_empty_cache", classmethod(lambda cls: flushes.append(1)))
    clock = _Clock()
    util, _events, _previews = _util(clock)

    util.signal_progress("Denoising", 0.1)
    assert len(flushes) == 1
    util.signal_progress("Denoising", 0.2, Image.new("RGB", (8, 8)), (64, 64))
    assert len(flushes) == 2


class _Render:
    """A preview render that counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> tuple[Image.Image, tuple[int, int]]:
        self.calls += 1
        return Image.new("RGB", (8, 8)), (64, 64)


def test_a_throttled_preview_frame_is_never_rendered() -> None:
    clock = _Clock()
    util, events, _previews = _util(clock)
    first, throttled, later = _Render(), _Render(), _Render()

    util._signal_preview("Denoising", 0.1, first)
    clock.now += 0.01
    util._signal_preview("Denoising", 0.2, throttled)
    clock.now += 1
    util._signal_preview("Denoising", 0.3, later)

    assert (first.calls, throttled.calls, later.calls) == (1, 0, 1)
    assert [call["percentage"] for call in events.emitted] == [0.1, 0.3]
    assert [(call["image"].width, call["image"].height) for call in events.emitted] == [(64, 64), (64, 64)]


def test_the_final_preview_frame_renders_inside_the_throttle_interval() -> None:
    clock = _Clock()
    util, events, previews = _util(clock)
    final = _Render()

    util._signal_preview("Denoising", 0.9, _Render())
    clock.now += 0.01
    util._signal_preview("Denoising", 1.0, final)

    assert final.calls == 1
    assert [call["revision"] for call in events.emitted] == [1, 2]
    retained = previews.get(7)
    assert retained is not None and retained.revision == 2


def test_signal_preview_flushes_a_deferred_empty_cache_even_when_throttled(monkeypatch) -> None:
    from invokeai.backend.util.devices import TorchDevice

    flushes: list[int] = []
    monkeypatch.setattr(TorchDevice, "flush_deferred_empty_cache", classmethod(lambda cls: flushes.append(1)))
    clock = _Clock()
    util, _events, _previews = _util(clock)

    util._signal_preview("Denoising", 0.1, _Render())
    util._signal_preview("Denoising", 0.2, _Render())  # throttled

    assert len(flushes) == 2
