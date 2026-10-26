import logging
from collections.abc import Hashable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING

import torch
from PIL import Image

from invokeai.app.util.step_callback import preview_frame, render_linear_preview
from invokeai.backend.architectures import LatentSpace, latent_space_with_channels, resolve_latent_space
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.tiny_autoencoder import named_tiny_autoencoder
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.model_manager.starter_models import STARTER_MODELS, StarterModel
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.tiny_autoencoder.decoder import TinyAutoencoderDecoder
from invokeai.backend.tiny_autoencoder.preview import decode_preview, select_tiny_autoencoder
from invokeai.backend.util.devices import TorchDevice

if TYPE_CHECKING:
    from invokeai.app.services.invocation_services import InvocationServices

# Predictable conditions already reported once, so later generations log them at DEBUG: a space with
# no installed TAE (keyed by the space's `id()` -- spaces are module-level constants that live as long
# as the process, and `LatentSpace` holds lists, so it does not hash) and a TAE record whose file is
# gone (keyed by the record's key). Shared by every generation thread.
_reported: set[Hashable] = set()
_reported_lock = Lock()

# A TAE frame (decode plus the JPEG encode of a full-size image) runs on the generation thread between
# two denoise steps; frames are accepted no closer than this many measured frame costs apart. The
# measurement excludes the bubble the frame leaves behind: syncing and encoding on the CPU drains the
# queue the CPU normally keeps ahead of the GPU, so the next step's launch latency shows, and on SDXL on
# an RTX 4090 that cost about as much again as the frame. A factor of 20 keeps the whole overhead near
# 10% whatever the GPU or the step time; the store's 0.1 s floor still applies to cheap frames.
RENDER_COST_INTERVAL_FACTOR = 20

# Weight of the newest frame in the running render-cost estimate: recent enough to follow a change in
# load, smooth enough that one slow frame does not stall the previews.
RENDER_COST_SMOOTHING = 0.5


def tiny_autoencoder_starters(space: LatentSpace) -> list[StarterModel]:
    """The starter models that install a TAE for `space`, read from the catalog by the TAE each one installs."""
    starters = []
    for starter in STARTER_MODELS:
        if starter.type is not ModelType.TinyAutoencoder:
            continue
        named = named_tiny_autoencoder((starter.source,))
        if named is not None and latent_space_with_channels(starter.base, named[2]) is space:
            starters.append(starter)
    return starters


def _first_report(key: Hashable) -> bool:
    with _reported_lock:
        first = key not in _reported
        _reported.add(key)
    return first


@dataclass(frozen=True)
class _HeldDecoder:
    decoder: TinyAutoencoderDecoder
    device: torch.device


class TinyAutoencoderPreviews:
    """Denoising previews decoded by an installed Tiny AutoEncoder, for one invocation.

    The TAE is locked in the model cache when it first resolves and stays locked until `close()`, which
    the invocation's teardown calls however the invocation ends. Locking it per frame instead would end
    every lock in the cache's allocator-reserve release, an `empty_cache()` at up to the preview rate.

    Falls back to the linear projection when no TAE decodes the sample's latent space, when the TAE
    would not be fully resident in VRAM, and, for the rest of the invocation, after any load or decode
    failure. Only the invocation's generation thread calls `render`, so its state needs no lock.
    """

    def __init__(self, services: "InvocationServices", user_id: str | None = None) -> None:
        self._services = services
        self._user_id = user_id
        self._disabled = False
        self._configs: list[AnyModelConfig] | None = None
        # One entry per latent space seen, compared by identity; None when the space renders linearly.
        self._decoders: list[tuple[LatentSpace, _HeldDecoder | None]] = []
        self._locks = ExitStack()
        # The render-cost estimate covers only frames the TAE decoded: a linear frame costs next to nothing
        # and must not pace anything, but its `.cpu()` waits for queued denoise work, so timing it would.
        self._tae_frames = 0
        self._render_cost_s: float | None = None
        self._decoded_on: torch.device | None = None  # where this frame's TAE decode ran; None if linear
        self._last_tae_device: torch.device | None = None

    @property
    def min_frame_interval_s(self) -> float | None:
        """How far apart TAE frames must be to keep their cost near 10% of the denoise; None for the store's own
        interval, which applies until a TAE frame was measured and again once the TAE is disabled."""
        if self._disabled or self._render_cost_s is None:
            return None
        return RENDER_COST_INTERVAL_FACTOR * self._render_cost_s

    @contextmanager
    def measure_frame(self) -> Iterator[None]:
        """Time one emitted frame, from its render through its encode, into the render-cost estimate.

        The TAE's device is synchronized first, so denoise work still queued on it is not counted as the
        frame's. After that, wall time is the frame's whole cost: the decode copies its image to the CPU,
        which waits for the decode. The first TAE frame is not counted, because it also looks the TAE up,
        loads and locks it.
        """
        if self._last_tae_device is not None:
            TorchDevice.synchronize(self._last_tae_device)
        self._decoded_on = None
        start = monotonic()
        yield
        cost = monotonic() - start
        if self._decoded_on is None or self._disabled:
            return
        self._last_tae_device = self._decoded_on
        self._tae_frames += 1
        if self._tae_frames == 1:
            return
        if self._render_cost_s is None:
            self._render_cost_s = cost
        else:
            self._render_cost_s += RENDER_COST_SMOOTHING * (cost - self._render_cost_s)

    def render(self, base_model: BaseModelType, sample: torch.Tensor) -> tuple[Image.Image, tuple[int, int]]:
        space = resolve_latent_space(base_model, preview_frame(sample))
        if not self._disabled:
            try:
                held = self._decoder_for(base_model, space)
                if held is not None:
                    image = decode_preview(held.decoder, sample, device=held.device)
                    self._decoded_on = held.device
                    size = (sample.shape[-1] * space.spatial_compression, sample.shape[-2] * space.spatial_compression)
                    return image, size
            except Exception:
                # OOM included: the preview is optional, the denoise around it is not.
                self._disable()
                self._services.logger.warning(
                    "Tiny AutoEncoder preview failed; using linear previews for the rest of this invocation.",
                    exc_info=True,
                )
        return render_linear_preview(base_model, sample)

    def close(self) -> None:
        """Release every TAE this invocation locked. Later frames render linearly."""
        self._disable()

    def _disable(self) -> None:
        self._disabled = True
        self._decoders.clear()
        self._locks.close()

    def _decoder_for(self, base_model: BaseModelType, space: LatentSpace) -> _HeldDecoder | None:
        for known, held in self._decoders:
            if known is space:
                return held
        held = self._resolve(base_model, space)
        if not self._disabled:
            self._decoders.append((space, held))
        return held

    def _resolve(self, base_model: BaseModelType, space: LatentSpace) -> _HeldDecoder | None:
        model_manager = self._services.model_manager
        if self._configs is None:
            self._configs = model_manager.store.search_by_attr(model_type=ModelType.TinyAutoencoder)
        config = select_tiny_autoencoder(self._configs, space)
        if config is None:
            self._report_missing(base_model, space)
            return None

        # Loaded through the service rather than `context.models.load`, which would signal a "Loading model"
        # progress message mid-denoise. The service's model_load_started/complete events still fire, once per
        # invocation.
        try:
            loaded = model_manager.load.load_model(config, user_id=self._user_id)
        except FileNotFoundError as error:
            self._report_unloadable(config, error)
            return None

        if not self._fits_in_vram(loaded):
            self._report_not_resident(config)
            return None

        lock = ExitStack()
        _, decoder = lock.enter_context(loaded.model_on_device())
        if not isinstance(decoder, TinyAutoencoderDecoder):
            lock.close()
            raise TypeError(f"expected a TinyAutoencoderDecoder, the loader returned {type(decoder).__name__}")
        if loaded.compute_device.type != "cpu" and loaded.resident_weight_bytes < loaded.weight_bytes:
            # The cache's own budget (which `max_vram_cache_size_gb` caps) left the TAE partly in RAM. A CPU
            # model reports no resident bytes at all: its lock does not move it.
            lock.close()
            self._disable()
            self._report_not_resident(config)
            return None
        self._locks.push(lock)
        return _HeldDecoder(decoder=decoder, device=loaded.compute_device)

    def _fits_in_vram(self, loaded: LoadedModel) -> bool:
        """Whether locking `loaded` would make it fully resident, asked before locking it.

        `make_room_in_vram` offloads unlocked models by the policy a lock applies and keeps the same
        working-memory reserve, so a TAE that would not fit is refused without the lock's "VRAM budget
        short" warning -- under partial loading, on every generation. It measures the device, not the
        cache's cap, so `_resolve` checks residency again after locking.
        """
        if loaded.compute_device.type == "cpu":
            return True
        missing = loaded.weight_bytes - loaded.resident_weight_bytes
        if missing <= 0:
            return True
        return self._services.model_manager.load.ram_cache.make_room_in_vram(missing) >= missing

    def _report_missing(self, base_model: BaseModelType, space: LatentSpace) -> None:
        starters = tiny_autoencoder_starters(space)
        if starters:
            install = " or ".join(f"the {s.name} starter model (listed under {s.base.value})" for s in starters)
            message = (
                f"No installed Tiny AutoEncoder decodes {base_model.value} latents; using linear previews. "
                f"Install {install}."
            )
        else:
            message = f"{base_model.value} has no Tiny AutoEncoder; using linear previews."
        self._log_once(("space", id(space)), logging.INFO, message)

    def _report_unloadable(self, config: AnyModelConfig, error: FileNotFoundError) -> None:
        self._log_once(
            ("file", config.key),
            logging.WARNING,
            f"Tiny AutoEncoder '{config.name}' could not be loaded ({error}); using linear previews. "
            "Reinstall it or remove it from the model manager.",
        )

    def _report_not_resident(self, config: AnyModelConfig) -> None:
        self._services.logger.info(
            f"Tiny AutoEncoder '{config.name}' does not fit in VRAM next to the running model; "
            "using linear previews for this invocation."
        )

    def _log_once(self, key: Hashable, level: int, message: str) -> None:
        self._services.logger.log(level if _first_report(key) else logging.DEBUG, message)
