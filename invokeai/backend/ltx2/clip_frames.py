"""Reading a gallery clip's frames onto an LTX-2 canvas."""

from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image

from invokeai.app.util.video_thumbnails import iter_video_frames
from invokeai.backend.ltx2.image_conditioning import fit_to_canvas
from invokeai.backend.ltx2.packing import normalize_video_latents
from invokeai.backend.ltx2.video_decoding import scoped_ltx2_tiling
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.util.cancel_hooks import cancel_before_forward

# How often the read loop reports. Decoding and cover-cropping a long clip is tens of seconds;
# without this the progress bar sits still for all of it.
PROGRESS_FRAME_INTERVAL = 48


class CanvasClip(NamedTuple):
    """Frames fitted to the canvas, and the length of the clip they came out of."""

    frames: list[np.ndarray]
    #: Frames the source holds, or ``None`` when the read stopped before the end and cannot say.
    #: A caller that needs the clip's span in *time* -- to line a soundtrack up against it -- has no
    #: other way to get it: a decode's audio runs past the picture by the codec's end padding.
    source_frames: int | None


def read_canvas_frames(
    path: Path,
    *,
    width: int,
    height: int,
    cap: int,
    tail: bool = False,
    is_canceled: Callable[[], bool] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> CanvasClip:
    """Decode up to ``cap`` frames of ``path``, each cover-cropped to ``width`` x ``height``.

    ``cap`` is a ceiling and not a formality: nothing upstream bounds a clip's length, so a workflow
    handing one of these nodes a ten-minute recording would otherwise materialize every frame before
    anything refused it. With ``tail`` the last ``cap`` frames are kept instead of the first, which
    is what a video extension conditions on.

    The two directions buffer differently, on purpose. Reading the head can stop at ``cap``, so each
    frame is fitted as it arrives and only canvas-sized frames are ever held. Reading the tail
    cannot stop early -- the last frames are only known once the clip ends -- so fitting on the way
    past would run a LANCZOS resize over the whole recording to keep a second of it: measured at
    20 ms a frame for 1080p onto a 1248x704 canvas, that is ~145 s of pure resizing for a five
    minute source, twice over on a two-stage run. The survivors are fitted instead, which trades a
    bounded deque of native frames (17 at 4K is ~420 MiB, and the cap for a tail read is the
    extension's context length, not the clip's) for work proportional to what is actually used.
    """

    def fitted(frame: np.ndarray) -> np.ndarray:
        # `fit_to_canvas` takes (height, width) and a PIL image, and already returns RGB.
        return np.asarray(fit_to_canvas(Image.fromarray(frame), height, width), dtype=np.uint8)

    frames: deque[np.ndarray] = deque(maxlen=cap) if tail else deque()
    read = 0

    for frame in iter_video_frames(path, is_canceled=is_canceled):
        frames.append(frame if tail else fitted(frame))
        read += 1
        if not tail and len(frames) >= cap:
            break
        if on_progress is not None and read % PROGRESS_FRAME_INTERVAL == 0:
            on_progress(read)

    # A tail read always runs to the end, so it knows the clip's length; a head read only does when
    # it ran out of frames before reaching the cap.
    source_frames = read if tail or read < cap else None

    return CanvasClip([fitted(frame) for frame in frames] if tail else list(frames), source_frames)


@torch.no_grad()
def encode_canvas_clip(
    vae: torch.nn.Module,
    frames: list[np.ndarray],
    *,
    tile_size: int,
    temporal_tile: int,
    is_canceled: Callable[[], bool] | None = None,
) -> torch.Tensor:
    """Canvas-sized uint8 frames -> normalized video latents ``[1, 128, T, h, w]``.

    ``frames`` is CONSUMED: the list is emptied once it has been stacked, so the caller cannot keep
    a second copy of the clip alive across the encode.

    Tiled, and not by preference: an untiled encode's activation grows with the whole clip rather
    than with one tile, so a 1248x704 clip of 121 frames needs about 65 GiB of it. Tiled at the
    defaults the same encode runs in 3.2 GiB, flat in the clip's length.
    """
    # Stacked as uint8 and converted on the device: the fp32 copy is four times the size, and making
    # it on the host would put ~2.4 GiB (for a 241-frame 1248x704 clip) outside every budget the
    # model cache knows about, on top of the list it is built from. CONSUMES `frames` for the same
    # reason -- the stack is a second copy of the clip, and holding both across the encode is 635
    # MiB at the caller's own cap that no budget accounts for.
    pixels = torch.from_numpy(np.stack(frames))
    frames.clear()
    vae_dtype = next(iter(vae.parameters())).dtype
    # Resolved from the model: the cache decides where it actually lands, and a partially-loaded or
    # CPU-resident VAE would reject input placed on the accelerator.
    device = get_effective_device(vae)

    source = pixels.to(device=device).permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, T, H, W]
    # Subtract before dividing: the arithmetic runs in the VAE's own (low-precision) dtype to keep
    # one copy of the clip rather than two, and `x / 127.5 - 1` cancels there -- mid-grey lands
    # within a bf16 ulp of 1.0, so the subtraction throws the value away. This order is the same
    # maths with no cancellation, and matches an fp32 round-trip.
    source = source.to(dtype=vae_dtype).sub_(127.5).div_(127.5)
    del pixels

    with (
        scoped_ltx2_tiling(vae, tile_size=tile_size, temporal_tile=temporal_tile),
        # One poll per encoder forward: a tiled encode of a long clip is hundreds of them, and
        # polling only around the whole call would leave a cancel waiting for all of it.
        cancel_before_forward([vae.encoder], is_canceled, device),
    ):
        latents = vae.encode(source).latent_dist.mode().to(torch.float32)
    del source

    return normalize_video_latents(latents, vae.latents_mean, vae.latents_std, float(vae.config.scaling_factor))
