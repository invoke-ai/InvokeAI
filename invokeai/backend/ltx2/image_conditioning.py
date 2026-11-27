"""Preparing an image as LTX-2 video conditioning.

Three steps, in the order the released pipeline applies them: re-compress the source image as a
single H.264 intra frame, fit it to the generation canvas, and VAE-encode it as one latent frame.

The re-compression is not an optimization. The model was trained on frames that had been through a
video codec, and a pristine PNG first frame is out of that distribution: the generated clip drifts
away from it over the first few frames. The CRF is the model generation's (18 for 2.5), and the
round trip goes through the same ffmpeg binary the gallery's MP4 writer uses, since PyAV -- what
the reference implementations use -- is not a dependency here.
"""

import subprocess
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from invokeai.backend.ltx2.packing import normalize_video_latents


def fit_to_canvas(image: Image.Image, height: int, width: int) -> Image.Image:
    """Orient, convert and cover-crop an image onto the generation canvas.

    Cover-cropping (rather than stretching to fit) is what upstream's image conditioner does, and
    what the rest of this codebase does with a keyframe: an anamorphic first frame would otherwise
    teach the model the wrong geometry for the whole clip.
    """
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.fit(oriented, (width, height), method=Image.Resampling.LANCZOS)


def recompress_h264(image: Image.Image, crf: int) -> Image.Image:
    """Encode the image as one H.264 frame at ``crf`` and decode it back. ``crf`` 0 is a no-op."""
    if crf <= 0:
        return image

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    # H.264 4:2:0 needs even dimensions; the reference crops rather than pads.
    height, width = rgb.shape[0] // 2 * 2, rgb.shape[1] // 2 * 2
    if height < 2 or width < 2:
        raise ValueError(f"An image conditioning must be at least 2x2 pixels to re-compress; got {image.size}.")
    rgb = np.ascontiguousarray(rgb[:height, :width])

    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    encoded = _run_ffmpeg(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "1", "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-frames:v", "1", "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1",
        ],
        rgb.tobytes(),
        "encode",
    )  # fmt: skip
    decoded = _run_ffmpeg(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "mp4", "-i", "pipe:0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ],
        encoded,
        "decode",
    )  # fmt: skip

    expected = height * width * 3
    if len(decoded) != expected:
        raise RuntimeError(
            f"H.264 re-compression returned {len(decoded)} bytes for a {width}x{height} frame, expected {expected}."
        )
    return Image.fromarray(np.frombuffer(decoded, dtype=np.uint8).reshape(height, width, 3))


def _run_ffmpeg(args: list[str], payload: bytes, stage: str) -> bytes:
    result = subprocess.run(args, input=payload, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", "replace").strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"ffmpeg failed to {stage} the image conditioning: {detail}")
    return result.stdout


@torch.no_grad()
def encode_image_latents(vae: Any, image: Image.Image, *, device: torch.device) -> torch.Tensor:
    """VAE-encode a canvas-sized image as one normalized latent frame ``(1, 128, 1, h, w)`` (CPU).

    The posterior's mode is taken rather than a sample, so the conditioning does not depend on the
    request seed.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    pixels = torch.from_numpy(rgb).float().div_(127.5).sub_(1.0)
    pixels = pixels.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]

    vae_dtype = next(iter(vae.parameters())).dtype
    with torch.inference_mode():
        posterior = vae.encode(pixels.to(device=device, dtype=vae_dtype), return_dict=False)[0]
        latents = posterior.mode().float()
        latents = normalize_video_latents(latents, vae.latents_mean, vae.latents_std, float(vae.config.scaling_factor))
    return latents.cpu()
