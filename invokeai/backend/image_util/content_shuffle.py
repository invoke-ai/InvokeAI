# Adapted from https://github.com/huggingface/controlnet_aux

import cv2
import numpy as np
from PIL import Image

from invokeai.backend.image_util.util import np_to_pil, pil_to_np


def make_noise_disk(H, W, C, F):
    noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F : F + H, F : F + W]
    noise -= np.min(noise)
    noise /= np.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise


def content_shuffle(input_image: Image.Image, scale_factor: int | None = None) -> Image.Image:
    """Shuffles the content of an image using a disk noise pattern, similar to a 'liquify' effect."""

    np_img = pil_to_np(input_image)

    height, width, _channels = np_img.shape

    if scale_factor is None:
        scale_factor = 256

    x = make_noise_disk(height, width, 1, scale_factor) * float(width - 1)
    y = make_noise_disk(height, width, 1, scale_factor) * float(height - 1)

    flow = np.concatenate([x, y], axis=2).astype(np.float32)

    shuffled_img = cv2.remap(np_img, flow, None, cv2.INTER_LINEAR)

    output_img = np_to_pil(shuffled_img)

    return output_img
