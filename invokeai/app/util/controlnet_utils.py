from typing import Any, Literal, Union

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from invokeai.backend.image_util.util import nms, normalize_image_channel_count

CONTROLNET_RESIZE_VALUES = Literal[
    "just_resize",
    "crop_resize",
    "fill_resize",
    "just_resize_simple",
]
CONTROLNET_MODE_VALUES = Literal["balanced", "more_prompt", "more_control", "unbalanced"]

###################################################################
# Copy of scripts/lvminthin.py from Mikubill/sd-webui-controlnet
###################################################################
# High Quality Edge Thinning using Pure Python
# Written by Lvmin Zhangu
# 2023 April
# Stanford University
# If you use this, please Cite "High Quality Edge Thinning using Pure Python", Lvmin Zhang, In Mikubill/sd-webui-controlnet.

lvmin_kernels_raw = [
    np.array([[-1, -1, -1], [0, 1, 0], [1, 1, 1]], dtype=np.int32),
    np.array([[0, -1, -1], [1, 1, -1], [0, 1, 0]], dtype=np.int32),
]

lvmin_kernels = []
lvmin_kernels += [np.rot90(x, k=0, axes=(0, 1)) for x in lvmin_kernels_raw]
lvmin_kernels += [np.rot90(x, k=1, axes=(0, 1)) for x in lvmin_kernels_raw]
lvmin_kernels += [np.rot90(x, k=2, axes=(0, 1)) for x in lvmin_kernels_raw]
lvmin_kernels += [np.rot90(x, k=3, axes=(0, 1)) for x in lvmin_kernels_raw]

lvmin_prunings_raw = [
    np.array([[-1, -1, -1], [-1, 1, -1], [0, 0, -1]], dtype=np.int32),
    np.array([[-1, -1, -1], [-1, 1, -1], [-1, 0, 0]], dtype=np.int32),
]

lvmin_prunings = []
lvmin_prunings += [np.rot90(x, k=0, axes=(0, 1)) for x in lvmin_prunings_raw]
lvmin_prunings += [np.rot90(x, k=1, axes=(0, 1)) for x in lvmin_prunings_raw]
lvmin_prunings += [np.rot90(x, k=2, axes=(0, 1)) for x in lvmin_prunings_raw]
lvmin_prunings += [np.rot90(x, k=3, axes=(0, 1)) for x in lvmin_prunings_raw]


def remove_pattern(x, kernel):
    objects = cv2.morphologyEx(x, cv2.MORPH_HITMISS, kernel)
    objects = np.where(objects > 127)
    x[objects] = 0
    return x, objects[0].shape[0] > 0


def thin_one_time(x, kernels):
    y = x
    is_done = True
    for k in kernels:
        y, has_update = remove_pattern(y, k)
        if has_update:
            is_done = False
    return y, is_done


def lvmin_thin(x, prunings=True):
    y = x
    for _i in range(32):
        y, is_done = thin_one_time(y, lvmin_kernels)
        if is_done:
            break
    if prunings:
        y, _ = thin_one_time(y, lvmin_prunings)
    return y


################################################################################
# copied from Mikubill/sd-webui-controlnet external_code.py and modified for InvokeAI
################################################################################
# FIXME: not using yet, if used in the future will most likely require modification of preprocessors
def pixel_perfect_resolution(
    image: np.ndarray,
    target_H: int,
    target_W: int,
    resize_mode: str,
) -> int:
    """
    Calculate the estimated resolution for resizing an image while preserving aspect ratio.

    The function first calculates scaling factors for height and width of the image based on the target
    height and width. Then, based on the chosen resize mode, it either takes the smaller or the larger
    scaling factor to estimate the new resolution.

    If the resize mode is OUTER_FIT, the function uses the smaller scaling factor, ensuring the whole image
    fits within the target dimensions, potentially leaving some empty space.

    If the resize mode is not OUTER_FIT, the function uses the larger scaling factor, ensuring the target
    dimensions are fully filled, potentially cropping the image.

    After calculating the estimated resolution, the function prints some debugging information.

    Args:
        image (np.ndarray): A 3D numpy array representing an image. The dimensions represent [height, width, channels].
        target_H (int): The target height for the image.
        target_W (int): The target width for the image.
        resize_mode (ResizeMode): The mode for resizing.

    Returns:
        int: The estimated resolution after resizing.
    """
    raw_H, raw_W, _ = image.shape

    k0 = float(target_H) / float(raw_H)
    k1 = float(target_W) / float(raw_W)

    if resize_mode == "fill_resize":
        estimation = min(k0, k1) * float(min(raw_H, raw_W))
    else:  # "crop_resize" or "just_resize" (or possibly "just_resize_simple"?)
        estimation = max(k0, k1) * float(min(raw_H, raw_W))

    # print(f"Pixel Perfect Computation:")
    # print(f"resize_mode = {resize_mode}")
    # print(f"raw_H = {raw_H}")
    # print(f"raw_W = {raw_W}")
    # print(f"target_H = {target_H}")
    # print(f"target_W = {target_W}")
    # print(f"estimation = {estimation}")

    return int(np.round(estimation))


def clone_contiguous(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Get a memory-contiguous clone of the given numpy array, as a safety measure and to improve computation efficiency."""
    return np.ascontiguousarray(x).copy()


def np_img_to_torch(np_img: np.ndarray[Any, Any], device: torch.device) -> torch.Tensor:
    """Convert a numpy image to a PyTorch tensor. The image is normalized to 0-1, rearranged to BCHW format and sent to
    the specified device."""

    torch_img = torch.from_numpy(np_img)
    normalized = torch_img.float() / 255.0
    bchw = rearrange(normalized, "h w c -> 1 c h w")
    on_device = bchw.to(device)
    return on_device.clone()


def heuristic_resize(np_img: np.ndarray[Any, Any], size: tuple[int, int]) -> np.ndarray[Any, Any]:
    """Resizes an image using a heuristic to choose the best resizing strategy.

    - If the image appears to be an edge map, special handling will be applied to ensure the edges are not distorted.
    - Single-pixel edge maps use NMS and thinning to keep the edges as single-pixel lines.
    - Low-color-count images are resized with nearest-neighbor to preserve color information (for e.g. segmentation maps).
    - The alpha channel is handled separately to ensure it is resized correctly.

    Args:
        np_img (np.ndarray): The input image.
        size (tuple[int, int]): The target size for the image.

    Returns:
        np.ndarray: The resized image.

    Adapted from https://github.com/Mikubill/sd-webui-controlnet.
    """

    # Return early if the image is already at the requested size
    if np_img.shape[0] == size[1] and np_img.shape[1] == size[0]:
        return np_img

    # If the image has an alpha channel, separate it for special handling later.
    inpaint_mask = None
    if np_img.ndim == 3 and np_img.shape[2] == 4:
        inpaint_mask = np_img[:, :, 3]
        np_img = np_img[:, :, 0:3]

    new_size_is_smaller = (size[0] * size[1]) < (np_img.shape[0] * np_img.shape[1])
    new_size_is_bigger = (size[0] * size[1]) > (np_img.shape[0] * np_img.shape[1])
    unique_color_count = np.unique(np_img.reshape(-1, np_img.shape[2]), axis=0).shape[0]
    is_one_pixel_edge = False
    is_binary = False

    if unique_color_count == 2:
        # If the image has only two colors, it is likely binary. Check if the image has one-pixel edges.
        is_binary = np.min(np_img) < 16 and np.max(np_img) > 240
        if is_binary:
            eroded = cv2.erode(np_img, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
            dilated = cv2.dilate(eroded, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
            one_pixel_edge_count = np.where(dilated < np_img)[0].shape[0]
            all_edge_count = np.where(np_img > 127)[0].shape[0]
            is_one_pixel_edge = one_pixel_edge_count * 2 > all_edge_count

    if 2 < unique_color_count < 200:
        # With a low color count, we assume this is a map where exact colors are important. Near-neighbor preserves
        # the colors as needed.
        interpolation = cv2.INTER_NEAREST
    elif new_size_is_smaller:
        # This works best for downscaling
        interpolation = cv2.INTER_AREA
    else:
        # Fall back for other cases
        interpolation = cv2.INTER_CUBIC  # Must be CUBIC because we now use nms. NEVER CHANGE THIS

    # This may be further transformed depending on the binary nature of the image.
    resized = cv2.resize(np_img, size, interpolation=interpolation)

    if inpaint_mask is not None:
        # Resize the inpaint mask to match the resized image using the same interpolation method.
        inpaint_mask = cv2.resize(inpaint_mask, size, interpolation=interpolation)

    # If the image is binary, we will perform some additional processing to ensure the edges are preserved.
    if is_binary:
        resized = np.mean(resized.astype(np.float32), axis=2).clip(0, 255).astype(np.uint8)
        if is_one_pixel_edge:
            # Use NMS and thinning to keep the edges as single-pixel lines.
            resized = nms(resized)
            _, resized = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            resized = lvmin_thin(resized, prunings=new_size_is_bigger)
        else:
            _, resized = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resized = np.stack([resized] * 3, axis=2)

    # Restore the alpha channel if it was present.
    if inpaint_mask is not None:
        inpaint_mask = (inpaint_mask > 127).astype(np.float32) * 255.0
        inpaint_mask = inpaint_mask[:, :, None].clip(0, 255).astype(np.uint8)
        resized = np.concatenate([resized, inpaint_mask], axis=2)

    return resized


###########################################################################
# Copied from detectmap_proc method in scripts/detectmap_proc.py in Mikubill/sd-webui-controlnet
#    modified for InvokeAI
###########################################################################
def np_img_resize(
    np_img: np.ndarray,
    resize_mode: CONTROLNET_RESIZE_VALUES,
    h: int,
    w: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, np.ndarray[Any, Any]]:
    np_img = normalize_image_channel_count(np_img)

    if resize_mode == "just_resize":  # RESIZE
        np_img = heuristic_resize(np_img, (w, h))
        np_img = clone_contiguous(np_img)
        return np_img_to_torch(np_img, device), np_img

    old_h, old_w, _ = np_img.shape
    old_w = float(old_w)
    old_h = float(old_h)
    k0 = float(h) / old_h
    k1 = float(w) / old_w

    def safeint(x: Union[int, float]) -> int:
        return int(np.round(x))

    if resize_mode == "fill_resize":  # OUTER_FIT
        k = min(k0, k1)
        borders = np.concatenate([np_img[0, :, :], np_img[-1, :, :], np_img[:, 0, :], np_img[:, -1, :]], axis=0)
        high_quality_border_color = np.median(borders, axis=0).astype(np_img.dtype)
        if len(high_quality_border_color) == 4:
            # Inpaint hijack
            high_quality_border_color[3] = 255
        high_quality_background = np.tile(high_quality_border_color[None, None], [h, w, 1])
        np_img = heuristic_resize(np_img, (safeint(old_w * k), safeint(old_h * k)))
        new_h, new_w, _ = np_img.shape
        pad_h = max(0, (h - new_h) // 2)
        pad_w = max(0, (w - new_w) // 2)
        high_quality_background[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = np_img
        np_img = high_quality_background
        np_img = clone_contiguous(np_img)
        return np_img_to_torch(np_img, device), np_img
    else:  # resize_mode == "crop_resize"  (INNER_FIT)
        k = max(k0, k1)
        np_img = heuristic_resize(np_img, (safeint(old_w * k), safeint(old_h * k)))
        new_h, new_w, _ = np_img.shape
        pad_h = max(0, (new_h - h) // 2)
        pad_w = max(0, (new_w - w) // 2)
        np_img = np_img[pad_h : pad_h + h, pad_w : pad_w + w]
        np_img = clone_contiguous(np_img)
        return np_img_to_torch(np_img, device), np_img


def prepare_control_image(
    image: Image.Image,
    width: int,
    height: int,
    num_channels: int = 3,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    control_mode: CONTROLNET_MODE_VALUES = "balanced",
    resize_mode: CONTROLNET_RESIZE_VALUES = "just_resize_simple",
    do_classifier_free_guidance: bool = True,
) -> torch.Tensor:
    """Pre-process images for ControlNets or T2I-Adapters.

    Args:
        image (Image): The PIL image to pre-process.
        width (int): The target width in pixels.
        height (int): The target height in pixels.
        num_channels (int, optional): The target number of image channels. This is achieved by converting the input
            image to RGB, then naively taking the first `num_channels` channels. The primary use case is converting a
            RGB image to a single-channel grayscale image. Raises if `num_channels` cannot be achieved. Defaults to 3.
        device (str | torch.Device, optional): The target device for the output image. Defaults to "cuda".
        dtype (_type_, optional): The dtype for the output image. Defaults to torch.float16.
        do_classifier_free_guidance (bool, optional): If True, repeat the output image along the batch dimension.
            Defaults to True.
        control_mode (str, optional): Defaults to "balanced".
        resize_mode (str, optional): Defaults to "just_resize_simple".

    Raises:
        ValueError: If `resize_mode` is not recognized.
        ValueError: If `num_channels` is out of range.

    Returns:
        torch.Tensor: The pre-processed input tensor.
    """
    if resize_mode == "just_resize_simple":
        image = image.convert("RGB")
        image = image.resize((width, height), resample=Image.LANCZOS)
        nimage = np.array(image)
        nimage = nimage[None, :]
        nimage = np.concatenate([nimage], axis=0)
        # normalizing RGB values to [0,1] range (in PIL.Image they are [0-255])
        nimage = np.array(nimage).astype(np.float32) / 255.0
        nimage = nimage.transpose(0, 3, 1, 2)
        timage = torch.from_numpy(nimage)

    # use fancy lvmin controlnet resizing
    elif resize_mode == "just_resize" or resize_mode == "crop_resize" or resize_mode == "fill_resize":
        nimage = np.array(image)
        timage, nimage = np_img_resize(
            np_img=nimage,
            resize_mode=resize_mode,
            h=height,
            w=width,
            device=torch.device(device),
        )
    else:
        raise ValueError(f"Unsupported resize_mode: '{resize_mode}'.")

    if timage.shape[1] < num_channels or num_channels <= 0:
        raise ValueError(f"Cannot achieve the target of num_channels={num_channels}.")
    timage = timage[:, :num_channels, :, :]

    timage = timage.to(device=device, dtype=dtype)
    cfg_injection = control_mode == "more_control" or control_mode == "unbalanced"
    if do_classifier_free_guidance and not cfg_injection:
        timage = torch.cat([timage] * 2)
    return timage
