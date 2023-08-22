from typing import Union
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers.utils import PIL_INTERPOLATION

from einops import rearrange
from controlnet_aux.util import HWC3

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
    for i in range(32):
        y, is_done = thin_one_time(y, lvmin_kernels)
        if is_done:
            break
    if prunings:
        y, _ = thin_one_time(y, lvmin_prunings)
    return y


def nake_nms(x):
    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    y = np.zeros_like(x)
    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)
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


###########################################################################
# Copied from detectmap_proc method in scripts/detectmap_proc.py in Mikubill/sd-webui-controlnet
#    modified for InvokeAI
###########################################################################
# def detectmap_proc(detected_map, module, resize_mode, h, w):
def np_img_resize(np_img: np.ndarray, resize_mode: str, h: int, w: int, device: torch.device = torch.device("cpu")):
    # if 'inpaint' in module:
    #     np_img = np_img.astype(np.float32)
    # else:
    #     np_img = HWC3(np_img)
    np_img = HWC3(np_img)

    def safe_numpy(x):
        # A very safe method to make sure that Apple/Mac works
        y = x

        # below is very boring but do not change these. If you change these Apple or Mac may fail.
        y = y.copy()
        y = np.ascontiguousarray(y)
        y = y.copy()
        return y

    def get_pytorch_control(x):
        # A very safe method to make sure that Apple/Mac works
        y = x

        # below is very boring but do not change these. If you change these Apple or Mac may fail.
        y = torch.from_numpy(y)
        y = y.float() / 255.0
        y = rearrange(y, "h w c -> 1 c h w")
        y = y.clone()
        # y = y.to(devices.get_device_for("controlnet"))
        y = y.to(device)
        y = y.clone()
        return y

    def high_quality_resize(x: np.ndarray, size):
        # Written by lvmin
        # Super high-quality control map up-scaling, considering binary, seg, and one-pixel edges
        inpaint_mask = None
        if x.ndim == 3 and x.shape[2] == 4:
            inpaint_mask = x[:, :, 3]
            x = x[:, :, 0:3]

        new_size_is_smaller = (size[0] * size[1]) < (x.shape[0] * x.shape[1])
        new_size_is_bigger = (size[0] * size[1]) > (x.shape[0] * x.shape[1])
        unique_color_count = np.unique(x.reshape(-1, x.shape[2]), axis=0).shape[0]
        is_one_pixel_edge = False
        is_binary = False
        if unique_color_count == 2:
            is_binary = np.min(x) < 16 and np.max(x) > 240
            if is_binary:
                xc = x
                xc = cv2.erode(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
                xc = cv2.dilate(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
                one_pixel_edge_count = np.where(xc < x)[0].shape[0]
                all_edge_count = np.where(x > 127)[0].shape[0]
                is_one_pixel_edge = one_pixel_edge_count * 2 > all_edge_count

        if 2 < unique_color_count < 200:
            interpolation = cv2.INTER_NEAREST
        elif new_size_is_smaller:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC  # Must be CUBIC because we now use nms. NEVER CHANGE THIS

        y = cv2.resize(x, size, interpolation=interpolation)
        if inpaint_mask is not None:
            inpaint_mask = cv2.resize(inpaint_mask, size, interpolation=interpolation)

        if is_binary:
            y = np.mean(y.astype(np.float32), axis=2).clip(0, 255).astype(np.uint8)
            if is_one_pixel_edge:
                y = nake_nms(y)
                _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                y = lvmin_thin(y, prunings=new_size_is_bigger)
            else:
                _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            y = np.stack([y] * 3, axis=2)

        if inpaint_mask is not None:
            inpaint_mask = (inpaint_mask > 127).astype(np.float32) * 255.0
            inpaint_mask = inpaint_mask[:, :, None].clip(0, 255).astype(np.uint8)
            y = np.concatenate([y, inpaint_mask], axis=2)

        return y

    # if resize_mode == external_code.ResizeMode.RESIZE:
    if resize_mode == "just_resize":  # RESIZE
        np_img = high_quality_resize(np_img, (w, h))
        np_img = safe_numpy(np_img)
        return get_pytorch_control(np_img), np_img

    old_h, old_w, _ = np_img.shape
    old_w = float(old_w)
    old_h = float(old_h)
    k0 = float(h) / old_h
    k1 = float(w) / old_w

    def safeint(x: Union[int, float]) -> int:
        return int(np.round(x))

    # if resize_mode == external_code.ResizeMode.OUTER_FIT:
    if resize_mode == "fill_resize":  # OUTER_FIT
        k = min(k0, k1)
        borders = np.concatenate([np_img[0, :, :], np_img[-1, :, :], np_img[:, 0, :], np_img[:, -1, :]], axis=0)
        high_quality_border_color = np.median(borders, axis=0).astype(np_img.dtype)
        if len(high_quality_border_color) == 4:
            # Inpaint hijack
            high_quality_border_color[3] = 255
        high_quality_background = np.tile(high_quality_border_color[None, None], [h, w, 1])
        np_img = high_quality_resize(np_img, (safeint(old_w * k), safeint(old_h * k)))
        new_h, new_w, _ = np_img.shape
        pad_h = max(0, (h - new_h) // 2)
        pad_w = max(0, (w - new_w) // 2)
        high_quality_background[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = np_img
        np_img = high_quality_background
        np_img = safe_numpy(np_img)
        return get_pytorch_control(np_img), np_img
    else:  # resize_mode == "crop_resize"  (INNER_FIT)
        k = max(k0, k1)
        np_img = high_quality_resize(np_img, (safeint(old_w * k), safeint(old_h * k)))
        new_h, new_w, _ = np_img.shape
        pad_h = max(0, (new_h - h) // 2)
        pad_w = max(0, (new_w - w) // 2)
        np_img = np_img[pad_h : pad_h + h, pad_w : pad_w + w]
        np_img = safe_numpy(np_img)
        return get_pytorch_control(np_img), np_img


def prepare_control_image(
    # image used to be Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor, List[torch.Tensor]]
    # but now should be able to assume that image is a single PIL.Image, which simplifies things
    image: Image,
    # FIXME: need to fix hardwiring of width and height, change to basing on latents dimensions?
    # latents_to_match_resolution, # TorchTensor of shape (batch_size, 3, height, width)
    width=512,  # should be 8 * latent.shape[3]
    height=512,  # should be 8 * latent height[2]
    # batch_size=1, # currently no batching
    # num_images_per_prompt=1, # currently only single image
    device="cuda",
    dtype=torch.float16,
    do_classifier_free_guidance=True,
    control_mode="balanced",
    resize_mode="just_resize_simple",
):
    # FIXME: implement "crop_resize_simple" and "fill_resize_simple", or pull them out
    if (
        resize_mode == "just_resize_simple"
        or resize_mode == "crop_resize_simple"
        or resize_mode == "fill_resize_simple"
    ):
        image = image.convert("RGB")
        if resize_mode == "just_resize_simple":
            image = image.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
        elif resize_mode == "crop_resize_simple":  # not yet implemented
            pass
        elif resize_mode == "fill_resize_simple":  # not yet implemented
            pass
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
            # device=torch.device('cpu')
            device=device,
        )
    else:
        pass
        print("ERROR: invalid resize_mode ==> ", resize_mode)
        exit(1)

    timage = timage.to(device=device, dtype=dtype)
    cfg_injection = control_mode == "more_control" or control_mode == "unbalanced"
    if do_classifier_free_guidance and not cfg_injection:
        timage = torch.cat([timage] * 2)
    return timage
