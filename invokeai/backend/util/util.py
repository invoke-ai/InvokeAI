import base64
import importlib
import io
import math
import os
import re
from inspect import isfunction
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import invokeai.backend.util.logging as logger

from .devices import torch_dtype

# actual size of a gig
GIG = 1073741824


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.load_default()
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start : start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            logger.warning("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        logger.debug(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def rand_perlin_2d(shape, res, device, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, res[0], delta[0]),
                torch.arange(0, res[1], delta[1]),
                indexing="ij",
            ),
            dim=-1,
        ).to(device)
        % 1
    )

    rand_val = torch.rand(res[0] + 1, res[1] + 1)

    angles = 2 * math.pi * rand_val
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).to(device)

    tile_grads = (
        lambda slice1, slice2: gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
        .repeat_interleave(d[0], 0)
        .repeat_interleave(d[1], 1)
    )

    def dot(grad, shift):
        return (
            torch.stack(
                (
                    grid[: shape[0], : shape[1], 0] + shift[0],
                    grid[: shape[0], : shape[1], 1] + shift[1],
                ),
                dim=-1,
            )
            * grad[: shape[0], : shape[1]]
        ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0]).to(device)
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0]).to(device)
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1]).to(device)
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1]).to(device)
    t = fade(grid[: shape[0], : shape[1]])
    noise = math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1]).to(
        device
    )
    return noise.to(dtype=torch_dtype(device))


def ask_user(question: str, answers: list):
    from itertools import chain, repeat

    user_prompt = f"\n>> {question} {answers}: "
    invalid_answer_msg = "Invalid answer. Please try again."
    pose_question = chain([user_prompt], repeat("\n".join([invalid_answer_msg, user_prompt])))
    user_answers = map(input, pose_question)
    valid_response = next(filter(answers.__contains__, user_answers))
    return valid_response


# -------------------------------------
def download_with_resume(url: str, dest: Path, access_token: str = None) -> Optional[Path]:
    """
    Download a model file.
    :param url:  https, http or ftp URL
    :param dest: A Path object. If path exists and is a directory, then we try to derive the filename
                 from the URL's Content-Disposition header and copy the URL contents into
                 dest/filename
    :param access_token: Access token to access this resource
    """
    header = {"Authorization": f"Bearer {access_token}"} if access_token else {}
    open_mode = "wb"
    exist_size = 0

    resp = requests.get(url, header, stream=True)
    content_length = int(resp.headers.get("content-length", 0))

    if dest.is_dir():
        file_name = response_attachment(resp) or os.path.basename(url)
        dest = dest / file_name
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        exist_size = dest.stat().st_size
        header["Range"] = f"bytes={exist_size}-"
        open_mode = "ab"
        resp = requests.get(url, headers=header, stream=True)  # new request with range

    if exist_size > content_length:
        logger.warning("corrupt existing file found. re-downloading")
        os.remove(dest)
        exist_size = 0

    if resp.status_code == 416 or (content_length > 0 and exist_size == content_length):
        logger.warning(f"{dest}: complete file found. Skipping.")
        return dest
    elif resp.status_code == 206 or exist_size > 0:
        logger.warning(f"{dest}: partial file found. Resuming...")
    elif resp.status_code != 200:
        logger.error(f"An error occurred during downloading {dest}: {resp.reason}")
    else:
        logger.info(f"{dest}: Downloading...")

    try:
        if content_length < 2000:
            logger.error(f"ERROR DOWNLOADING {url}: {resp.text}")
            return None

        with open(dest, open_mode) as file, tqdm(
            desc=str(dest),
            initial=exist_size,
            total=content_length,
            unit="iB",
            unit_scale=True,
            unit_divisor=1000,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    except Exception as e:
        logger.error(f"An error occurred while downloading {dest}: {str(e)}")
        return None

    return dest


def response_attachment(response: requests.Response) -> Optional[str]:
    try:
        if disposition := response.headers.get("Content-Disposition"):
            if match := re.search('filename="(.+)"', disposition):
                return match.group(1)
        return None
    except Exception:
        return None


def url_attachment_name(url: str) -> Optional[str]:
    resp = requests.get(url)
    if resp.ok:
        return response_attachment(resp)
    else:
        return None


def download_with_progress_bar(url: str, dest: Path) -> bool:
    result = download_with_resume(url, dest, access_token=None)
    return result is not None


def image_to_dataURL(image: Image.Image, image_format: str = "PNG") -> str:
    """
    Converts an image into a base64 image dataURL.
    """
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)
    mime_type = Image.MIME.get(image_format.upper(), "image/" + image_format.lower())
    image_base64 = f"data:{mime_type};base64," + base64.b64encode(buffered.getvalue()).decode("UTF-8")
    return image_base64


def directory_size(directory: Path) -> int:
    """
    Returns the aggregate size of all files in a directory (bytes).
    """
    sum = 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            sum += Path(root, f).stat().st_size
        for d in dirs:
            sum += Path(root, d).stat().st_size
    return sum


class Chdir(object):
    """Context manager to chdir to desired directory and change back after context exits:
    Args:
        path (Path): The path to the cwd
    """

    def __init__(self, path: Path):
        self.path = path
        self.original = Path().absolute()

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, *args):
        os.chdir(self.original)
