import torch
import numpy as np
from PIL import Image
from einops import rearrange

from invokeai.backend.flux.modules.autoencoder import AutoEncoder

def prepare_control(
    img: torch.Tensor,
    ae: AutoEncoder,
    cond_image: Image.Image,
) -> torch.Tensor:
    # load and encode the conditioning image
    _, h, w = img.shape
    img_cond = cond_image.convert("RGB")
    width = w * 8
    height = h * 8
    img_cond = img_cond.resize((width, height), Image.Resampling.LANCZOS)
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")
    img_cond = img_cond.to(dtype=img.dtype, device=img.device)
    img_cond = ae.encode(img_cond)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    return img_cond
