import torch
import numpy as np
from PIL import Image
from einops import rearrange

from invokeai.backend.flux.modules.autoencoder import AutoEncoder

def prepare_control(
    height: int,
    width: int,
    seed: int,
    ae: AutoEncoder,
    cond_image: Image.Image,
) -> torch.Tensor:
    # load and encode the conditioning image
    img_cond = cond_image.convert("RGB")
    img_cond = img_cond.resize((width, height), Image.Resampling.LANCZOS)
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float()
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")
    ae_dtype = next(iter(ae.parameters())).dtype
    ae_device = next(iter(ae.parameters())).device
    img_cond = img_cond.to(device=ae_device, dtype=ae_dtype)
    generator = torch.Generator(device=ae_device).manual_seed(seed)
    img_cond = ae.encode(img_cond, sample=True, generator=generator)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    return img_cond
