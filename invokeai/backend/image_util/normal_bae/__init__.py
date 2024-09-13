# Adapted from https://github.com/huggingface/controlnet_aux

import pathlib
import types

import cv2
import huggingface_hub
import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image

from invokeai.backend.image_util.normal_bae.nets.NNET import NNET
from invokeai.backend.image_util.util import np_to_pil, pil_to_np, resize_to_multiple


class NormalMapDetector:
    """Simple wrapper around the Normal BAE model for normal map generation."""

    hf_repo_id = "lllyasviel/Annotators"
    hf_filename = "scannet.pt"

    @classmethod
    def get_model_url(cls) -> str:
        """Get the URL to download the model from the Hugging Face Hub."""
        return huggingface_hub.hf_hub_url(cls.hf_repo_id, cls.hf_filename)

    @classmethod
    def load_model(cls, model_path: pathlib.Path) -> NNET:
        """Load the model from a file."""

        args = types.SimpleNamespace()
        args.mode = "client"
        args.architecture = "BN"
        args.pretrained = "scannet"
        args.sampling_ratio = 0.4
        args.importance_ratio = 0.7

        model = NNET(args)

        ckpt = torch.load(model_path, map_location="cpu")["model"]
        load_dict = {}
        for k, v in ckpt.items():
            if k.startswith("module."):
                k_ = k.replace("module.", "")
                load_dict[k_] = v
            else:
                load_dict[k] = v

        model.load_state_dict(load_dict)
        model.eval()

        return model

    def __init__(self, model: NNET) -> None:
        self.model = model
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def run(self, image: Image.Image):
        """Processes an image and returns the detected normal map."""

        device = next(iter(self.model.parameters())).device
        np_image = pil_to_np(image)

        height, width, _channels = np_image.shape

        # The model requires the image to be a multiple of 8
        np_image = resize_to_multiple(np_image, 8)

        image_normal = np_image

        with torch.no_grad():
            image_normal = torch.from_numpy(image_normal).float().to(device)
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, "h w c -> 1 c h w")
            image_normal = self.norm(image_normal)

            normal = self.model(image_normal)
            normal = normal[0][-1][:, :3]
            normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], "c h w -> h w c").cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

        # Back to the original size
        output_image = cv2.resize(normal_image, (width, height), interpolation=cv2.INTER_LINEAR)

        return np_to_pil(output_image)
