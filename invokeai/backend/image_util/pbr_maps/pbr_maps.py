# Original: https://github.com/joeyballentine/Material-Map-Generator
# Adopted and optimized for Invoke AI

import pathlib
from typing import Any, Literal

import cv2
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from invokeai.backend.image_util.pbr_maps.architecture.pbr_rrdb_net import PBR_RRDB_Net
from invokeai.backend.image_util.pbr_maps.utils.image_ops import crop_seamless, esrgan_launcher_split_merge

NORMAL_MAP_MODEL = "https://github.com/joeyballentine/Material-Map-Generator/blob/master/utils/models/1x_NormalMapGenerator-CX-Lite_200000_G.pth"
OTHER_MAP_MODEL = "https://github.com/joeyballentine/Material-Map-Generator/blob/master/utils/models/1x_FrankenMapGenerator-CX-Lite_215000_G.pth"


class PBRMapsGenerator:
    def __init__(self, normal_map_model: PBR_RRDB_Net, other_map_model: PBR_RRDB_Net, device: torch.device) -> None:
        self.normal_map_model = normal_map_model
        self.other_map_model = other_map_model
        self.device = device

    @staticmethod
    def load_model(model_path: pathlib.Path, device: torch.device) -> PBR_RRDB_Net:
        state_dict = torch.load(model_path.as_posix(), map_location="cpu")

        model = PBR_RRDB_Net(
            3,
            3,
            32,
            12,
            gc=32,
            upscale=1,
            norm_type=None,
            act_type="leakyrelu",
            mode="CNA",
            res_scale=1,
            upsample_mode="upconv",
        )

        model.load_state_dict(state_dict, strict=False)
        del state_dict
        model.eval()

        for _, v in model.named_parameters():
            v.requires_grad = False

        return model.to(device)

    def process(self, img: npt.NDArray[Any], model: PBR_RRDB_Net):
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
        img = img[..., ::-1].copy()
        tensor_img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = model(tensor_img).data.squeeze(0).float().cpu().clamp_(0, 1).numpy()
            output = output[[2, 1, 0], :, :]
            output = np.transpose(output, (1, 2, 0))
            output = (output * 255.0).round()
            return output

    def _cv2_to_pil(self, image: npt.NDArray[Any]):
        return Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))

    def generate_maps(
        self,
        image: Image.Image,
        tile_size: int = 512,
        border_mode: Literal["none", "seamless", "mirror", "replicate"] = "none",
    ):
        models = [self.normal_map_model, self.other_map_model]
        np_image = np.array(image).astype(np.uint8)

        match border_mode:
            case "seamless":
                np_image = cv2.copyMakeBorder(np_image, 16, 16, 16, 16, cv2.BORDER_WRAP)
            case "mirror":
                np_image = cv2.copyMakeBorder(np_image, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
            case "replicate":
                np_image = cv2.copyMakeBorder(np_image, 16, 16, 16, 16, cv2.BORDER_REPLICATE)
            case "none":
                pass

        img_height, img_width = np_image.shape[:2]

        # Checking whether to perform tiled inference
        do_split = img_height > tile_size or img_width > tile_size

        if do_split:
            rlts = esrgan_launcher_split_merge(np_image, self.process, models, scale_factor=1, tile_size=tile_size)
        else:
            rlts = [self.process(np_image, model) for model in models]

        if border_mode != "none":
            rlts = [crop_seamless(rlt) for rlt in rlts]

        normal_map = self._cv2_to_pil(rlts[0])
        roughness = self._cv2_to_pil(rlts[1][:, :, 1])
        displacement = self._cv2_to_pil(rlts[1][:, :, 0])

        return normal_map, roughness, displacement
