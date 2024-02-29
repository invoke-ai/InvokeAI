import pathlib
from typing import Any, List, Literal, Tuple

import numpy as np
from PIL import Image

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.backend.image_util.segment_anything import SamPredictor, sam_model_registry
from invokeai.backend.util.util import download_with_progress_bar

config = InvokeAIAppConfig.get_config()

SEGMENT_ANYTHING_MODELS = {
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "local": "any/annotators/segment_anything/sam_vit_h_4b8939.pth",
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "local": "any/annotators/segment_anything/sam_vit_l_0b3195.pth",
    },
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "local": "any/annotators/segment_anything/sam_vit_b_01ec64.pth",
    },
}

SEGMENT_ANYTHING_MODEL_TYPES = Literal["vit_h", "vit_l", "vit_b"]


class SAMImagePredictor:
    def __init__(self) -> None:
        self.sam = None
        self.predictor = None

    def load_model(self, model_type: SEGMENT_ANYTHING_MODEL_TYPES = "vit_h"):
        SEGMENT_ANYTHING_MODEL_PATH = pathlib.Path(config.models_path / SEGMENT_ANYTHING_MODELS[model_type]["local"])
        if not SEGMENT_ANYTHING_MODEL_PATH.exists():
            download_with_progress_bar(SEGMENT_ANYTHING_MODELS[model_type]["url"], SEGMENT_ANYTHING_MODEL_PATH)

        sam = sam_model_registry[model_type](checkpoint=SEGMENT_ANYTHING_MODEL_PATH)

        if not self.sam or self.sam != sam:
            self.sam = sam

        self.predictor = SamPredictor(self.sam)
        return self.predictor

    def __call__(
        self, image: Image.Image, background: bool = False, position: Tuple[int, int] = (0, 0), *args: Any, **kwds: Any
    ) -> Any:
        input_image = np.array(image.convert("RGB")) if image.mode != "RGB" else np.array(image)
        input_point = np.array([[position[0], position[1]]])
        input_label = np.array([0]) if background else np.array([1])

        if self.predictor:
            self.predictor.set_image(input_image)
            masks, _, _ = self.predictor.predict(input_point, input_label)
            mask = Image.fromarray(masks[0])
            return mask
        else:
            return Image.new("RGB", (image.width, image.height), color="black")
