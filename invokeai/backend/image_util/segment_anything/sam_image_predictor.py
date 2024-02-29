import pathlib
from typing import Literal, Tuple

import numpy as np
from PIL import Image, ImageOps

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.backend.image_util.segment_anything import SamPredictor, sam_model_registry
from invokeai.backend.util.util import download_with_progress_bar

config = InvokeAIAppConfig.get_config()

SEGMENT_ANYTHING_MODELS = {
    "small": {
        "model_type": "vit_b",
        "url": "https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_b_01ec64.pth",
        "local": "any/annotators/segment_anything/sam_vit_b_01ec64.pth",
    },
    "medium": {
        "model_type": "vit_l",
        "url": "https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_l_0b3195.pth",
        "local": "any/annotators/segment_anything/sam_vit_l_0b3195.pth",
    },
    "large": {
        "model_type": "vit_h",
        "url": "https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth",
        "local": "any/annotators/segment_anything/sam_vit_h_4b8939.pth",
    },
    "small_hq": {
        "model_type": "vit_b",
        "url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth",
        "local": "any/annotators/segment_anything/sam_hq_vit_b.pth",
    },
    "medium_hq": {
        "model_type": "vit_l",
        "url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth",
        "local": "any/annotators/segment_anything/sam_hq_vit_l.pth",
    },
    "large_hq": {
        "model_type": "vit_h",
        "url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth",
        "local": "any/annotators/segment_anything/sam_hq_vit_h.pth",
    },
    "mobile": {
        "model_type": "vit_tiny",
        "url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth",
        "local": "any/annotators/segment_anything/sam_hq_vit_tiny.pth",
    },
}

SEGMENT_ANYTHING_MODEL_TYPES = Literal["small", "medium", "large", "small_hq", "medium_hq", "large_hq", "mobile"]
sam_model_type: SEGMENT_ANYTHING_MODEL_TYPES | None = None
sam_model = None
sam_predictor = None


class SAMImagePredictor:
    def __init__(self) -> None:
        pass

    def load_model(self, model_type: SEGMENT_ANYTHING_MODEL_TYPES = "small"):
        global sam_model, sam_model_type, sam_predictor

        SEGMENT_ANYTHING_MODEL_PATH = pathlib.Path(config.models_path / SEGMENT_ANYTHING_MODELS[model_type]["local"])
        if not SEGMENT_ANYTHING_MODEL_PATH.exists():
            download_with_progress_bar(SEGMENT_ANYTHING_MODELS[model_type]["url"], SEGMENT_ANYTHING_MODEL_PATH)

        if not sam_model or not sam_predictor or sam_model_type != model_type:
            sam_model_type = model_type
            sam_model = sam_model_registry[SEGMENT_ANYTHING_MODELS[model_type]["model_type"]](
                checkpoint=SEGMENT_ANYTHING_MODEL_PATH
            )
            sam_predictor = sam_predictor = SamPredictor(sam_model)

    def __call__(
        self, image: Image.Image, background: bool = False, position: Tuple[int, int] = (0, 0), invert: bool = False
    ) -> Image.Image:
        global sam_predictor

        input_image = np.array(image.convert("RGB")) if image.mode != "RGB" else np.array(image)
        input_point = np.array([[position[0], position[1]]])
        input_label = np.array([0]) if background else np.array([1])

        if sam_predictor:
            sam_predictor.set_image(input_image)
            masks, _, _ = sam_predictor.predict(input_point, input_label)
            mask = Image.fromarray(masks[0]).convert("RGB")
            if invert:
                mask = ImageOps.invert(mask)
            return mask
        else:
            return Image.new("RGB", (image.width, image.height), color="black")
