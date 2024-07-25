from typing import Dict, List, Literal, Optional

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision

from invokeai.backend.image_util.grounding_segment_anything.groundingdino.util.inference import Model
from invokeai.backend.image_util.grounding_segment_anything.segment_anything.build_sam import sam_model_registry
from invokeai.backend.image_util.grounding_segment_anything.segment_anything.predictor import SamPredictor

GROUNDING_SEGMENT_ANYTHING_MODELS = {
    "groundingdino_swint_ogc": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    "segment_anything_vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}


class GroundingSegmentAnythingDetector:
    def __init__(self) -> None:
        self.grounding_dino_model: Optional[Model] = None
        self.segment_anything_model: Optional[SamPredictor] = None
        self.grounding_dino_config: str = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.sam_encoder: Literal["vit_h"] = "vit_h"
        self.device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def build_grounding_dino(self):
        return Model(
            model_config_path=self.grounding_dino_config,
            model_checkpoint_path="./checkpoints/groundingdino_swint_ogc.pth",
        )

    def build_segment_anything(self):
        sam = sam_model_registry[self.sam_encoder](checkpoint="./checkpoints/sam_vit_h_4b8939.pth")
        sam.to(device=self.device)
        return SamPredictor(sam)

    def build_grounding_sam(self):
        self.grounding_dino_model = self.build_grounding_dino()
        self.segment_anything_model = self.build_segment_anything()

    def detect_objects(
        self,
        image: np.ndarray,
        prompts: List[str],
        box_threshold: float = 0.5,
        text_threshold: float = 0.5,
        nms_threshold: float = 0.8,
    ):
        detections = self.grounding_dino_model.predict_with_classes(
            image=image, classes=prompts, box_threshold=box_threshold, text_threshold=text_threshold
        )

        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy), torch.from_numpy(detections.confidence), nms_threshold
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        return detections

    def segment_detections(
        self, image: np.ndarray, detections: sv.Detections, prompts: List[str]
    ) -> Dict[str, np.ndarray]:
        self.segment_anything_model.set_image(image)
        result_masks = {}
        for box, class_id in zip(detections.xyxy, detections.class_id):
            masks, scores, logits = self.segment_anything_model.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            result_masks.update({prompts[class_id]: masks[index]})
        return result_masks

    def predict(
        self,
        image: str,
        prompt: str,
        box_threshold: float = 0.5,
        text_threshold: float = 0.5,
        nms_threshold: float = 0.8,
    ):
        if not self.grounding_dino_model or not self.segment_anything_model:
            self.build_grounding_sam()

        image = cv2.imread(image)
        prompts = prompt.split(",")

        detections = self.detect_objects(image, prompts, box_threshold, text_threshold, nms_threshold)
        segments = self.segment_detections(image, detections, prompts)

        if len(segments) > 0:
            combined_mask = np.zeros_like(list(segments.values())[0])
            for mask in list(segments.values()):
                combined_mask = np.logical_or(combined_mask, mask)
            mask_preview = (combined_mask * 255).astype(np.uint8)
        else:
            mask_preview = np.zeros(image.shape, np.uint8)

        cv2.imwrite("mask.png", mask_preview)


if __name__ == "__main__":
    gsa = GroundingSegmentAnythingDetector()
    image = "./assets/image.webp"

    while True:
        prompt = input("Segment: ")
        gsa.predict(image, prompt, 0.5, 0.5, 0.8)
