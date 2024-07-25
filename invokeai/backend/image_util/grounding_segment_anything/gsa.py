import pathlib
from typing import Dict, List, Optional

import numpy as np
import supervision as sv
import torch
import torchvision
from PIL import Image

from invokeai.backend.image_util.grounding_segment_anything.groundingdino.util.inference import Model
from invokeai.backend.image_util.grounding_segment_anything.segment_anything.build_sam import sam_model_registry
from invokeai.backend.image_util.grounding_segment_anything.segment_anything.predictor import SamPredictor


class GroundingSegmentAnythingDetector:
    def __init__(self, grounding_dino_model: Model, segment_anything_model: SamPredictor) -> None:
        self.grounding_dino_model: Optional[Model] = grounding_dino_model
        self.segment_anything_model: Optional[SamPredictor] = segment_anything_model

    @staticmethod
    def build_grounding_dino(grounding_dino_state_dict: Dict[str, torch.Tensor]):
        grounding_dino_config = pathlib.Path(
            "./invokeai/backend/image_util/grounding_segment_anything/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        )
        return Model(
            model_state_dict=grounding_dino_state_dict,
            model_config_path=grounding_dino_config.as_posix(),
        )

    @staticmethod
    def build_segment_anything(segment_anything_state_dict: Dict[str, torch.Tensor], device: torch.device):
        sam = sam_model_registry["vit_h"](checkpoint=segment_anything_state_dict)
        sam.to(device=device)
        return SamPredictor(sam)

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
        image: Image.Image,
        prompt: str,
        box_threshold: float = 0.5,
        text_threshold: float = 0.5,
        nms_threshold: float = 0.8,
    ):
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        prompts = prompt.split(",")

        detections = self.detect_objects(open_cv_image, prompts, box_threshold, text_threshold, nms_threshold)
        segments = self.segment_detections(open_cv_image, detections, prompts)

        if len(segments) > 0:
            combined_mask = np.zeros_like(list(segments.values())[0])
            for mask in list(segments.values()):
                combined_mask = np.logical_or(combined_mask, mask)
            mask_preview = (combined_mask * 255).astype(np.uint8)
        else:
            mask_preview = np.zeros(open_cv_image.shape, np.uint8)

        return Image.fromarray(mask_preview)
