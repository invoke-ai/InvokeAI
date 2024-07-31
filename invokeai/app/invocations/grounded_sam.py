from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from transformers.models.sam import SamModel
from transformers.models.sam.processing_sam import SamProcessor
from transformers.pipelines import ZeroShotObjectDetectionPipeline

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.grounded_sam.detection_result import DetectionResult
from invokeai.backend.grounded_sam.grounding_dino_pipeline import GroundingDinoPipeline
from invokeai.backend.grounded_sam.mask_refinement import mask_to_polygon, polygon_to_mask
from invokeai.backend.grounded_sam.segment_anything_model import SegmentAnythingModel

GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SEGMENT_ANYTHING_MODEL_ID = "facebook/sam-vit-base"


@invocation(
    "grounded_segment_anything",
    title="Segment Anything (Text Prompt)",
    tags=["prompt", "segmentation"],
    category="segmentation",
    version="1.0.0",
)
class GroundedSAMInvocation(BaseInvocation):
    """Runs Grounded-SAM, as proposed in https://arxiv.org/pdf/2401.14159.

    More specifically, a Grounding DINO model is run to obtain bounding boxes for a text prompt, then the bounding boxes
    are passed as a prompt to a Segment Anything model to obtain a segmentation mask.

    Reference:
    - https://huggingface.co/docs/transformers/v4.43.3/en/model_doc/grounding-dino#grounded-sam
    - https://github.com/NielsRogge/Transformers-Tutorials/blob/a39f33ac1557b02ebfb191ea7753e332b5ca933f/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb
    """

    prompt: str = InputField(description="The prompt describing the object to segment.")
    image: ImageField = InputField(description="The image to segment.")
    apply_polygon_refinement: bool = InputField(
        description="Whether to apply polygon refinement to the masks. This will smooth the edges of the mask slightly and ensure that each mask consists of a single closed polygon (before merging).",
        default=True,
    )
    mask_filter: Literal["all", "largest", "highest_box_score"] = InputField(
        description="The filtering to apply to the detected masks before merging them into a final output.",
        default="all",
    )
    detection_threshold: float = InputField(
        description="The detection threshold for the Grounding DINO model. All detected bounding boxes with scores above this threshold will be used.",
        ge=0.0,
        le=1.0,
        default=0.3,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        # The models expect a 3-channel RGB image.
        image_pil = context.images.get_pil(self.image.image_name, mode="RGB")

        detections = self._detect(
            context=context, image=image_pil, labels=[self.prompt], threshold=self.detection_threshold
        )

        if len(detections) == 0:
            combined_mask = np.zeros(image_pil.size[::-1], dtype=np.uint8)
        else:
            detections = self._segment(context=context, image=image_pil, detection_results=detections)

            detections = self._filter_detections(detections)
            masks = [detection.mask for detection in detections]
            combined_mask = self._merge_masks(masks)

        # Map [0, 1] to [0, 255].
        mask_np = combined_mask * 255
        mask_pil = Image.fromarray(mask_np)

        image_dto = context.images.save(image=mask_pil)
        return ImageOutput.build(image_dto)

    def _detect(
        self,
        context: InvocationContext,
        image: Image.Image,
        labels: list[str],
        threshold: float = 0.3,
    ) -> list[DetectionResult]:
        """Use Grounding DINO to detect bounding boxes for a set of labels in an image."""
        # TODO(ryand): I copied this "."-handling logic from the transformers example code. Test it and see if it
        # actually makes a difference.
        labels = [label if label.endswith(".") else label + "." for label in labels]

        def load_grounding_dino(model_path: Path):
            grounding_dino_pipeline = pipeline(
                model=str(model_path),
                task="zero-shot-object-detection",
                local_files_only=True,
                # TODO(ryand): Setting the torch_dtype here doesn't work. Investigate whether fp16 is supported by the
                # model, and figure out how to make it work in the pipeline.
                # torch_dtype=TorchDevice.choose_torch_dtype(),
            )
            assert isinstance(grounding_dino_pipeline, ZeroShotObjectDetectionPipeline)
            return GroundingDinoPipeline(grounding_dino_pipeline)

        with context.models.load_remote_model(source=GROUNDING_DINO_MODEL_ID, loader=load_grounding_dino) as detector:
            assert isinstance(detector, GroundingDinoPipeline)
            return detector.detect(image=image, candidate_labels=labels, threshold=threshold)

    def _segment(
        self,
        context: InvocationContext,
        image: Image.Image,
        detection_results: list[DetectionResult],
    ) -> list[DetectionResult]:
        """Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes."""

        def load_sam_model(model_path: Path):
            sam_model = AutoModelForMaskGeneration.from_pretrained(
                model_path,
                local_files_only=True,
                # TODO(ryand): Setting the torch_dtype here doesn't work. Investigate whether fp16 is supported by the
                # model, and figure out how to make it work in the pipeline.
                # torch_dtype=TorchDevice.choose_torch_dtype(),
            )
            assert isinstance(sam_model, SamModel)

            sam_processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            assert isinstance(sam_processor, SamProcessor)
            return SegmentAnythingModel(sam_model=sam_model, sam_processor=sam_processor)

        with (
            context.models.load_remote_model(source=SEGMENT_ANYTHING_MODEL_ID, loader=load_sam_model) as sam_pipeline,
        ):
            assert isinstance(sam_pipeline, SegmentAnythingModel)

            masks = sam_pipeline.segment(image=image, detection_results=detection_results)

        masks = self._to_numpy_masks(masks)
        if self.apply_polygon_refinement:
            masks = self._apply_polygon_refinement(masks)

        for detection_result, mask in zip(detection_results, masks, strict=True):
            detection_result.mask = mask

        return detection_results

    def _to_numpy_masks(self, masks: torch.Tensor) -> list[npt.NDArray[np.uint8]]:
        """Convert the tensor output from the Segment Anything model to a list of numpy masks."""
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(dim=-1)
        masks = (masks > 0).int()
        np_masks = masks.numpy().astype(np.uint8)
        return list(np_masks)

    def _apply_polygon_refinement(self, masks: list[npt.NDArray[np.uint8]]) -> list[npt.NDArray[np.uint8]]:
        """Apply polygon refinement to the masks.

        Convert each mask to a polygon, then back to a mask. This has the following effect:
        - Smooth the edges of the mask slightly.
        - Ensure that each mask consists of a single closed polygon
            - Removes small mask pieces.
            - Removes holes from the mask.
        """
        for idx, mask in enumerate(masks):
            shape = mask.shape
            assert len(shape) == 2  # Assert length to satisfy type checker.
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

        return masks

    def _filter_detections(self, detections: list[DetectionResult]) -> list[DetectionResult]:
        """Filter the detected masks based on the specified mask filter."""
        if self.mask_filter == "all":
            return detections
        elif self.mask_filter == "largest":
            # Find the largest mask.
            return [max(detections, key=lambda x: x.mask.sum())]
        elif self.mask_filter == "highest_box_score":
            # Find the detection with the highest box score.
            return [max(detections, key=lambda x: x.score)]
        else:
            raise ValueError(f"Invalid mask filter: {self.mask_filter}")

    def _merge_masks(self, masks: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.uint8]:
        """Merge multiple masks into a single mask."""
        # Merge all masks together.
        stacked_mask = np.stack(masks, axis=0)
        combined_mask = np.max(stacked_mask, axis=0)
        return combined_mask
