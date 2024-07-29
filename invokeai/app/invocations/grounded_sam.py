from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
from invokeai.backend.grounded_sam.grounding_dino_pipeline import GroundingDinoPipeline
from invokeai.backend.grounded_sam.mask_refinement import mask_to_polygon, polygon_to_mask
from invokeai.backend.grounded_sam.segment_anything_model import SegmentAnythingModel

GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SEGMENT_ANYTHING_MODEL_ID = "facebook/sam-vit-base"


@dataclass
class BoundingBox:
    """Bounding box helper class used locally for the Grounding DINO outputs."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def to_box(self) -> list[int]:
        """Convert to the array notation expected by SAM."""
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    """Detection result from Grounding DINO or Grounded SAM."""

    score: float
    label: str
    box: BoundingBox
    mask: Optional[npt.NDArray[Any]] = None

    @classmethod
    def from_dict(cls, detection_dict: dict[str, Any]):
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


@invocation(
    "grounded_segment_anything",
    title="Segment Anything (Text Prompt)",
    tags=["prompt", "segmentation"],
    category="segmentation",
    version="1.0.0",
)
class GroundedSAMInvocation(BaseInvocation):
    """Runs Grounded-SAM, as proposed in https://arxiv.org/pdf/2401.14159.

    More specifically, a Grounding DINO model is run to obtain bounding boxes for a text prompt, then the bounding box
    is passed as a prompt to a Segment Anything model to obtain a segmentation mask.

    Reference:
    - https://huggingface.co/docs/transformers/v4.43.3/en/model_doc/grounding-dino#grounded-sam
    - https://github.com/NielsRogge/Transformers-Tutorials/blob/a39f33ac1557b02ebfb191ea7753e332b5ca933f/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb
    """

    prompt: str = InputField(description="The prompt describing the object to segment.")
    image: ImageField = InputField(description="The image to segment.")
    apply_polygon_refinement: bool = InputField(
        description="Whether to apply polygon refinement to the mask. This will smooth the edges of the mask slightly "
        "and ensure that the mask consists of a single closed polygon.",
        default=False,
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_pil = context.images.get_pil(self.image.image_name)

        detections = self._detect(context=context, image=image_pil, labels=[self.prompt])
        detections = self._segment(context=context, image=image_pil, detection_results=detections)

        # Extract ouput mask.
        mask_np = detections[0].mask
        assert mask_np is not None
        # Map [0, 1] to [0, 255].
        mask_np = mask_np * 255
        mask_pil = Image.fromarray(mask_np)

        image_dto = context.images.save(image=mask_pil)
        return ImageOutput.build(image_dto)

    def _to_box_array(self, detection_results: list[DetectionResult]) -> list[list[list[int]]]:
        """Convert a list of DetectionResults to the format expected by the Segment Anything model.

        Args:
            detection_results (list[DetectionResult]): The Grounding DINO detection results.
        """
        boxes = [result.box.to_box() for result in detection_results]
        return [boxes]

    def _detect(
        self,
        context: InvocationContext,
        image: Image.Image,
        labels: list[str],
        threshold: float = 0.3,
    ) -> list[DetectionResult]:
        """Use Grounding DINO to detect bounding boxes for a set of labels in an image."""

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

            # TODO(ryand): I copied this "."-handling logic from the transformers example code. Test it and see if it
            # actually makes a difference.
            labels = [label if label.endswith(".") else label + "." for label in labels]

            results = detector(image, candidate_labels=labels, threshold=threshold)
            results = [DetectionResult.from_dict(result) for result in results]
            return results

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

            boxes = self._to_box_array(detection_results)
            masks = sam_pipeline.segment(image=image, boxes=boxes)
            masks = self._refine_masks(masks)

            for detection_result, mask in zip(detection_results, masks, strict=False):
                detection_result.mask = mask

            return detection_results

    def _refine_masks(self, masks: torch.Tensor) -> list[npt.NDArray[np.uint8]]:
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)

        if self.apply_polygon_refinement:
            for idx, mask in enumerate(masks):
                shape = mask.shape
                polygon = mask_to_polygon(mask)
                mask = polygon_to_mask(polygon, shape)
                masks[idx] = mask

        return masks
