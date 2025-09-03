from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoProcessor

from transformers.models.sam2 import Sam2Model
from transformers.models.sam2.processing_sam2 import Sam2Processor

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import BoundingBoxField, ImageField, InputField, TensorField
from invokeai.app.invocations.primitives import MaskOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.segment_anything.mask_refinement import mask_to_polygon, polygon_to_mask

# Import the pipeline directly since SAM2 should be available in transformers 4.56.0+
from invokeai.backend.image_util.segment_anything.segment_anything_2_pipeline import SegmentAnything2Pipeline

SegmentAnything2ModelKey = Literal[
    "segment-anything-2-tiny", "segment-anything-2-base", "segment-anything-2-large", "segment-anything-2-huge"
]
SEGMENT_ANYTHING_2_MODEL_IDS: dict[SegmentAnything2ModelKey, str] = {
    "segment-anything-2-tiny": "danelcsb/sam2.1_hiera_tiny",
    "segment-anything-2-base": "facebook/sam2.1-hiera-base",
    "segment-anything-2-large": "facebook/sam2.1-hiera-large",
    "segment-anything-2-huge": "facebook/sam2.1-hiera-huge",
}


class SAM2PointLabel(Enum):
    negative = -1
    neutral = 0
    positive = 1


class SAM2Point(BaseModel):
    x: int = Field(..., description="The x-coordinate of the point")
    y: int = Field(..., description="The y-coordinate of the point")
    label: SAM2PointLabel = Field(
        ..., description="The label of the point (-1 for background, 0 for neutral, 1 for foreground)"
    )


class SAM2PointsField(BaseModel):
    points: list[SAM2Point] = Field(..., description="The list of points for this object")

    def to_list(self) -> list[list[int]]:
        return [[point.x, point.y, point.label.value] for point in self.points]


@invocation(
    "segment_anything_2",
    title="Segment Anything 2",
    tags=["prompt", "segmentation", "sam2"],
    category="segmentation",
    version="1.0.0",
)
class SegmentAnything2Invocation(BaseInvocation):
    """Runs a Segment Anything 2 Model (SAM2)."""

    # Reference:
    # - https://arxiv.org/pdf/2401.05948 (SAM2 paper)
    # - https://huggingface.co/docs/transformers/v4.56.0/en/model_doc/sam2
    # - https://github.com/huggingface/transformers/pull/32317

    model: SegmentAnything2ModelKey = InputField(description="The Segment Anything 2 model to use.")
    image: ImageField = InputField(description="The image to segment.")
    bounding_boxes: list[BoundingBoxField] | None = InputField(
        default=None, description="The bounding boxes to prompt the SAM2 model with."
    )
    point_lists: list[SAM2PointsField] | None = InputField(
        default=None,
        description="The list of point lists to prompt the SAM2 model with. Each list of points represents a single object.",
    )
    apply_polygon_refinement: bool = InputField(
        description="Whether to apply polygon refinement to the masks. This will smooth the edges of the masks slightly and ensure that each mask consists of a single closed polygon (before merging).",
        default=True,
    )
    mask_filter: Literal["all", "largest", "highest_box_score"] = InputField(
        description="The filtering to apply to the detected masks before merging them into a final output.",
        default="all",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> MaskOutput:
        # The models expect a 3-channel RGB image.
        image_pil = context.images.get_pil(self.image.image_name)
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        if self.point_lists is not None and self.bounding_boxes is not None:
            raise ValueError("Only one of point_lists or bounding_box can be provided.")

        if (not self.bounding_boxes or len(self.bounding_boxes) == 0) and (
            not self.point_lists or len(self.point_lists) == 0
        ):
            combined_mask = torch.zeros(image_pil.size[::-1], dtype=torch.bool)
        else:
            masks = self._segment(context=context, image=image_pil)
            masks = self._filter_masks(masks=masks, bounding_boxes=self.bounding_boxes)

            # masks contains bool values, so we merge them via max-reduce.
            combined_mask, _ = torch.stack(masks).max(dim=0)

        # Unsqueeze the channel dimension.
        combined_mask = combined_mask.unsqueeze(0)
        mask_tensor_name = context.tensors.save(combined_mask)
        _, height, width = combined_mask.shape
        return MaskOutput(mask=TensorField(tensor_name=mask_tensor_name), width=width, height=height)

    @staticmethod
    def _load_sam2_model(model_path: Path):
        """Load a SAM2 model and processor from the given path."""
        try:
            # Load the SAM2 model - SAM2 models are unified and can handle both images and videos
            sam2_model = Sam2Model.from_pretrained(
                model_path,
                local_files_only=True,
                # TODO: Investigate whether fp16 is supported by SAM2
                # torch_dtype=TorchDevice.choose_torch_dtype(),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM2 model from {model_path}. Error: {str(e)}")

        try:
            # Use AutoProcessor to automatically detect the correct processor type
            # SAM2 models can work with both image and video processors
            sam2_processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

            # Log what type of processor we got for debugging
            processor_type = type(sam2_processor).__name__
            print(f"Loaded processor type: {processor_type} for model {model_path}")

            # SAM2 models are unified and can work with different processor types
            # We don't need to enforce a specific processor type
        except Exception as e:
            raise RuntimeError(f"Failed to load processor from {model_path}. Error: {str(e)}")

        return SegmentAnything2Pipeline(sam2_model=sam2_model, sam2_processor=sam2_processor)

    def _segment(self, context: InvocationContext, image: Image.Image) -> list[torch.Tensor]:
        """Use Segment Anything 2 (SAM2) to generate masks given an image + a set of bounding boxes."""
        # Convert the bounding boxes to the SAM2 input format.
        sam2_bounding_boxes = (
            [[bb.x_min, bb.y_min, bb.x_max, bb.y_max] for bb in self.bounding_boxes] if self.bounding_boxes else None
        )

        # Convert points to SAM2's 4D format
        if self.point_lists:
            # SAM2 expects: [[[[x, y]]]] for points and [[[label]]] for labels
            # Each point_list represents one object, so we need to structure accordingly
            sam2_points = []
            sam2_labels = []

            for point_list in self.point_lists:
                # Convert each point to the format expected by SAM2
                object_points = []
                object_labels = []

                for point in point_list.points:
                    object_points.append([point.x, point.y])
                    object_labels.append(point.label.value)

                # SAM2 expects: [[[[x, y]]]] and [[[label]]]
                sam2_points.append([object_points])
                sam2_labels.append([object_labels])
        else:
            sam2_points = None
            sam2_labels = None

        with (
            context.models.load_remote_model(
                source=SEGMENT_ANYTHING_2_MODEL_IDS[self.model], loader=SegmentAnything2Invocation._load_sam2_model
            ) as sam2_pipeline,
        ):
            assert isinstance(sam2_pipeline, SegmentAnything2Pipeline)
            masks = sam2_pipeline.segment(
                image=image, bounding_boxes=sam2_bounding_boxes, point_lists=sam2_points, point_labels=sam2_labels
            )

        masks = self._process_masks(masks)
        if self.apply_polygon_refinement:
            masks = self._apply_polygon_refinement(masks)

        return masks

    def _process_masks(self, masks: torch.Tensor) -> list[torch.Tensor]:
        """Convert the tensor output from the Segment Anything 2 model from a tensor of shape
        [num_masks, channels, height, width] to a list of tensors of shape [height, width].
        """
        assert masks.dtype == torch.bool
        # [num_masks, channels, height, width] -> [num_masks, height, width]
        masks, _ = masks.max(dim=1)
        # Split the first dimension into a list of masks.
        return list(masks.cpu().unbind(dim=0))

    def _apply_polygon_refinement(self, masks: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply polygon refinement to the masks.

        Convert each mask to a polygon, then back to a mask. This has the following effect:
        - Smooth the edges of the mask slightly.
        - Ensure that each mask consists of a single closed polygon
            - Removes small mask pieces.
            - Removes holes from the mask.
        """
        # Convert tensor masks to np masks.
        np_masks = [mask.cpu().numpy().astype(np.uint8) for mask in masks]

        # Apply polygon refinement.
        for idx, mask in enumerate(np_masks):
            shape = mask.shape
            assert len(shape) == 2  # Assert length to satisfy type checker.
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            np_masks[idx] = mask

        # Convert np masks back to tensor masks.
        masks = [torch.tensor(mask, dtype=torch.bool) for mask in np_masks]

        return masks

    def _filter_masks(
        self, masks: list[torch.Tensor], bounding_boxes: list[BoundingBoxField] | None
    ) -> list[torch.Tensor]:
        """Filter the detected masks based on the specified mask filter."""

        if self.mask_filter == "all":
            return masks
        elif self.mask_filter == "largest":
            # Find the largest mask.
            return [max(masks, key=lambda x: float(x.sum()))]
        elif self.mask_filter == "highest_box_score":
            assert bounding_boxes is not None, (
                "Bounding boxes must be provided to use the 'highest_box_score' mask filter."
            )
            assert len(masks) == len(bounding_boxes)
            # Find the index of the bounding box with the highest score.
            # Note that we fallback to -1.0 if the score is None. This is mainly to satisfy the type checker. In most
            # cases the scores should all be non-None when using this filtering mode. That being said, -1.0 is a
            # reasonable fallback since the expected score range is [0.0, 1.0].
            max_score_idx = max(range(len(bounding_boxes)), key=lambda i: bounding_boxes[i].score or -1.0)
            return [masks[max_score_idx]]
        else:
            raise ValueError(f"Invalid mask filter: {self.mask_filter}")
