from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoProcessor
from transformers.models.sam import SamModel
from transformers.models.sam.processing_sam import SamProcessor

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import BoundingBoxField, ImageField, InputField, TensorField
from invokeai.app.invocations.primitives import MaskOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.segment_anything.mask_refinement import mask_to_polygon, polygon_to_mask
from invokeai.backend.image_util.segment_anything.segment_anything_pipeline import SegmentAnythingPipeline

SegmentAnythingModelKey = Literal["segment-anything-base", "segment-anything-large", "segment-anything-huge"]
SEGMENT_ANYTHING_MODEL_IDS: dict[SegmentAnythingModelKey, str] = {
    "segment-anything-base": "facebook/sam-vit-base",
    "segment-anything-large": "facebook/sam-vit-large",
    "segment-anything-huge": "facebook/sam-vit-huge",
}


class SAMPointLabel(Enum):
    negative = -1
    neutral = 0
    positive = 1


class SAMPoint(BaseModel):
    x: int = Field(..., description="The x-coordinate of the point")
    y: int = Field(..., description="The y-coordinate of the point")
    label: SAMPointLabel = Field(..., description="The label of the point")


class SAMPointsField(BaseModel):
    points: list[SAMPoint] = Field(..., description="The points of the object")

    def to_list(self) -> list[list[int]]:
        return [[point.x, point.y, point.label.value] for point in self.points]


@invocation(
    "segment_anything",
    title="Segment Anything",
    tags=["prompt", "segmentation"],
    category="segmentation",
    version="1.2.0",
)
class SegmentAnythingInvocation(BaseInvocation):
    """Runs a Segment Anything Model."""

    # Reference:
    # - https://arxiv.org/pdf/2304.02643
    # - https://huggingface.co/docs/transformers/v4.43.3/en/model_doc/grounding-dino#grounded-sam
    # - https://github.com/NielsRogge/Transformers-Tutorials/blob/a39f33ac1557b02ebfb191ea7753e332b5ca933f/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb

    model: SegmentAnythingModelKey = InputField(description="The Segment Anything model to use.")
    image: ImageField = InputField(description="The image to segment.")
    bounding_boxes: list[BoundingBoxField] | None = InputField(
        default=None, description="The bounding boxes to prompt the SAM model with."
    )
    point_lists: list[SAMPointsField] | None = InputField(
        default=None,
        description="The list of point lists to prompt the SAM model with. Each list of points represents a single object.",
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
        image_pil = context.images.get_pil(self.image.image_name, mode="RGB")

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
    def _load_sam_model(model_path: Path):
        sam_model = SamModel.from_pretrained(
            model_path,
            local_files_only=True,
            # TODO(ryand): Setting the torch_dtype here doesn't work. Investigate whether fp16 is supported by the
            # model, and figure out how to make it work in the pipeline.
            # torch_dtype=TorchDevice.choose_torch_dtype(),
        )

        sam_processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        assert isinstance(sam_processor, SamProcessor)
        return SegmentAnythingPipeline(sam_model=sam_model, sam_processor=sam_processor)

    def _segment(self, context: InvocationContext, image: Image.Image) -> list[torch.Tensor]:
        """Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes."""
        # Convert the bounding boxes to the SAM input format.
        sam_bounding_boxes = (
            [[bb.x_min, bb.y_min, bb.x_max, bb.y_max] for bb in self.bounding_boxes] if self.bounding_boxes else None
        )
        sam_points = [p.to_list() for p in self.point_lists] if self.point_lists else None

        with (
            context.models.load_remote_model(
                source=SEGMENT_ANYTHING_MODEL_IDS[self.model], loader=SegmentAnythingInvocation._load_sam_model
            ) as sam_pipeline,
        ):
            assert isinstance(sam_pipeline, SegmentAnythingPipeline)
            masks = sam_pipeline.segment(image=image, bounding_boxes=sam_bounding_boxes, point_lists=sam_points)

        masks = self._process_masks(masks)
        if self.apply_polygon_refinement:
            masks = self._apply_polygon_refinement(masks)

        return masks

    def _process_masks(self, masks: torch.Tensor) -> list[torch.Tensor]:
        """Convert the tensor output from the Segment Anything model from a tensor of shape
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
