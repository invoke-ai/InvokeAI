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
from transformers.models.sam2 import Sam2Model

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import BoundingBoxField, ImageField, InputField, TensorField
from invokeai.app.invocations.primitives import MaskOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.segment_anything.mask_refinement import mask_to_polygon, polygon_to_mask
from invokeai.backend.image_util.segment_anything.segment_anything_2_pipeline import SegmentAnything2Pipeline
from invokeai.backend.image_util.segment_anything.segment_anything_pipeline import SegmentAnythingPipeline

SegmentAnythingModelKey = Literal[
    "segment-anything-base",
    "segment-anything-large",
    "segment-anything-huge",
    "segment-anything-2-tiny",
    "segment-anything-2-small",
    "segment-anything-2-base",
    "segment-anything-2-large",
]
SEGMENT_ANYTHING_MODEL_IDS: dict[SegmentAnythingModelKey, str] = {
    "segment-anything-base": "facebook/sam-vit-base",
    "segment-anything-large": "facebook/sam-vit-large",
    "segment-anything-huge": "facebook/sam-vit-huge",
    "segment-anything-2-tiny": "facebook/sam2.1-hiera-tiny",
    "segment-anything-2-small": "facebook/sam2.1-hiera-small",
    "segment-anything-2-base": "facebook/sam2.1-hiera-base-plus",
    "segment-anything-2-large": "facebook/sam2.1-hiera-large",
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
    tags=["prompt", "segmentation", "sam", "sam2"],
    category="segmentation",
    version="1.3.0",
)
class SegmentAnythingInvocation(BaseInvocation):
    """Runs a Segment Anything Model (SAM or SAM2)."""

    # Reference:
    # - https://arxiv.org/pdf/2304.02643
    # - https://huggingface.co/docs/transformers/v4.43.3/en/model_doc/grounding-dino#grounded-sam
    # - https://github.com/NielsRogge/Transformers-Tutorials/blob/a39f33ac1557b02ebfb191ea7753e332b5ca933f/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb

    model: SegmentAnythingModelKey = InputField(description="The Segment Anything model to use (SAM or SAM2).")
    image: ImageField = InputField(description="The image to segment.")
    bounding_boxes: list[BoundingBoxField] | None = InputField(
        default=None, description="The bounding boxes to prompt the model with."
    )
    point_lists: list[SAMPointsField] | None = InputField(
        default=None,
        description="The list of point lists to prompt the model with. Each list of points represents a single object.",
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
        """Load either SAM or SAM2 model based on the model path."""
        model_path_str = str(model_path).lower()

        if "sam2" in model_path_str:
            # Load SAM2 model
            try:
                sam2_model = Sam2Model.from_pretrained(
                    model_path,
                    local_files_only=True,
                    # TODO: Investigate whether fp16 is supported by SAM2
                    # torch_dtype=TorchDevice.choose_torch_dtype(),
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load SAM2 model from {model_path}. Error: {str(e)}")

            try:
                sam2_processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
                # Log what type of processor we got for debugging
                processor_type = type(sam2_processor).__name__
                print(f"Loaded processor type: {processor_type} for model {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load processor from {model_path}. Error: {str(e)}")

            return SegmentAnything2Pipeline(sam2_model=sam2_model, sam2_processor=sam2_processor)
        else:
            # Load SAM model
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
        """Use Segment Anything (SAM or SAM2) to generate masks given an image + a set of bounding boxes."""
        # Convert the bounding boxes to the input format.
        bounding_boxes = (
            [[bb.x_min, bb.y_min, bb.x_max, bb.y_max] for bb in self.bounding_boxes] if self.bounding_boxes else None
        )

        # Convert points to the format expected by the specific model
        # We'll determine the format based on the actual pipeline type after loading
        if self.point_lists:
            # Prepare both formats - we'll use the appropriate one based on pipeline type
            # SAM2 format: [[[[x, y]]]] and [[[label]]]
            sam2_point_lists = []
            sam2_point_labels = []
            for point_list in self.point_lists:
                object_points = []
                object_labels = []
                for point in point_list.points:
                    object_points.append([point.x, point.y])
                    object_labels.append(point.label.value)
                sam2_point_lists.append([object_points])
                sam2_point_labels.append([object_labels])

            # SAM format: [[x, y, label]]
            sam_point_lists = [p.to_list() for p in self.point_lists]
        else:
            sam2_point_lists = None
            sam2_point_labels = None
            sam_point_lists = None

        with (
            context.models.load_remote_model(
                source=SEGMENT_ANYTHING_MODEL_IDS[self.model], loader=SegmentAnythingInvocation._load_sam_model
            ) as pipeline,
        ):
            # Check pipeline type dynamically and use appropriate point format
            if isinstance(pipeline, SegmentAnything2Pipeline):
                masks = pipeline.segment(
                    image=image,
                    bounding_boxes=bounding_boxes,
                    point_lists=sam2_point_lists,
                    point_labels=sam2_point_labels,
                )
            elif isinstance(pipeline, SegmentAnythingPipeline):
                masks = pipeline.segment(image=image, bounding_boxes=bounding_boxes, point_lists=sam_point_lists)
            else:
                raise RuntimeError(f"Unknown pipeline type: {type(pipeline)}")

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
