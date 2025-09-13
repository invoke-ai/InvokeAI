from itertools import zip_longest
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field, model_validator
from transformers.models.sam import SamModel
from transformers.models.sam.processing_sam import SamProcessor
from transformers.models.sam2 import Sam2Model
from transformers.models.sam2.processing_sam2 import Sam2Processor

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import BoundingBoxField, ImageField, InputField, TensorField
from invokeai.app.invocations.primitives import MaskOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.segment_anything.mask_refinement import mask_to_polygon, polygon_to_mask
from invokeai.backend.image_util.segment_anything.segment_anything_2_pipeline import SegmentAnything2Pipeline
from invokeai.backend.image_util.segment_anything.segment_anything_pipeline import SegmentAnythingPipeline
from invokeai.backend.image_util.segment_anything.shared import SAMInput, SAMPoint

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


class SAMPointsField(BaseModel):
    points: list[SAMPoint] = Field(..., description="The points of the object", min_length=1)

    def to_list(self) -> list[list[float]]:
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

    @model_validator(mode="after")
    def validate_points_and_boxes_len(self):
        if self.point_lists is not None and self.bounding_boxes is not None:
            if len(self.point_lists) != len(self.bounding_boxes):
                raise ValueError("If both point_lists and bounding_boxes are provided, they must have the same length.")
        return self

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> MaskOutput:
        # The models expect a 3-channel RGB image.
        image_pil = context.images.get_pil(self.image.image_name, mode="RGB")

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
        sam_processor = SamProcessor.from_pretrained(model_path, local_files_only=True)
        return SegmentAnythingPipeline(sam_model=sam_model, sam_processor=sam_processor)

    @staticmethod
    def _load_sam_2_model(model_path: Path):
        sam2_model = Sam2Model.from_pretrained(model_path, local_files_only=True)
        sam2_processor = Sam2Processor.from_pretrained(model_path, local_files_only=True)
        return SegmentAnything2Pipeline(sam2_model=sam2_model, sam2_processor=sam2_processor)

    def _segment(self, context: InvocationContext, image: Image.Image) -> list[torch.Tensor]:
        """Use Segment Anything (SAM or SAM2) to generate masks given an image + a set of bounding boxes."""

        source = SEGMENT_ANYTHING_MODEL_IDS[self.model]
        inputs: list[SAMInput] = []
        for bbox_field, point_field in zip_longest(self.bounding_boxes or [], self.point_lists or [], fillvalue=None):
            inputs.append(
                SAMInput(
                    bounding_box=bbox_field,
                    points=point_field.points if point_field else None,
                )
            )

        if "sam2" in source:
            loader = SegmentAnythingInvocation._load_sam_2_model
            with context.models.load_remote_model(source=source, loader=loader) as pipeline:
                assert isinstance(pipeline, SegmentAnything2Pipeline)
                masks = pipeline.segment(image=image, inputs=inputs)
        else:
            loader = SegmentAnythingInvocation._load_sam_model
            with context.models.load_remote_model(source=source, loader=loader) as pipeline:
                assert isinstance(pipeline, SegmentAnythingPipeline)
                masks = pipeline.segment(image=image, inputs=inputs)

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
