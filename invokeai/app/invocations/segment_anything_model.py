from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor
from transformers.models.sam import SamModel
from transformers.models.sam.processing_sam import SamProcessor

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import BoundingBoxField, ImageField, InputField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.segment_anything.mask_refinement import mask_to_polygon, polygon_to_mask
from invokeai.backend.image_util.segment_anything.segment_anything_model import SegmentAnythingModel

SEGMENT_ANYTHING_MODEL_ID = "facebook/sam-vit-base"


@invocation(
    "segment_anything_model",
    title="Segment Anything Model",
    tags=["prompt", "segmentation"],
    category="segmentation",
    version="1.0.0",
)
class SegmentAnythingModelInvocation(BaseInvocation):
    """Runs a Segment Anything Model (https://arxiv.org/pdf/2304.02643).

    Reference:
    - https://huggingface.co/docs/transformers/v4.43.3/en/model_doc/grounding-dino#grounded-sam
    - https://github.com/NielsRogge/Transformers-Tutorials/blob/a39f33ac1557b02ebfb191ea7753e332b5ca933f/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb
    """

    image: ImageField = InputField(description="The image to segment.")
    bounding_boxes: list[BoundingBoxField] = InputField(description="The bounding boxes to prompt the SAM model with.")
    apply_polygon_refinement: bool = InputField(
        description="Whether to apply polygon refinement to the masks. This will smooth the edges of the masks slightly and ensure that each mask consists of a single closed polygon (before merging).",
        default=True,
    )
    mask_filter: Literal["all", "largest", "highest_box_score"] = InputField(
        description="The filtering to apply to the detected masks before merging them into a final output.",
        default="all",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        # The models expect a 3-channel RGB image.
        image_pil = context.images.get_pil(self.image.image_name, mode="RGB")

        if len(self.bounding_boxes) == 0:
            combined_mask = np.zeros(image_pil.size[::-1], dtype=np.uint8)
        else:
            masks = self._segment(context=context, image=image_pil)
            masks = self._filter_masks(masks=masks, bounding_boxes=self.bounding_boxes)
            # masks contains binary values of 0 or 1, so we merge them via max-reduce.
            combined_mask = np.maximum.reduce(masks)

        # Map [0, 1] to [0, 255].
        mask_np = combined_mask * 255
        mask_pil = Image.fromarray(mask_np)

        image_dto = context.images.save(image=mask_pil)
        return ImageOutput.build(image_dto)

    @staticmethod
    def _load_sam_model(model_path: Path):
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

    def _segment(
        self,
        context: InvocationContext,
        image: Image.Image,
    ) -> list[npt.NDArray[np.uint8]]:
        """Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes."""
        # Convert the bounding boxes to the SAM input format.
        sam_bounding_boxes = [[bb.x_min, bb.y_min, bb.x_max, bb.y_max] for bb in self.bounding_boxes]

        with (
            context.models.load_remote_model(
                source=SEGMENT_ANYTHING_MODEL_ID, loader=SegmentAnythingModelInvocation._load_sam_model
            ) as sam_pipeline,
        ):
            assert isinstance(sam_pipeline, SegmentAnythingModel)
            masks = sam_pipeline.segment(image=image, bounding_boxes=sam_bounding_boxes)

        masks = self._to_numpy_masks(masks)
        if self.apply_polygon_refinement:
            masks = self._apply_polygon_refinement(masks)

        return masks

    def _to_numpy_masks(self, masks: torch.Tensor) -> list[npt.NDArray[np.uint8]]:
        """Convert the tensor output from the Segment Anything model to a list of numpy masks."""
        eps = 0.0001
        # [num_masks, channels, height, width] -> [num_masks, height, width]
        masks = masks.permute(0, 2, 3, 1).float().mean(dim=-1)
        masks = masks > eps
        np_masks = masks.cpu().numpy().astype(np.uint8)
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

    def _filter_masks(
        self, masks: list[npt.NDArray[np.uint8]], bounding_boxes: list[BoundingBoxField]
    ) -> list[npt.NDArray[np.uint8]]:
        """Filter the detected masks based on the specified mask filter."""
        assert len(masks) == len(bounding_boxes)

        if self.mask_filter == "all":
            return masks
        elif self.mask_filter == "largest":
            # Find the largest mask.
            return [max(masks, key=lambda x: x.sum())]
        elif self.mask_filter == "highest_box_score":
            # Find the index of the bounding box with the highest score.
            # Note that we fallback to -1.0 if the score is None. This is mainly to satisfy the type checker. In most
            # cases the scores should all be non-None when using this filtering mode. That being said, -1.0 is a
            # reasonable fallback since the expected score range is [0.0, 1.0].
            max_score_idx = max(range(len(bounding_boxes)), key=lambda i: bounding_boxes[i].score or -1.0)
            return [masks[max_score_idx]]
        else:
            raise ValueError(f"Invalid mask filter: {self.mask_filter}")
