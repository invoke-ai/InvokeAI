from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from transformers import pipeline
from transformers.pipelines import ZeroShotObjectDetectionPipeline

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import BoundingBoxField, ImageField, InputField
from invokeai.app.invocations.primitives import BoundingBoxCollectionOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.grounding_dino.detection_result import DetectionResult
from invokeai.backend.image_util.grounding_dino.grounding_dino_pipeline import GroundingDinoPipeline

GroundingDinoModelKey = Literal["grounding-dino-tiny", "grounding-dino-base"]
GROUNDING_DINO_MODEL_IDS: dict[GroundingDinoModelKey, str] = {
    "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
    "grounding-dino-base": "IDEA-Research/grounding-dino-base",
}


@invocation(
    "grounding_dino",
    title="Grounding DINO (Text Prompt Object Detection)",
    tags=["prompt", "object detection"],
    category="image",
    version="1.0.0",
)
class GroundingDinoInvocation(BaseInvocation):
    """Runs a Grounding DINO model. Performs zero-shot bounding-box object detection from a text prompt."""

    # Reference:
    # - https://arxiv.org/pdf/2303.05499
    # - https://huggingface.co/docs/transformers/v4.43.3/en/model_doc/grounding-dino#grounded-sam
    # - https://github.com/NielsRogge/Transformers-Tutorials/blob/a39f33ac1557b02ebfb191ea7753e332b5ca933f/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb

    model: GroundingDinoModelKey = InputField(description="The Grounding DINO model to use.")
    prompt: str = InputField(description="The prompt describing the object to segment.")
    image: ImageField = InputField(description="The image to segment.")
    detection_threshold: float = InputField(
        description="The detection threshold for the Grounding DINO model. All detected bounding boxes with scores above this threshold will be returned.",
        ge=0.0,
        le=1.0,
        default=0.3,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> BoundingBoxCollectionOutput:
        # The model expects a 3-channel RGB image.
        image_pil = context.images.get_pil(self.image.image_name, mode="RGB")

        detections = self._detect(
            context=context, image=image_pil, labels=[self.prompt], threshold=self.detection_threshold
        )

        # Convert detections to BoundingBoxCollectionOutput.
        bounding_boxes: list[BoundingBoxField] = []
        for detection in detections:
            bounding_boxes.append(
                BoundingBoxField(
                    x_min=detection.box.xmin,
                    x_max=detection.box.xmax,
                    y_min=detection.box.ymin,
                    y_max=detection.box.ymax,
                    score=detection.score,
                )
            )
        return BoundingBoxCollectionOutput(collection=bounding_boxes)

    @staticmethod
    def _load_grounding_dino(model_path: Path):
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

        with context.models.load_remote_model(
            source=GROUNDING_DINO_MODEL_IDS[self.model], loader=GroundingDinoInvocation._load_grounding_dino
        ) as detector:
            assert isinstance(detector, GroundingDinoPipeline)
            return detector.detect(image=image, candidate_labels=labels, threshold=threshold)
