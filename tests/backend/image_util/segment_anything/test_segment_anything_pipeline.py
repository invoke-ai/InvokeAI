import torch
from PIL import Image
from transformers.models.sam.configuration_sam import SamConfig
from transformers.models.sam.image_processing_sam import SamImageProcessor
from transformers.models.sam.modeling_sam import SamModel
from transformers.models.sam.processing_sam import SamProcessor

from invokeai.backend.image_util.segment_anything.segment_anything_pipeline import (
    SegmentAnythingPipeline,
)
from invokeai.backend.image_util.segment_anything.shared import (
    SAMInput,
    SAMPoint,
    SAMPointLabel,
)

def test_segment_anything_pipeline_segment():
    width = 96
    height = 64

    image = Image.new("RGB", (width, height))

    model = SamModel(SamConfig())
    processor = SamProcessor(
        image_processor=SamImageProcessor(),
    )

    pipeline = SegmentAnythingPipeline(
        sam_model=model,
        sam_processor=processor,
    )

    inputs = [
        SAMInput(
            points=[
                SAMPoint(
                    x=width // 2,
                    y=height // 2,
                    label=SAMPointLabel.positive,
                )
            ]
        )
    ]

    with torch.inference_mode():
        masks = pipeline.segment(
            image=image,
            inputs=inputs,
        )

    assert masks.dtype == torch.bool
    assert masks.shape == (1, 3, height, width)
