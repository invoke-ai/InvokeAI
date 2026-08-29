import torch
from PIL import Image
from transformers.models.sam2.configuration_sam2 import Sam2Config
from transformers.models.sam2.image_processing_sam2 import Sam2ImageProcessor
from transformers.models.sam2.modeling_sam2 import Sam2Model
from transformers.models.sam2.processing_sam2 import Sam2Processor

from invokeai.backend.image_util.segment_anything.segment_anything_2_pipeline import (
    SegmentAnything2Pipeline,
)

from invokeai.backend.image_util.segment_anything.shared import (
    SAMInput,
    SAMPoint,
    SAMPointLabel,
)

def test_segment_anything_2_pipeline_segment():
    width = 96
    height = 64

    image = Image.new("RGB", (width, height))

    model = Sam2Model(Sam2Config())
    processor = Sam2Processor(
        image_processor=Sam2ImageProcessor(),
    )

    pipeline = SegmentAnything2Pipeline(
        sam2_model=model,
        sam2_processor=processor,
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
