import numpy as np
import torch
from pydantic import BaseModel

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import ColorField, ImageField, InputField, OutputField, TensorField, WithMetadata
from invokeai.app.invocations.primitives import MaskOutput


@invocation(
    "rectangle_mask",
    title="Create Rectangle Mask",
    tags=["conditioning"],
    category="conditioning",
    version="1.0.0",
)
class RectangleMaskInvocation(BaseInvocation, WithMetadata):
    """Create a rectangular mask."""

    height: int = InputField(description="The height of the entire mask.")
    width: int = InputField(description="The width of the entire mask.")
    y_top: int = InputField(description="The top y-coordinate of the rectangular masked region (inclusive).")
    x_left: int = InputField(description="The left x-coordinate of the rectangular masked region (inclusive).")
    rectangle_height: int = InputField(description="The height of the rectangular masked region.")
    rectangle_width: int = InputField(description="The width of the rectangular masked region.")

    def invoke(self, context: InvocationContext) -> MaskOutput:
        mask = torch.zeros((1, self.height, self.width), dtype=torch.bool)
        mask[:, self.y_top : self.y_top + self.rectangle_height, self.x_left : self.x_left + self.rectangle_width] = (
            True
        )

        mask_tensor_name = context.tensors.save(mask)
        return MaskOutput(
            mask=TensorField(tensor_name=mask_tensor_name),
            width=self.width,
            height=self.height,
        )


class PromptColorPair(BaseModel):
    prompt: str
    color: ColorField


class PromptMaskPair(BaseModel):
    prompt: str
    mask: TensorField


default_prompt_color_pairs = [
    PromptColorPair(prompt="Strawberries", color=ColorField(r=200, g=0, b=0, a=255)),
    PromptColorPair(prompt="Frog", color=ColorField(r=0, g=200, b=0, a=255)),
    PromptColorPair(prompt="Banana", color=ColorField(r=0, g=0, b=200, a=255)),
    PromptColorPair(prompt="A gnome", color=ColorField(r=215, g=0, b=255, a=255)),
]


@invocation_output("extract_masks_and_prompts_output")
class ExtractMasksAndPromptsOutput(BaseInvocationOutput):
    prompt_mask_pairs: list[PromptMaskPair] = OutputField(description="List of prompts and their corresponding masks.")


@invocation(
    "extract_masks_and_prompts",
    title="Extract Masks and Prompts",
    tags=["conditioning"],
    category="conditioning",
    version="1.0.0",
)
class ExtractMasksAndPromptsInvocation(BaseInvocation):
    """Extract masks and prompts from a segmented mask image and prompt-to-color map."""

    prompt_color_pairs: list[PromptColorPair] = InputField(
        default=default_prompt_color_pairs, description="List of prompts and their corresponding colors."
    )
    image: ImageField = InputField(description="Mask to apply to the prompts.")

    def invoke(self, context: InvocationContext) -> ExtractMasksAndPromptsOutput:
        prompt_mask_pairs: list[PromptMaskPair] = []
        image = context.images.get_pil(self.image.image_name)
        image_as_tensor = torch.from_numpy(np.array(image, dtype=np.uint8))

        for pair in self.prompt_color_pairs:
            # TODO(ryand): Make this work for both RGB and RGBA images.
            mask = torch.all(image_as_tensor == torch.tensor(pair.color.tuple()), dim=-1)
            # Add explicit channel dimension.
            mask = mask.unsqueeze(0)
            mask_tensor_name = context.tensors.save(mask)
            prompt_mask_pairs.append(PromptMaskPair(prompt=pair.prompt, mask=TensorField(tensor_name=mask_tensor_name)))
        return ExtractMasksAndPromptsOutput(prompt_mask_pairs=prompt_mask_pairs)


@invocation_output("split_mask_prompt_pair_output")
class SplitMaskPromptPairOutput(BaseInvocationOutput):
    prompt: str = OutputField()
    mask: TensorField = OutputField()


@invocation(
    "split_mask_prompt_pair",
    title="Split Mask-Prompt pair",
    tags=["conditioning"],
    category="conditioning",
    version="1.0.0",
)
class SplitMaskPromptPair(BaseInvocation):
    """Extract masks and prompts from a segmented mask image and prompt-to-color map."""

    prompt_mask_pair: PromptMaskPair = InputField()

    def invoke(self, context: InvocationContext) -> SplitMaskPromptPairOutput:
        return SplitMaskPromptPairOutput(mask=self.prompt_mask_pair.mask, prompt=self.prompt_mask_pair.prompt)
