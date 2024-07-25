import numpy as np
import torch
from PIL import Image

from invokeai.backend.vto_workflow.clipseg import load_clipseg_model, run_clipseg


@torch.no_grad()
def generate_dress_mask(model_image):
    """Return a mask of the dress in the image.

    Returns:
        np.ndarray: Shape (H, W) of dtype bool. True where the dress is, False elsewhere.
    """
    clipseg_processor, clipseg_model = load_clipseg_model()

    masks = run_clipseg(
        images=[model_image],
        prompt="a dress",
        clipseg_processor=clipseg_processor,
        clipseg_model=clipseg_model,
        clipseg_temp=1.0,
        device=torch.device("cuda"),
    )

    mask_np = np.array(masks[0])
    thresh = 128
    binary_mask = mask_np > thresh
    return binary_mask


@torch.inference_mode()
def main():
    # Load the model image.
    model_image = Image.open("/home/ryan/src/InvokeAI/invokeai/backend/vto_workflow/dress.jpeg")

    # Load the pattern image.
    pattern_image = Image.open("/home/ryan/src/InvokeAI/invokeai/backend/vto_workflow/pattern1.jpg")

    # Generate a mask for the dress.
    mask = generate_dress_mask(model_image)

    print("hi")


if __name__ == "__main__":
    main()
