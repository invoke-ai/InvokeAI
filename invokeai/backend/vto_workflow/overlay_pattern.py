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


def multiply_images(image_1: Image.Image, image_2: Image.Image) -> Image.Image:
    """Multiply two images together.

    Args:
        image_1 (Image.Image): The first image.
        image_2 (Image.Image): The second image.

    Returns:
        Image.Image: The product of the two images.
    """
    image_1_np = np.array(image_1, dtype=np.float32)
    if image_1_np.ndim == 2:
        # If the image is greyscale, add a channel dimension.
        image_1_np = np.expand_dims(image_1_np, axis=-1)
    image_2_np = np.array(image_2, dtype=np.float32)
    if image_2_np.ndim == 2:
        # If the image is greyscale, add a channel dimension.
        image_2_np = np.expand_dims(image_2_np, axis=-1)
    product_np = image_1_np * image_2_np // 255
    product_np = np.clip(product_np, 0, 255).astype(np.uint8)
    product = Image.fromarray(product_np)
    return product


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
