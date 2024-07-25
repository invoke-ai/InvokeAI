import torch
from PIL import Image
from transformers import AutoProcessor, CLIPSegForImageSegmentation, CLIPSegProcessor


def load_clipseg_model() -> tuple[CLIPSegProcessor, CLIPSegForImageSegmentation]:
    # Load the model.
    clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    return clipseg_processor, clipseg_model


def run_clipseg(
    images: list[Image.Image],
    prompt: str,
    clipseg_processor,
    clipseg_model,
    clipseg_temp: float,
    device: torch.device,
) -> list[Image.Image]:
    """Run ClipSeg on a list of images.

    Args:
        clipseg_temp (float): Temperature applied to the CLIPSeg logits. Higher values cause the mask to be 'smoother'
            and include more of the background. Recommended range: 0.5 to 1.0.
    """

    orig_image_sizes = [img.size for img in images]

    prompts = [prompt] * len(images)
    # TODO(ryand): Should we run the same image with and without the prompt to normalize for any bias in the model?
    inputs = clipseg_processor(text=prompts, images=images, padding=True, return_tensors="pt")

    # Move inputs and clipseg_model to the correct device and dtype.
    inputs = {k: v.to(device=device) for k, v in inputs.items()}
    clipseg_model = clipseg_model.to(device=device)

    outputs = clipseg_model(**inputs)

    logits = outputs.logits
    if logits.ndim == 2:
        # The model squeezes the batch dimension if it's 1, so we need to unsqueeze it.
        logits = logits.unsqueeze(0)
    probs = torch.nn.functional.sigmoid(logits / clipseg_temp)
    # Normalize each mask to 0-255. Note that each mask is normalized independently.
    probs = 255 * probs / probs.amax(dim=(1, 2), keepdim=True)

    # Make mask greyscale.
    masks: list[Image.Image] = []
    for prob, orig_size in zip(probs, orig_image_sizes, strict=True):
        mask = Image.fromarray(prob.cpu().detach().numpy()).convert("L")
        mask = mask.resize(orig_size)
        masks.append(mask)

    return masks


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
