"""A script to generate a linear approximation of the VAE decode operation. The resultant matrix can be used to quickly
visualize intermediate states of the denoising process.
"""

import argparse
from pathlib import Path

import einops
import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL
from PIL import Image
from tqdm import tqdm


def trim_to_multiple_of(*args: int, multiple_of: int = 8) -> tuple[int, ...]:
    return tuple((x - x % multiple_of) for x in args)


def image_to_tensor(image: Image.Image, h: int, w: int, normalize: bool = True) -> torch.Tensor:
    transformation = T.Compose([T.Resize((h, w), T.InterpolationMode.LANCZOS), T.ToTensor()])
    tensor: torch.Tensor = transformation(image)  # type: ignore
    if normalize:
        tensor = tensor * 2.0 - 1.0
    return tensor


def vae_preprocess(image: Image.Image, normalize: bool = True, multiple_of: int = 8) -> torch.Tensor:
    w, h = trim_to_multiple_of(*image.size, multiple_of=multiple_of)
    return image_to_tensor(image, h, w, normalize)


@torch.no_grad()
def vae_encode(vae: AutoencoderKL, image_tensor: torch.Tensor) -> torch.Tensor:
    if image_tensor.dim() == 3:
        image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

    orig_dtype = vae.dtype

    vae.enable_tiling()

    image_tensor = image_tensor.to(device=vae.device, dtype=vae.dtype)
    image_tensor_dist = vae.encode(image_tensor).latent_dist
    latents = image_tensor_dist.sample().to(dtype=vae.dtype)  # FIXME: uses torch.randn. make reproducible!

    latents = vae.config.scaling_factor * latents
    latents = latents.to(dtype=orig_dtype)
    return latents.detach()


@torch.no_grad()
def prepare_data(
    vae: AutoencoderKL, image_dir: str, device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    latents: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    image_paths = Path(image_dir).iterdir()
    image_paths = list(filter(lambda p: p.suffix.lower() in [".png", ".jpg", ".jpeg"], image_paths))

    for image_path in tqdm(image_paths, desc="Preparing images"):
        image = Image.open(image_path).convert("RGB")
        image_tensor = vae_preprocess(image)
        latent = vae_encode(vae, image_tensor)
        latent = latent.squeeze(0)
        _, h, w = latent.shape
        # Resize the image to the latent size.
        target = image_to_tensor(image=image, h=h, w=w)

        latents.append(latent)
        targets.append(target)

    return latents, targets


def train(
    latents: list[torch.Tensor],
    targets: list[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    num_epochs: int = 500,
    lr: float = 0.01,
):
    # Initialize latent_rgb_factors randomly
    latent_channels, _, _ = latents[0].shape
    latent_to_image = torch.randn(latent_channels, 3, device=device, dtype=dtype, requires_grad=True)

    optimizer = torch.optim.Adam([latent_to_image], lr=lr)
    loss_fn = torch.nn.MSELoss()

    epoch_pbar = tqdm(range(num_epochs), desc="Training")
    for _ in epoch_pbar:
        total_loss = 0.0
        for latent, target in zip(latents, targets, strict=True):
            latent = latent.to(device=device, dtype=dtype)
            target = target.to(device=device, dtype=dtype)

            # latent and target have shape [C, H, W]. Rearrange to [H, W, C].
            latent = latent.permute(1, 2, 0)
            target = target.permute(1, 2, 0)

            # Forward pass
            predicted = latent @ latent_to_image  # [H, W, 3]

            # Compute loss
            loss = loss_fn(predicted, target)
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(latents)
        epoch_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return latent_to_image.detach()


@torch.no_grad()
def validate(vae: AutoencoderKL, latent_to_image: torch.Tensor, test_image_dir: str):
    val_dir = Path("vae_approx_out")
    val_dir.mkdir(exist_ok=True)

    for image_path in Path(test_image_dir).iterdir():
        if image_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        image = Image.open(image_path).convert("RGB")
        image_tensor = vae_preprocess(image)
        latent = vae_encode(vae, image_tensor)

        latent = latent.squeeze(0).permute(1, 2, 0).to(device="cpu")
        predicted_image_tensor = latent @ latent_to_image.to(device="cpu")
        predicted_rgb = (((predicted_image_tensor + 1) / 2).clamp(0, 1).mul(0xFF)).to(dtype=torch.uint8)
        predicted_img = Image.fromarray(predicted_rgb.numpy())

        out_path = val_dir / f"{image_path.stem}.png"
        predicted_img.save(out_path)
        print(f"Saved validation image to: {out_path}")


def generate_linear_approximation(vae_path: str, train_image_dir: str, test_image_dir: str):
    device = torch.device("cuda")

    # Load the VAE model.
    print(f"Loading VAE model from: {vae_path}")
    vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True)
    vae.to(device=device)  # type: ignore
    print("Loaded VAE model.")

    print(f"Loading training images from: {train_image_dir}")
    latents, targets = prepare_data(vae, train_image_dir, device=torch.device("cuda"))
    print(f"Loaded {len(latents)} images for training.")

    latent_to_image = train(latents, targets, device=device, dtype=torch.float32)
    print(f"\nTrained latent_to_image matrix:\n{latent_to_image.cpu().numpy()}")

    validate(vae, latent_to_image, test_image_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate a linear approximation of the VAE decode operation.")
    parser.add_argument("--vae", type=str, required=True, help="Path to a diffusers AutoencoderKL model directory.")
    parser.add_argument(
        "--train_image_dir",
        type=str,
        required=True,
        help="Path to a directory containing images to be used for training.",
    )
    parser.add_argument(
        "--test_image_dir",
        type=str,
        required=True,
        help="Path to a directory containing images to be used for validation.",
    )

    args = parser.parse_args()

    generate_linear_approximation(args.vae, args.train_image_dir, args.test_image_dir)


if __name__ == "__main__":
    main()
