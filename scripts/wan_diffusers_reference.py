"""Run TI2V-5B (or any Wan 2.2 Diffusers checkpoint) via the upstream
WanPipeline directly, with the same arguments InvokeAI's wan_denoise uses.

Use to A/B against InvokeAI output when image quality is questionable.
Generates one image and saves it next to this script.

Example:
    python scripts/wan_diffusers_reference.py \
        --model-path /home/lstein/invokeai-delete/models/<UUID> \
        --prompt "a photograph of a young redheaded woman sitting on a three-legged stool next to a potted fern" \
        --seed 42 --steps 40 --cfg 4.0 --width 1024 --height 1024
"""

import argparse
from pathlib import Path

import torch
from diffusers import WanPipeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Path to a Diffusers Wan model directory.")
    p.add_argument("--prompt", required=True)
    p.add_argument(
        "--negative",
        default="",
        help="Negative prompt (default empty string — matches WanPipeline.encode_prompt behaviour).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--output", default="wan_diffusers_reference.png")
    p.add_argument(
        "--offload",
        choices=["model", "sequential", "none"],
        default="model",
        help="VRAM-saving strategy. 'model' (default) keeps one component on GPU at a time — fits TI2V-5B "
        "in ~16 GB. 'sequential' is even more aggressive (per-module offload) and slower. "
        "'none' loads everything to GPU at once (~24 GB+).",
    )
    args = p.parse_args()

    print(f"Loading WanPipeline from {args.model_path} ...")
    pipe = WanPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)

    if args.offload == "model":
        # enable_model_cpu_offload puts each component (transformer, vae, text_encoder)
        # on GPU only while it's actively running; the rest sit on CPU. Adds a little
        # latency between stages but cuts peak VRAM dramatically.
        pipe.enable_model_cpu_offload()
    elif args.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    print(
        f"Generating: prompt={args.prompt!r}\n"
        f"  steps={args.steps}, cfg={args.cfg}, size={args.width}x{args.height}, seed={args.seed}"
    )
    # num_frames=1 → image generation
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        height=args.height,
        width=args.width,
        num_frames=1,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=generator,
        output_type="pil",
    )
    # WanPipelineOutput.frames is a list of [PIL.Image] sequences (one per video).
    image = result.frames[0][0]
    out = Path(args.output)
    image.save(out)
    print(f"Saved {out.resolve()}")


if __name__ == "__main__":
    main()
