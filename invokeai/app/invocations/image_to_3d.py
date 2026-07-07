from pathlib import Path

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import Asset3DOutput
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.misc import uuid_string
from invokeai.backend.image_util.triposplat.triposplat_model import TripoSplatModel

# HuggingFace repo id. The repo holds five safetensors checkpoints in subfolders (diffusion_models/, vae/,
# clip_vision/, background_removal/). NOTE: we fetch it with huggingface_hub.snapshot_download rather than
# InvokeAI's load_remote_model, because the latter's filter_files() only keeps weight files whose name
# matches `model.*` — which excludes ALL of TripoSplat's checkpoints (it would resolve "0 files").
# License: MIT (code + weights).
TRIPOSPLAT_SOURCE = "VAST-AI/TripoSplat"

# Bounds enforced by the upstream pipeline (TripoSplatPipeline._validate_num_gaussians).
TRIPOSPLAT_MIN_GAUSSIANS = 32768
TRIPOSPLAT_MAX_GAUSSIANS = 262144


@invocation(
    "image_to_3d",
    title="Image to 3D (TripoSplat)",
    tags=["3d", "image", "gaussian", "splat"],
    category="3d",
    version="1.0.0",
    classification=Classification.Prototype,
)
class ImageTo3DInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates a 3D Gaussian splat (.ply) from a single image using TripoSplat."""

    image: ImageField = InputField(description="The image to convert to a 3D Gaussian splat.")
    remove_background: bool = InputField(
        default=True,
        description="Isolate the subject by removing the background (BiRefNet). Disable to feed the full image as-is.",
    )
    num_gaussians: int = InputField(
        default=TRIPOSPLAT_MAX_GAUSSIANS,
        ge=TRIPOSPLAT_MIN_GAUSSIANS,
        le=TRIPOSPLAT_MAX_GAUSSIANS,
        description="Number of 3D Gaussians to generate. Higher is more detailed but larger and slower to render.",
    )
    steps: int = InputField(default=20, ge=1, le=100, description="Number of flow-matching sampler steps.")
    guidance_scale: float = InputField(
        default=3.0, ge=0.0, description="Classifier-free guidance scale. Values <= 1.0 disable guidance."
    )
    seed: int = InputField(default=42, ge=0, description="Seed for reproducible generation.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> Asset3DOutput:
        if self.remove_background:
            # RGBA honors an existing alpha cutout; a fully-opaque input falls back to TripoSplat's own
            # BiRefNet background removal (which isolates the single most salient subject).
            image = context.images.get_pil(self.image.image_name, mode="RGBA")
        else:
            # Force-skip BiRefNet by handing the model a uniformly near-opaque alpha — TripoSplat only runs
            # background removal when there's no real alpha channel, so this feeds the whole frame as-is.
            image = context.images.get_pil(self.image.image_name, mode="RGB").convert("RGBA")
            image.putalpha(254)

        # Fetch the full repo snapshot directly (see TRIPOSPLAT_SOURCE note re: load_remote_model). Cached
        # by huggingface_hub after the first download; subsequent runs resolve instantly.
        from huggingface_hub import snapshot_download

        context.util.signal_progress("Downloading TripoSplat weights (first run only)")
        weights_dir = Path(snapshot_download(repo_id=TRIPOSPLAT_SOURCE))

        loaded_model = context.models.load_local_model(weights_dir, TripoSplatModel.load_model)

        with loaded_model as model:
            assert isinstance(model, TripoSplatModel)

            def _on_step(step: int, total: int) -> None:
                if context.util.is_canceled():
                    raise CanceledException
                context.util.signal_progress("Generating 3D Gaussian splat", step / total)

            gaussian, _prepared = model._pipe.run(
                image,
                seed=self.seed,
                steps=self.steps,
                guidance_scale=self.guidance_scale,
                num_gaussians=self.num_gaussians,
                show_progress=False,
                callback=_on_step,
            )
            ply_bytes = gaussian.to_ply_bytes()

        asset_name = f"{uuid_string()}.ply"
        # asset_files is a custom service with no curated InvocationContext accessor, so reach it directly.
        context._services.asset_files.save(asset_name, ply_bytes)
        return Asset3DOutput.build(asset_name=asset_name)
