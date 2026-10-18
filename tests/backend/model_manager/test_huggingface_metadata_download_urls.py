from pathlib import Path
from unittest.mock import Mock

from invokeai.backend.model_manager.metadata.metadata_base import HuggingFaceMetadata, RemoteModelFile


def test_repo_id_diffusers_download_uses_model_index_components() -> None:
    session = Mock()
    response = Mock(status_code=200)
    response.json.return_value = {
        "_class_name": "StableDiffusionPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    session.get.return_value = response
    repo_files = [
        "model_index.json",
        "model.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "README.md",
    ]
    metadata = HuggingFaceMetadata(
        id="org/diffusers-repo",
        name="diffusers-repo",
        files=[
            RemoteModelFile(
                url=f"https://huggingface.co/org/diffusers-repo/resolve/main/{path}",
                path=Path("diffusers-repo", path),
            )
            for path in repo_files
        ],
    )

    selected = {file.path.as_posix() for file in metadata.download_urls(session=session)}

    assert selected == {
        "diffusers-repo/model_index.json",
        "diffusers-repo/unet/config.json",
        "diffusers-repo/unet/diffusion_pytorch_model.safetensors",
        "diffusers-repo/vae/config.json",
        "diffusers-repo/vae/diffusion_pytorch_model.safetensors",
    }
    assert session.get.call_args.args[0].endswith("model_index.json")


def test_repo_id_diffusers_subfolder_uses_its_model_index_components() -> None:
    session = Mock()
    response = Mock(status_code=200)
    response.json.return_value = {
        "_class_name": "StableDiffusionPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    session.get.return_value = response
    repo_files = [
        "pipeline/model_index.json",
        "pipeline/model.safetensors",
        "pipeline/unet/config.json",
        "pipeline/unet/diffusion_pytorch_model.safetensors",
        "pipeline/vae/config.json",
        "pipeline/vae/diffusion_pytorch_model.safetensors",
    ]
    metadata = HuggingFaceMetadata(
        id="org/diffusers-repo",
        name="diffusers-repo",
        files=[
            RemoteModelFile(
                url=f"https://huggingface.co/org/diffusers-repo/resolve/main/{path}",
                path=Path("diffusers-repo", path),
            )
            for path in repo_files
        ],
    )

    selected = {file.path.as_posix() for file in metadata.download_urls(subfolder=Path("pipeline"), session=session)}

    assert selected == {
        "diffusers-repo/pipeline/model_index.json",
        "diffusers-repo/pipeline/unet/config.json",
        "diffusers-repo/pipeline/unet/diffusion_pytorch_model.safetensors",
        "diffusers-repo/pipeline/vae/config.json",
        "diffusers-repo/pipeline/vae/diffusion_pytorch_model.safetensors",
    }
    assert session.get.call_args.args[0].endswith("pipeline/model_index.json")
