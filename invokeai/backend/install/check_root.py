"""
Check that the invokeai_root is correctly configured and exit if not.
"""
import sys
from invokeai.app.services.config import (
    InvokeAIAppConfig,
)


def check_invokeai_root(config: InvokeAIAppConfig):
    try:
        assert config.model_conf_path.exists(), f"{config.model_conf_path} not found"
        assert config.db_path.parent.exists(), f"{config.db_path.parent} not found"
        assert config.models_path.exists(), f"{config.models_path} not found"
        for model in [
            "CLIP-ViT-bigG-14-laion2B-39B-b160k",
            "bert-base-uncased",
            "clip-vit-large-patch14",
            "sd-vae-ft-mse",
            "stable-diffusion-2-clip",
            "stable-diffusion-safety-checker",
        ]:
            path = config.models_path / f"core/convert/{model}"
            assert path.exists(), f"{path} is missing"
    except Exception as e:
        print()
        print(f"An exception has occurred: {str(e)}")
        print("== STARTUP ABORTED ==")
        print("** One or more necessary files is missing from your InvokeAI root directory **")
        print("** Please rerun the configuration script to fix this problem. **")
        print("** From the launcher, selection option [7]. **")
        print(
            '** From the command line, activate the virtual environment and run "invokeai-configure --yes --skip-sd-weights" **'
        )
        input("Press any key to continue...")
        sys.exit(0)
