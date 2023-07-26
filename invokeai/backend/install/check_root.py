"""
Check that the invokeai_root is correctly configured and exit if not.
"""
import sys
from invokeai.app.services.config import (
    InvokeAIAppConfig,
)

def check_invokeai_root(config: InvokeAIAppConfig):
    try:
        assert config.model_conf_path.exists()
        assert config.db_path.exists()
        assert config.models_path.exists()
        for model in [
                'CLIP-ViT-bigG-14-laion2B-39B-b160k',
                'bert-base-uncased',
                'clip-vit-large-patch14',
                'sd-vae-ft-mse',
                'stable-diffusion-2-clip',
                'stable-diffusion-safety-checker']:
            assert (config.models_path / f'core/convert/{model}').exists()
    except:
        print()
        print('== STARTUP ABORTED ==')
        print('** One or more necessary files is missing from your InvokeAI root directory **')
        print('** Please rerun the configuration script to fix this problem. **')
        print('** From the launcher, selection option [7]. **')
        print('** From the command line, activate the virtual environment and run "invokeai-configure --yes --skip-sd-weights" **')
        input('Press any key to continue...')
        sys.exit(0)

