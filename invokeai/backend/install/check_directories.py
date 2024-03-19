"""
Check that the invokeai_root is correctly configured and exit if not.
"""

import sys

from invokeai.app.services.config import InvokeAIAppConfig


# TODO(psyche): Should this also check for things like ESRGAN models, database, etc?
def validate_directories(config: InvokeAIAppConfig) -> None:
    assert config.db_path.parent.exists(), f"{config.db_path.parent} not found"
    assert config.models_path.exists(), f"{config.models_path} not found"


def check_directories(config: InvokeAIAppConfig):
    try:
        validate_directories(config)
    except Exception as e:
        print()
        print(f"An exception has occurred: {str(e)}")
        print("== STARTUP ABORTED ==")
        print("** One or more necessary files is missing from your InvokeAI directories **")
        print("** Please rerun the configuration script to fix this problem. **")
        print("** From the launcher, selection option [6]. **")
        print(
            '** From the command line, activate the virtual environment and run "invokeai-configure --yes --skip-sd-weights" **'
        )
        input("Press any key to continue...")
        sys.exit(0)
