import os
import shutil
from .constants import LORA_DIR

def delete_lora(lora_name: str) -> None:
    """Delete a single LoRA weights folder safely.

    Args:
        lora_name: The name of the LoRA to delete. Must correspond to a
            subdirectory under :data:`LORA_DIR`.
    """
    lora_path = os.path.join(LORA_DIR, lora_name)
    if not os.path.isdir(lora_path):
        raise FileNotFoundError(f"LoRA '{lora_name}' not found in {LORA_DIR}")
    
    # Remove only the specific LoRA directory, not its parent.
    try:
        shutil.rmtree(lora_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to delete LoRA '{lora_name}': {exc}")


def list_installed_loras() -> list[str]:
    """Return a list of installed LoRA folder names."""
    if not os.path.isdir(LORA_DIR):
        return []
    return [d for d in os.listdir(LORA_DIR) if os.path.isdir(os.path.join(LORA_DIR, d))]
