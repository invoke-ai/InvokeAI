"""Utility functions for extracting metadata from LoRA model files."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from PIL import Image

from invokeai.app.util.thumbnails import make_thumbnail
from invokeai.backend.model_manager.config import AnyModelConfig, ModelType

logger = logging.getLogger(__name__)


def extract_lora_metadata(
    model_path: Path, model_key: str, model_images_path: Path
) -> Tuple[Optional[str], Optional[Set[str]]]:
    """
    Extract metadata for a LoRA model from associated JSON and image files.

    Args:
        model_path: Path to the LoRA model file
        model_key: Unique key for the model
        model_images_path: Path to the model images directory

    Returns:
        Tuple of (description, trigger_phrases)
    """
    model_stem = model_path.stem
    model_dir = model_path.parent

    # Find and process preview image
    _process_preview_image(model_stem, model_dir, model_key, model_images_path)

    # Extract metadata from JSON
    description, trigger_phrases = _extract_json_metadata(model_stem, model_dir)

    return description, trigger_phrases


def _process_preview_image(model_stem: str, model_dir: Path, model_key: str, model_images_path: Path) -> bool:
    """Find and process a preview image for the model, saving it to the model images store."""
    image_extensions = [".png", ".jpg", ".jpeg", ".webp"]

    for ext in image_extensions:
        image_path = model_dir / f"{model_stem}{ext}"
        if image_path.exists():
            try:
                # Open the image
                with Image.open(image_path) as img:
                    # Create thumbnail and save to model images directory
                    thumbnail = make_thumbnail(img, 256)
                    thumbnail_path = model_images_path / f"{model_key}.webp"
                    thumbnail.save(thumbnail_path, format="webp")

                logger.info(f"Processed preview image {image_path.name} for model {model_key}")
                return True

            except Exception as e:
                logger.warning(f"Failed to process preview image {image_path.name}: {e}")
                return False

    return False


def _extract_json_metadata(model_stem: str, model_dir: Path) -> Tuple[Optional[str], Optional[Set[str]]]:
    """Extract metadata from a JSON file with the same name as the model."""
    json_path = model_dir / f"{model_stem}.json"

    if not json_path.exists():
        return None, None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Extract description
        description = _build_description(metadata)

        # Extract trigger phrases
        trigger_phrases = _extract_trigger_phrases(metadata)

        if description or trigger_phrases:
            logger.info(f"Applied metadata from {json_path.name}")

        return description, trigger_phrases

    except (json.JSONDecodeError, IOError, Exception) as e:
        logger.warning(f"Failed to read metadata from {json_path}: {e}")
        return None, None


def _build_description(metadata: Dict[str, Any]) -> Optional[str]:
    """Build a description from metadata fields."""
    description_parts = []

    if description := metadata.get("description"):
        description_parts.append(str(description).strip())

    if notes := metadata.get("notes"):
        description_parts.append(str(notes).strip())

    return " | ".join(description_parts) if description_parts else None


def _extract_trigger_phrases(metadata: Dict[str, Any]) -> Optional[Set[str]]:
    """Extract trigger phrases from metadata."""
    if not (activation_text := metadata.get("activation text")):
        return None

    activation_text = str(activation_text).strip()
    if not activation_text:
        return None

    # Split on commas and clean up each phrase
    phrases = [phrase.strip() for phrase in activation_text.split(",") if phrase.strip()]

    return set(phrases) if phrases else None


def apply_lora_metadata(info: AnyModelConfig, model_path: Path, model_images_path: Path) -> None:
    """
    Apply extracted metadata to a LoRA model configuration.

    Args:
        info: The model configuration to update
        model_path: Path to the LoRA model file
        model_images_path: Path to the model images directory
    """
    # Only process LoRA models
    if info.type != ModelType.LoRA:
        return

    # Extract and apply metadata
    description, trigger_phrases = extract_lora_metadata(model_path, info.key, model_images_path)

    # We don't set cover_image path in the config anymore since images are stored
    # separately in the model images store by model key

    if description:
        info.description = description

    if trigger_phrases:
        info.trigger_phrases = trigger_phrases
