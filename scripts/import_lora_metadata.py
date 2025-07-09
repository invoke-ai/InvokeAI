#!/usr/bin/env python3
"""
Import LoRA metadata from JSON files into InvokeAI database.

This script reads JSON files with the following format:
{
    "description": "",
    "sd version": "Unknown",
    "activation text": "",
    "preferred weight": 0,
    "negative text": "",
    "notes": ""
}

And imports the metadata into existing LoRA models in the InvokeAI database.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_records import ModelRecordChanges, ModelRecordServiceBase
from invokeai.app.services.model_records.model_records_sql import ModelRecordServiceSQL
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.backend.model_manager.config import AnyModelConfig
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.util.logging import InvokeAILogger


def map_sd_version_to_base_model(sd_version: str) -> Optional[BaseModelType]:
    """Map SD version string to BaseModelType."""
    sd_version_lower = sd_version.lower()
    
    if "xl" in sd_version_lower or "sdxl" in sd_version_lower:
        return BaseModelType.StableDiffusionXL
    elif "2" in sd_version_lower:
        return BaseModelType.StableDiffusion2
    elif "1" in sd_version_lower or "1.5" in sd_version_lower:
        return BaseModelType.StableDiffusion1
    elif "flux" in sd_version_lower:
        return BaseModelType.Flux
    else:
        return None  # Will not update base model if unknown


def build_description(json_data: Dict[str, Any]) -> str:
    """Build a comprehensive description from JSON data."""
    parts = []
    
    if json_data.get("description"):
        parts.append(json_data["description"])
    
    if json_data.get("preferred weight") and json_data["preferred weight"] != 0:
        parts.append(f"Preferred weight: {json_data['preferred weight']}")
    
    if json_data.get("negative text"):
        parts.append(f"Negative prompt: {json_data['negative text']}")
    
    if json_data.get("notes"):
        parts.append(f"Notes: {json_data['notes']}")
    
    return "\n\n".join(parts) if parts else ""


def process_lora_metadata(
    model_record_service: ModelRecordServiceBase,
    lora_model: AnyModelConfig,
    json_data: Dict[str, Any],
    update_base_model: bool = False,
) -> bool:
    """Process and update a single LoRA model with metadata from JSON."""
    changes = ModelRecordChanges()
    
    # Map activation text to trigger phrases
    if json_data.get("activation text"):
        activation_texts = [text.strip() for text in json_data["activation text"].split(",")]
        changes.trigger_phrases = set(activation_texts)
    
    # Build description from multiple fields
    description = build_description(json_data)
    if description:
        changes.description = description
    
    # Optionally update base model type
    if update_base_model and json_data.get("sd version"):
        base_model = map_sd_version_to_base_model(json_data["sd version"])
        if base_model:
            changes.base = base_model
    
    # Only update if we have changes
    if changes.model_dump(exclude_none=True):
        try:
            model_record_service.update_model(lora_model.key, changes)
            return True
        except Exception as e:
            print(f"Error updating model {lora_model.name}: {e}")
            return False
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Import LoRA metadata from JSON files into InvokeAI database"
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to JSON file containing LoRA metadata",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Name of the LoRA model to update (if not specified, will try to match based on filename)",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        help="Key of the LoRA model to update (takes precedence over --model-name)",
    )
    parser.add_argument(
        "--update-base-model",
        action="store_true",
        help="Update the base model type based on 'sd version' field",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple JSON files in batch mode (json_file should be a directory)",
    )
    
    args = parser.parse_args()
    
    # Initialize configuration and services
    config = InvokeAIAppConfig.get_config()
    logger = InvokeAILogger.get_logger("import_lora_metadata")
    
    # Initialize database
    db = SqliteDatabase(db_path=config.db_path, logger=logger)
    model_record_service = ModelRecordServiceSQL(db, logger)
    
    # Process single file or batch
    if args.batch:
        if not args.json_file.is_dir():
            print(f"Error: {args.json_file} is not a directory", file=sys.stderr)
            sys.exit(1)
        
        json_files = list(args.json_file.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {args.json_file}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            process_single_file(
                json_file, model_record_service, args, logger
            )
    else:
        process_single_file(
            args.json_file, model_record_service, args, logger
        )


def process_single_file(
    json_file: Path,
    model_record_service: ModelRecordServiceBase,
    args: argparse.Namespace,
    logger: Any,
) -> None:
    """Process a single JSON file."""
    if not json_file.exists():
        print(f"Error: {json_file} does not exist", file=sys.stderr)
        return
    
    try:
        with open(json_file, "r") as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading {json_file}: {e}", file=sys.stderr)
        return
    
    # Find the LoRA model to update
    lora_model = None
    
    if args.model_key:
        try:
            lora_model = model_record_service.get_model(args.model_key)
            if lora_model.type != ModelType.LoRA:
                print(f"Error: Model {args.model_key} is not a LoRA model", file=sys.stderr)
                return
        except Exception:
            print(f"Error: Model with key {args.model_key} not found", file=sys.stderr)
            return
    elif args.model_name:
        # Search for LoRA by name
        models = model_record_service.search_by_attr(
            model_name=args.model_name,
            model_type=ModelType.LoRA
        )
        if not models:
            print(f"Error: No LoRA model found with name '{args.model_name}'", file=sys.stderr)
            return
        elif len(models) > 1:
            print(f"Error: Multiple LoRA models found with name '{args.model_name}':", file=sys.stderr)
            for model in models:
                print(f"  - {model.key}: {model.name} ({model.base})")
            print("Please specify --model-key to select one", file=sys.stderr)
            return
        lora_model = models[0]
    else:
        # Try to match based on filename
        base_name = json_file.stem
        models = model_record_service.search_by_attr(
            model_name=base_name,
            model_type=ModelType.LoRA
        )
        if not models:
            # Try partial match
            all_loras = model_record_service.search_by_attr(model_type=ModelType.LoRA)
            matches = [m for m in all_loras if base_name.lower() in m.name.lower()]
            
            if not matches:
                print(f"Error: No LoRA model found matching filename '{base_name}'", file=sys.stderr)
                return
            elif len(matches) > 1:
                print(f"Error: Multiple LoRA models found matching '{base_name}':", file=sys.stderr)
                for model in matches:
                    print(f"  - {model.key}: {model.name} ({model.base})")
                print("Please specify --model-name or --model-key", file=sys.stderr)
                return
            lora_model = matches[0]
        elif len(models) > 1:
            print(f"Error: Multiple LoRA models found with name '{base_name}':", file=sys.stderr)
            for model in models:
                print(f"  - {model.key}: {model.name} ({model.base})")
            print("Please specify --model-key to select one", file=sys.stderr)
            return
        else:
            lora_model = models[0]
    
    # Display current and proposed changes
    print(f"\nProcessing: {json_file.name}")
    print(f"Target LoRA: {lora_model.name} (key: {lora_model.key})")
    print(f"Current base model: {lora_model.base}")
    
    if args.dry_run:
        print("\n--- DRY RUN MODE ---")
        print("Proposed changes:")
        
        if json_data.get("activation text"):
            print(f"  Trigger phrases: {json_data['activation text']}")
        
        description = build_description(json_data)
        if description:
            print(f"  Description: {description[:100]}..." if len(description) > 100 else f"  Description: {description}")
        
        if args.update_base_model and json_data.get("sd version"):
            base_model = map_sd_version_to_base_model(json_data["sd version"])
            if base_model:
                print(f"  Base model: {lora_model.base} → {base_model}")
        
        print("--- END DRY RUN ---\n")
    else:
        # Apply the updates
        success = process_lora_metadata(
            model_record_service,
            lora_model,
            json_data,
            args.update_base_model
        )
        
        if success:
            print("✓ Successfully updated metadata")
        else:
            print("✗ No changes made")


if __name__ == "__main__":
    main()