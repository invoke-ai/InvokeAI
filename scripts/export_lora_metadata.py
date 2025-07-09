#!/usr/bin/env python3
"""
Export LoRA metadata from InvokeAI database to JSON files.

This script exports LoRA metadata to JSON files with the following format:
{
    "description": "",
    "sd version": "Unknown",
    "activation text": "",
    "preferred weight": 0,
    "negative text": "",
    "notes": ""
}
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_records.model_records_sql import ModelRecordServiceSQL
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.backend.model_manager.config import AnyModelConfig
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.util.logging import InvokeAILogger


def map_base_model_to_sd_version(base_model: BaseModelType) -> str:
    """Map BaseModelType to SD version string."""
    mapping = {
        BaseModelType.StableDiffusion1: "SD 1.5",
        BaseModelType.StableDiffusion2: "SD 2.x", 
        BaseModelType.StableDiffusionXL: "SDXL",
        BaseModelType.Flux: "FLUX",
    }
    return mapping.get(base_model, "Unknown")


def parse_description(description: Optional[str]) -> Dict[str, Any]:
    """Parse description field to extract structured data."""
    result = {
        "description": "",
        "preferred_weight": 0,
        "negative_text": "",
        "notes": ""
    }
    
    if not description:
        return result
    
    # Try to extract structured parts from description
    lines = description.split("\n")
    current_section = "description"
    section_content = []
    
    for line in lines:
        line = line.strip()
        
        # Check for section markers
        if line.startswith("Preferred weight:"):
            # Save previous section
            if current_section == "description" and section_content:
                result["description"] = "\n".join(section_content).strip()
            
            # Extract weight
            weight_match = re.search(r"Preferred weight:\s*([\d.]+)", line)
            if weight_match:
                try:
                    result["preferred_weight"] = float(weight_match.group(1))  # type: ignore
                except ValueError:
                    pass
            current_section = "after_weight"
            section_content = []
        
        elif line.startswith("Negative prompt:"):
            # Extract negative text
            negative_text = line[len("Negative prompt:"):].strip()
            result["negative_text"] = negative_text
            current_section = "after_negative"
            section_content = []
        
        elif line.startswith("Notes:"):
            # Extract notes
            notes = line[len("Notes:"):].strip()
            result["notes"] = notes
            current_section = "notes"
            section_content = [notes] if notes else []
        
        elif line and current_section == "notes":
            # Continue adding to notes
            section_content.append(line)
        
        elif line and current_section == "description":
            # Add to description
            section_content.append(line)
    
    # Save final section
    if current_section == "description" and section_content:
        result["description"] = "\n".join(section_content).strip()
    elif current_section == "notes" and section_content:
        result["notes"] = "\n".join(section_content).strip()
    
    return result


def export_lora_metadata(lora_model: AnyModelConfig) -> Dict[str, Any]:
    """Export LoRA model metadata to JSON format."""
    # Parse description to extract structured data
    parsed = parse_description(lora_model.description)
    
    # Build activation text from trigger phrases
    activation_text = ""
    if hasattr(lora_model, 'trigger_phrases') and lora_model.trigger_phrases:
        activation_text = ", ".join(sorted(lora_model.trigger_phrases))
    
    # Build final JSON structure
    return {
        "description": parsed["description"],
        "sd version": map_base_model_to_sd_version(lora_model.base),
        "activation text": activation_text,
        "preferred weight": parsed["preferred_weight"],
        "negative text": parsed["negative_text"],
        "notes": parsed["notes"]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export LoRA metadata from InvokeAI database to JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save JSON files (default: current directory)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Export only the specified LoRA model by name",
    )
    parser.add_argument(
        "--model-key", 
        type=str,
        help="Export only the specified LoRA model by key",
    )
    parser.add_argument(
        "--filename-pattern",
        type=str,
        default="{name}.json",
        help="Filename pattern for JSON files (default: {name}.json). "
             "Available placeholders: {name}, {key}, {base}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    
    args = parser.parse_args()
    
    # Initialize configuration and services
    config = InvokeAIAppConfig.get_config()
    logger = InvokeAILogger.get_logger("export_lora_metadata")
    
    # Initialize database
    db = SqliteDatabase(db_path=config.db_path, logger=logger)
    model_record_service = ModelRecordServiceSQL(db, logger)
    
    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get LoRA models to export
    lora_models = []
    
    if args.model_key:
        try:
            model = model_record_service.get_model(args.model_key)
            if model.type != ModelType.LoRA:
                print(f"Error: Model {args.model_key} is not a LoRA model", file=sys.stderr)
                sys.exit(1)
            lora_models = [model]
        except Exception:
            print(f"Error: Model with key {args.model_key} not found", file=sys.stderr)
            sys.exit(1)
    
    elif args.model_name:
        models = model_record_service.search_by_attr(
            model_name=args.model_name,
            model_type=ModelType.LoRA
        )
        if not models:
            print(f"Error: No LoRA model found with name '{args.model_name}'", file=sys.stderr)
            sys.exit(1)
        lora_models = models
    
    else:
        # Export all LoRA models
        lora_models = model_record_service.search_by_attr(model_type=ModelType.LoRA)
    
    if not lora_models:
        print("No LoRA models found in database", file=sys.stderr)
        sys.exit(1)
    
    print(f"Exporting {len(lora_models)} LoRA model(s)...")
    
    # Export each model
    exported_count = 0
    skipped_count = 0
    
    for lora_model in lora_models:
        # Generate filename
        filename = args.filename_pattern.format(
            name=lora_model.name,
            key=lora_model.key,
            base=lora_model.base.value
        )
        
        # Sanitize filename
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        output_path = args.output_dir / filename
        
        # Check if file exists
        if output_path.exists() and not args.overwrite:
            print(f"Skipping {lora_model.name}: {output_path} already exists")
            skipped_count += 1
            continue
        
        # Export metadata
        metadata = export_lora_metadata(lora_model)
        
        # Write JSON file
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                if args.pretty:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(metadata, f, ensure_ascii=False)
            
            print(f"Exported {lora_model.name} → {output_path}")
            exported_count += 1
            
        except Exception as e:
            print(f"Error exporting {lora_model.name}: {e}", file=sys.stderr)
    
    # Summary
    print(f"\nExport complete:")
    print(f"  Exported: {exported_count}")
    if skipped_count > 0:
        print(f"  Skipped: {skipped_count} (use --overwrite to replace)")


if __name__ == "__main__":
    main()