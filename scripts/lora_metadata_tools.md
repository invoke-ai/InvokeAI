# LoRA Metadata Import/Export Tools

These scripts allow you to import and export LoRA metadata between JSON files and the InvokeAI database.

## JSON Format

The JSON format used by these tools is:

```json
{
    "description": "Description of the LoRA",
    "sd version": "SDXL",
    "activation text": "trigger1, trigger2",
    "preferred weight": 0.8,
    "negative text": "negative prompts to avoid",
    "notes": "Additional notes about the LoRA"
}
```

## Import Script: `import_lora_metadata.py`

Imports metadata from JSON files into existing LoRA models in the InvokeAI database.

### Usage

```bash
# Import metadata for a single LoRA (matches by filename)
python scripts/import_lora_metadata.py my_lora.json

# Import metadata by specifying the model name
python scripts/import_lora_metadata.py metadata.json --model-name "My LoRA Model"

# Import metadata by specifying the model key
python scripts/import_lora_metadata.py metadata.json --model-key "abc123def456"

# Dry run to see what would be changed
python scripts/import_lora_metadata.py my_lora.json --dry-run

# Update the base model type based on "sd version" field
python scripts/import_lora_metadata.py my_lora.json --update-base-model

# Batch import multiple JSON files from a directory
python scripts/import_lora_metadata.py /path/to/json/directory --batch
```

### Field Mappings

- `activation text` → `trigger_phrases` (comma-separated list)
- `description` → `description`
- `preferred weight`, `negative text`, `notes` → Combined into `description` field
- `sd version` → `base` (when `--update-base-model` is used)

### SD Version Mapping

When using `--update-base-model`, the script maps SD versions as follows:
- Contains "xl" or "sdxl" → StableDiffusionXL
- Contains "2" → StableDiffusion2
- Contains "1" or "1.5" → StableDiffusion1
- Contains "flux" → Flux
- Other → No update

## Export Script: `export_lora_metadata.py`

Exports LoRA metadata from the InvokeAI database to JSON files.

### Usage

```bash
# Export all LoRA models to current directory
python scripts/export_lora_metadata.py

# Export to specific directory
python scripts/export_lora_metadata.py --output-dir /path/to/output

# Export specific LoRA by name
python scripts/export_lora_metadata.py --model-name "My LoRA Model"

# Export specific LoRA by key
python scripts/export_lora_metadata.py --model-key "abc123def456"

# Custom filename pattern
python scripts/export_lora_metadata.py --filename-pattern "{base}_{name}.json"

# Pretty-print JSON output
python scripts/export_lora_metadata.py --pretty

# Overwrite existing files
python scripts/export_lora_metadata.py --overwrite
```

### Filename Patterns

Available placeholders for `--filename-pattern`:
- `{name}` - LoRA model name
- `{key}` - LoRA model key/ID
- `{base}` - Base model type (e.g., "sdxl", "sd-1", etc.)

## Examples

### Example 1: Import metadata for a newly added LoRA

1. Add a LoRA model to InvokeAI (e.g., `anime_style_v2.safetensors`)
2. Create a JSON file with metadata (`anime_style_v2.json`):
   ```json
   {
       "description": "Anime style LoRA trained on modern anime artwork",
       "sd version": "SDXL",
       "activation text": "anime style, modern anime",
       "preferred weight": 0.7,
       "negative text": "realistic, photorealistic",
       "notes": "Works best with anime-focused base models"
   }
   ```
3. Import the metadata:
   ```bash
   python scripts/import_lora_metadata.py anime_style_v2.json
   ```

### Example 2: Batch export and import

1. Export all LoRA metadata:
   ```bash
   python scripts/export_lora_metadata.py --output-dir ./lora_metadata --pretty
   ```
2. Edit the JSON files as needed
3. Import all metadata back:
   ```bash
   python scripts/import_lora_metadata.py ./lora_metadata --batch
   ```

## Notes

- The import script requires that LoRA models already exist in the InvokeAI database
- When importing, the script will try to match JSON filenames to LoRA model names
- Use `--dry-run` to preview changes before applying them
- The scripts preserve existing data when possible (e.g., appending to descriptions rather than replacing)