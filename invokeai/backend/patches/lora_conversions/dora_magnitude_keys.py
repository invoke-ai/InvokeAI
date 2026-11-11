"""DoRA magnitude key suffixes shared by the lora_A/lora_B converters, mapped to the value keys ``DoRALayer`` reads.

The two orientations must not be mixed up (see ``DoRALayer``):
- ``.dora_scale`` (LyCORIS) may index either axis; ``DoRALayer`` detects which from the shape.
- ``.lora_magnitude_vector[.weight]`` (PEFT, with or without the ``.weight`` of the wrapped parameter) and
  ``.magnitude`` (ai-toolkit) always index the output dim.
"""

DORA_MAGNITUDE_SUFFIX_TO_VALUE_KEY = {
    ".dora_scale": "dora_scale",
    ".lora_magnitude_vector.weight": "dora_magnitude",
    ".lora_magnitude_vector": "dora_magnitude",
    ".magnitude": "dora_magnitude",
}

DORA_MAGNITUDE_VALUE_KEYS = ("dora_scale", "dora_magnitude")
