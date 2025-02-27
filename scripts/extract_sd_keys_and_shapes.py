import argparse
import json

from safetensors.torch import load_file


def extract_sd_keys_and_shapes(safetensors_file: str):
    sd = load_file(safetensors_file)

    keys_to_shapes = {k: v.shape for k, v in sd.items()}

    out_file = "keys_and_shapes.json"
    with open(out_file, "w") as f:
        json.dump(keys_to_shapes, f, indent=4)

    print(f"Keys and shapes written to '{out_file}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Extracts the keys and shapes from the state dict in a safetensors file. Intended for creating "
        + "dummy state dicts for use in unit tests."
    )
    parser.add_argument("safetensors_file", type=str, help="Path to the safetensors file.")
    args = parser.parse_args()
    extract_sd_keys_and_shapes(args.safetensors_file)


if __name__ == "__main__":
    main()
