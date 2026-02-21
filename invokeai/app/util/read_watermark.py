"""CLI command to decode invisible watermarks from Invoke-generated images."""

import argparse
import sys

from PIL import Image

from invokeai.backend.image_util.invisible_watermark import InvisibleWatermark


def read_watermark() -> None:
    """Read and print invisible watermarks from a list of image files."""
    parser = argparse.ArgumentParser(
        prog="invoke-readwatermark",
        description="Decode invisible watermarks from Invoke-generated images.",
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE", help="Image file(s) to read watermarks from.")
    parser.add_argument(
        "--length",
        type=int,
        default=8,
        metavar="BYTES",
        help="Expected watermark length in bytes (default: %(default)s, matching the default 'InvokeAI' watermark text).",
    )
    args = parser.parse_args()

    for path in args.images:
        try:
            image = Image.open(path)
        except OSError as e:
            print(f"{path}: error opening image: {e}", file=sys.stderr)
            continue
        watermark = InvisibleWatermark.decode_watermark(image, watermark_length=args.length)
        print(f"{path}: {watermark}")
