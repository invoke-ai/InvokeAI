"""Small subprocess entry point for bounded fontTools work.

The API process sends one font as stdin and receives a pickled result on stdout. Keeping
this module separate from the web application avoids importing the inference stack in a
worker and lets the service terminate a job that exceeds its deadline.
"""

from __future__ import annotations

import json
import pickle
import sys

from invokeai.app.services.fonts.fonts_default import _instantiate_font_bytes, _parse_font_bytes


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: fonts_worker.py parse <filename> <max_bytes> | instance <filename> <coordinates>")

    operation = sys.argv[1]
    filename = sys.argv[2]
    data = sys.stdin.buffer.read()
    try:
        if operation == "parse" and len(sys.argv) == 4:
            result = _parse_font_bytes(filename, data, max_bytes=int(sys.argv[3]))
        elif operation == "instance" and len(sys.argv) == 4:
            result = _instantiate_font_bytes(data, filename, json.loads(sys.argv[3]))
        else:
            raise ValueError("invalid font worker operation")
    except BaseException as error:
        payload = ("error", f"{type(error).__name__}: {error}")
    else:
        payload = ("ok", result)
    sys.stdout.buffer.write(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


if __name__ == "__main__":
    main()
