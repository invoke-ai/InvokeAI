import re
import sys
import urllib.request
from pathlib import Path

# This script checks the classifiers in a pyproject.toml file against the official Trove classifier list.
# If the classifiers are invalid, PyPI will reject the package upload.

# Step 1: Get pyproject.toml path from args
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} path/to/pyproject.toml", file=sys.stderr)
    sys.exit(1)

pyproject_path = Path(sys.argv[1])
if not pyproject_path.is_file():
    print(f"File not found: {pyproject_path}", file=sys.stderr)
    sys.exit(1)

# Step 1: Download the official Trove classifier list
url = "https://pypi.org/pypi?%3Aaction=list_classifiers"
with urllib.request.urlopen(url) as response:
    trove_classifiers = {line.decode("utf-8").strip() for line in response}

# Step 2: Extract classifiers from pyproject.toml
with open(pyproject_path) as f:
    content = f.read()

match = re.search(r"classifiers\s*=\s*\[([^\]]*)\]", content, re.MULTILINE | re.DOTALL)
if not match:
    print("No 'classifiers' block found in pyproject.toml", file=sys.stderr)
    sys.exit(1)

raw_block = match.group(1)
classifiers = [c.strip(" \"'\n") for c in raw_block.split(",") if c.strip()]

# Step 3: Check for invalid classifiers
invalid = [c for c in classifiers if c not in trove_classifiers]

if invalid:
    print("❌ Invalid classifiers:")
    for c in invalid:
        print(f"  - {c}")
    print("Valid classifiers:")
    for c in sorted(trove_classifiers):
        print(f"  - {c}")
    sys.exit(1)
else:
    print("✅ All classifiers are valid.")
