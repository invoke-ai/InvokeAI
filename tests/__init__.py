""" Create phony root directory for subsequent tests to use """
import os
from pathlib import Path
from tempfile import TemporaryDirectory

td = TemporaryDirectory()
for file in ["models", "databases", "autoimport", "nodes", "outputs"]:
    (Path(td.name) / file).mkdir(exist_ok=True, parents=True)
os.environ["INVOKEAI_ROOT"] = td.name
