""" Create phony root directory for subsequent tests to use """
import os
from tempfile import TemporaryDirectory
from pathlib import Path

td = TemporaryDirectory()
for file in ['models','databases','autoimport','nodes','outputs']:
    (Path(td.name) / file).mkdir(exist_ok=True, parents=True)
os.environ['INVOKEAI_ROOT']=td.name

