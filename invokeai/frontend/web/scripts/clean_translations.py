# Cleans translations by removing unused keys
# Usage: python clean_translations.py
# Note: Must be run from invokeai/frontend/web/scripts directory
#
# After running the script, open `en.json` and check for empty objects (`{}`) and remove them manually.
# Also, the script does not handle keys with underscores. They need to be checked manually.

import json
import os
import re
from typing import TypeAlias, Union

from tqdm import tqdm

RecursiveDict: TypeAlias = dict[str, Union["RecursiveDict", str]]


class TranslationCleaner:
    file_cache: dict[str, str] = {}

    def _get_keys(self, obj: RecursiveDict, current_path: str = "", keys: list[str] | None = None):
        if keys is None:
            keys = []
        for key in obj:
            new_path = f"{current_path}.{key}" if current_path else key
            next_ = obj[key]
            if isinstance(next_, dict):
                self._get_keys(next_, new_path, keys)
            elif "_" in key:
                # This typically means its a pluralized key
                continue
            else:
                keys.append(new_path)
        return keys

    def _search_codebase(self, key: str):
        for root, _dirs, files in os.walk("../src"):
            for file in files:
                if file.endswith(".ts") or file.endswith(".tsx"):
                    full_path = os.path.join(root, file)
                    if full_path in self.file_cache:
                        content = self.file_cache[full_path]
                    else:
                        with open(full_path, "r") as f:
                            content = f.read()
                            self.file_cache[full_path] = content

                    # match the whole key, surrounding by quotes
                    if re.search(r"['\"`]" + re.escape(key) + r"['\"`]", self.file_cache[full_path]):
                        return True
                    # math the stem of the key, with quotes at the end
                    if re.search(re.escape(key.split(".")[-1]) + r"['\"`]", self.file_cache[full_path]):
                        return True
        return False

    def _remove_key(self, obj: RecursiveDict, key: str):
        path = key.split(".")
        last_key = path[-1]
        for k in path[:-1]:
            obj = obj[k]
        del obj[last_key]

    def clean(self, obj: RecursiveDict) -> RecursiveDict:
        keys = self._get_keys(obj)
        pbar = tqdm(keys, desc="Checking keys")
        for key in pbar:
            if not self._search_codebase(key):
                self._remove_key(obj, key)
        return obj


def main():
    try:
        with open("../public/locales/en.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Unable to find en.json file - must be run from invokeai/frontend/web/scripts directory"
        ) from e

    cleaner = TranslationCleaner()
    cleaned_data = cleaner.clean(data)

    with open("../public/locales/en.json", "w") as f:
        json.dump(cleaned_data, f, indent=4)


if __name__ == "__main__":
    main()
