# Copyright (c) 2016-2024 Martin Donath <martin.donath@squidfunk.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import os

from mkdocs.config.base import Config
from mkdocs.config.config_options import DictOfItems, Type

# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------

# Privacy plugin configuration
class PrivacyConfig(Config):
    enabled = Type(bool, default = True)
    concurrency = Type(int, default = max(1, os.cpu_count() - 1))

    # Settings for caching
    cache = Type(bool, default = True)
    cache_dir = Type(str, default = ".cache/plugin/privacy")

    # Settings for external assets
    assets = Type(bool, default = True)
    assets_fetch = Type(bool, default = True)
    assets_fetch_dir = Type(str, default = "assets/external")
    assets_expr_map = DictOfItems(Type(str), default = {})
