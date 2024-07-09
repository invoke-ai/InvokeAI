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

import re

from math import ceil

from .parser import ReadtimeParser

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# Compute readtime - we first used the original readtime library, but the list
# of dependencies it brings with it increased the size of the Docker image by
# 20 MB (packed), which is an increase of 50%. For this reason, we adapt the
# original readtime algorithm to our needs - see https://t.ly/fPZ7L
def readtime(html: str, words_per_minute: int):
    parser = ReadtimeParser()
    parser.feed(html)
    parser.close()

    # Extract words from text and compute readtime in seconds
    words = len(re.split(r"\W+", "".join(parser.text)))
    seconds = ceil(words / words_per_minute * 60)

    # Account for additional images
    delta = 12
    for _ in range(parser.images):
        seconds += delta
        if delta > 3: delta -= 1

    # Return readtime in minutes
    return ceil(seconds / 60)
