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

from markdown.treeprocessors import Treeprocessor
from mkdocs.structure.pages import Page
from mkdocs.utils import get_relative_url
from xml.etree.ElementTree import Element

# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------

# Excerpt tree processor
class ExcerptTreeprocessor(Treeprocessor):

    # Initialize excerpt tree processor
    def __init__(self, page: Page, base: Page = None):
        self.page = page
        self.base = base

    # Transform HTML after Markdown processing
    def run(self, root: Element):
        main = True

        # We're only interested in anchors, which is why we continue when the
        # link does not start with an anchor tag
        for el in root.iter("a"):
            anchor = el.get("href")
            if not anchor.startswith("#"):
                continue

            # The main headline should link to the post page, not to a specific
            # anchor, which is why we remove the anchor in that case
            path = get_relative_url(self.page.url, self.base.url)
            if main:
                el.set("href", path)
            else:
                el.set("href", path + anchor)

            # Main headline has been seen
            main = False
