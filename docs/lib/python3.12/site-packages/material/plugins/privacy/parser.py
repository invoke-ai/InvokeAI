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

from html.parser import HTMLParser
from xml.etree.ElementTree import Element

# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------

# Fragment parser - previously, we used lxml for fault-tolerant HTML5 parsing,
# but it blows up the size of the Docker image by 20 MB. We can't just use the
# built-in XML parser, as it doesn't handle HTML5 (because, yeah, it's not XML),
# so we use a streaming parser and construct the element ourselves.
class FragmentParser(HTMLParser):

    # Initialize parser
    def __init__(self):
        super().__init__(convert_charrefs = True)
        self.result = None

    # Create element
    def handle_starttag(self, tag, attrs):
        self.result = Element(tag, dict(attrs))
