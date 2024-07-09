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

from collections.abc import Callable
from mkdocs.config.config_options import Choice, Deprecated, Optional, Type
from mkdocs.config.base import Config
from pymdownx.slugs import slugify

# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------

# Blog plugin configuration
class BlogConfig(Config):
    enabled = Type(bool, default = True)

    # Settings for blog
    blog_dir = Type(str, default = "blog")
    blog_toc = Type(bool, default = False)

    # Settings for posts
    post_dir = Type(str, default = "{blog}/posts")
    post_date_format = Type(str, default = "long")
    post_url_date_format = Type(str, default = "yyyy/MM/dd")
    post_url_format = Type(str, default = "{date}/{slug}")
    post_url_max_categories = Type(int, default = 1)
    post_slugify = Type(Callable, default = slugify(case = "lower"))
    post_slugify_separator = Type(str, default = "-")
    post_excerpt = Choice(["optional", "required"], default = "optional")
    post_excerpt_max_authors = Type(int, default = 1)
    post_excerpt_max_categories = Type(int, default = 5)
    post_excerpt_separator = Type(str, default = "<!-- more -->")
    post_readtime = Type(bool, default = True)
    post_readtime_words_per_minute = Type(int, default = 265)

    # Settings for archive
    archive = Type(bool, default = True)
    archive_name = Type(str, default = "blog.archive")
    archive_date_format = Type(str, default = "yyyy")
    archive_url_date_format = Type(str, default = "yyyy")
    archive_url_format = Type(str, default = "archive/{date}")
    archive_toc = Optional(Type(bool))

    # Settings for categories
    categories = Type(bool, default = True)
    categories_name = Type(str, default = "blog.categories")
    categories_url_format = Type(str, default = "category/{slug}")
    categories_slugify = Type(Callable, default = slugify(case = "lower"))
    categories_slugify_separator = Type(str, default = "-")
    categories_allowed = Type(list, default = [])
    categories_toc = Optional(Type(bool))

    # Settings for authors
    authors = Type(bool, default = True)
    authors_file = Type(str, default = "{blog}/.authors.yml")

    # Settings for pagination
    pagination = Type(bool, default = True)
    pagination_per_page = Type(int, default = 10)
    pagination_url_format = Type(str, default = "page/{page}")
    pagination_format = Type(str, default = "~2~")
    pagination_if_single_page = Type(bool, default = False)
    pagination_keep_content = Type(bool, default = False)

    # Settings for drafts
    draft = Type(bool, default = False)
    draft_on_serve = Type(bool, default = True)
    draft_if_future_date = Type(bool, default = False)

    # Deprecated settings
    pagination_template = Deprecated(moved_to = "pagination_format")
