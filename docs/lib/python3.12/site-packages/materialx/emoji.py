"""
Emoji extras for Material.

Override the indexes with an extended version that includes short names for Material icons, FontAwesome, etc.
"""
import os
import glob
import copy
import codecs
import functools
import inspect
import material
import pymdownx
from pymdownx.emoji import TWEMOJI_SVG_CDN, add_attriubtes
import xml.etree.ElementTree as etree  # noqa: N813
import warnings
from functools import wraps
import logging

log = logging.getLogger('mkdocs')

DEPRECATED = """\
Material emoji logic has been officially moved into mkdocs-material
version 9.4. Please use Material's '{}'
instead of '{}' in your 'mkdocs.yml' file.

```
markdown_extensions:
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
      emoji_generator: !!python/name:material.extensions.emoji.to_svg
```

'mkdocs_material_extensions' is deprecated and will no longer be
supported moving forward. This is the last release.
"""


OPTION_SUPPORT = pymdownx.__version_info__ >= (7, 1, 0)
RESOURCES = os.path.dirname(inspect.getfile(material))
if os.path.exists(os.path.join(RESOURCES, 'templates', '.icons')):  # pragma: no cover
    RES_PATH = os.path.join(RESOURCES, 'templates', '.icons')
else:  # pragma: no cover
    RES_PATH = os.path.join(RESOURCES, '.icons')


@functools.lru_cache(maxsize=None)
def log_msg(message):
    """Log message."""

    log.warning(message)


def deprecated(message, stacklevel=2, name=None):  # pragma: no cover
    """
    Raise a `DeprecationWarning` when wrapped function/method is called.

    Usage:

        @deprecated("This method will be removed in version X; use Y instead.")
        def some_method()"
            pass
    """

    def _wrapper(func):
        @wraps(func)
        def _deprecated_func(*args, **kwargs):
            warnings.warn(
                f"'{func.__name__ if name is None else name}' is deprecated.\n{message}",
                category=DeprecationWarning,
                stacklevel=stacklevel
            )

            log_msg(message)
            return func(*args, **kwargs)
        return _deprecated_func
    return _wrapper


@deprecated(
    DEPRECATED.format('material.extensions.emoji.twemoji', 'materialx.emoji.twemoji'),
    name='materialx.emoji.twemoji'
)
def _patch_index(options):
    """Patch the given index."""

    icon_locations = options.get('custom_icons', [])[:]
    icon_locations.append(RES_PATH)
    return _patch_index_for_locations(tuple(icon_locations))


@functools.lru_cache(maxsize=None)
def _patch_index_for_locations(icon_locations):
    import pymdownx.twemoji_db as twemoji_db

    # Copy the Twemoji index
    index = {
        "name": 'twemoji',
        "emoji": copy.deepcopy(twemoji_db.emoji) if not OPTION_SUPPORT else twemoji_db.emoji,
        "aliases": copy.deepcopy(twemoji_db.aliases) if not OPTION_SUPPORT else twemoji_db.aliases
    }

    # Find our icons
    for icon_path in icon_locations:
        norm_base = icon_path.replace('\\', '/') + '/'
        for result in glob.glob(glob.escape(icon_path.replace('\\', '/')) + '/**/*.svg', recursive=True):
            name = ':{}:'.format(result.replace('\\', '/').replace(norm_base, '', 1).replace('/', '-').lstrip('.')[:-4])
            if name not in index['emoji'] and name not in index['aliases']:
                # Easiest to just store the path and pull it out from the index
                index["emoji"][name] = {'name': name, 'path': result}
    return index


if OPTION_SUPPORT:  # pragma: no cover
    def twemoji(options, md):
        """Provide a copied Twemoji index with additional codes for Material included icons."""

        return _patch_index(options)

else:  # pragma: no cover
    def twemoji():
        """Provide a copied Twemoji index with additional codes for Material included icons."""

        return _patch_index({})


@deprecated(
    DEPRECATED.format('material.extensions.emoji.to_svg', 'materialx.emoji.to_svg'),
    1,
    name='materialx.emoji.to_svg'
)
def to_svg(index, shortname, alias, uc, alt, title, category, options, md):
    """Return SVG element."""

    is_unicode = uc is not None

    if is_unicode:
        # Handle Twemoji emoji.
        svg_path = TWEMOJI_SVG_CDN

        attributes = {
            "class": options.get('classes', index),
            "alt": alt,
            "src": "%s%s.svg" % (
                options.get('image_path', svg_path),
                uc
            )
        }

        if title:
            attributes['title'] = title

        add_attriubtes(options, attributes)

        return etree.Element("img", attributes)
    else:
        # Handle Material SVG assets.
        el = etree.Element('span', {"class": options.get('classes', index)})
        svg_path = md.inlinePatterns['emoji'].emoji_index['emoji'][shortname]['path']
        with codecs.open(svg_path, 'r', encoding='utf-8') as f:
            el.text = md.htmlStash.store(f.read())
        return el
