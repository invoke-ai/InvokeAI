"""Details."""
import xml.etree.ElementTree as etree
from .block import Block, type_boolean, type_html_identifier
from ..blocks import BlocksExtension
import re

RE_SEP = re.compile(r'[_-]+')
RE_VALID_NAME = re.compile(r'[\w-]+')


class Details(Block):
    """
    Details.

    Arguments (1 optional):
    - A summary.

    Options:
    - `open` (boolean): force the details block to be in an open state opposed to collapsed.
    - `type` (string): Attach a single special class for styling purposes. If more are needed,
      use the built-in `attributes` options to apply as many classes as desired.

    Content:
    Detail body.
    """

    NAME = 'details'

    ARGUMENT = None
    OPTIONS = {
        'open': [False, type_boolean],
        'type': ['', type_html_identifier]
    }

    DEF_TITLE = None
    DEF_CLASS = None

    def on_validate(self, parent):
        """Handle on validate event."""

        if self.NAME != 'details':
            self.options['type'] = {'name': self.NAME}
            if self.DEF_TITLE:
                self.options['type']['title'] = self.DEF_TITLE
            if self.DEF_TITLE:
                self.options['type']['class'] = self.DEF_CLASS
        return True

    def on_create(self, parent):
        """Create the element."""

        # Is it open?
        attributes = {}
        if self.options['open']:
            attributes['open'] = 'open'

        # Set classes
        obj = self.options['type']
        dtype = def_title = class_name = ''
        if isinstance(obj, dict):
            dtype = obj['name']
            class_name = obj.get('class', dtype)
            def_title = obj.get('title', RE_SEP.sub(' ', class_name).title())
        elif isinstance(obj, str):
            dtype = obj
            class_name = dtype
            def_title = RE_SEP.sub(' ', class_name).title()
        if dtype:
            attributes['class'] = class_name

        # Create Detail element
        el = etree.SubElement(parent, 'details', attributes)

        # Create the summary
        summary = None
        if self.argument is None:
            if dtype:
                summary = def_title
        elif self.argument:
            summary = self.argument

        # Create the summary
        if summary is not None:
            s = etree.SubElement(el, 'summary')
            s.text = summary

        return el


class DetailsExtension(BlocksExtension):
    """Admonition Blocks Extension."""

    def __init__(self, *args, **kwargs):
        """Initialize."""

        self.config = {
            "types": [
                [],
                "Generate Admonition block extensions for the given types."
            ]
        }

        super().__init__(*args, **kwargs)

    def extendMarkdownBlocks(self, md, block_mgr):
        """Extend Markdown blocks."""

        block_mgr.register(Details, self.getConfigs())

        # Generate an details subclass based on the given names.
        for obj in self.getConfig('types', []):
            if isinstance(obj, dict):
                name = obj['name']
                class_name = obj.get('class', name)
                title = obj.get('title', RE_SEP.sub(' ', class_name).title())
            else:
                name = obj
                class_name = name
                title = RE_SEP.sub(' ', class_name).title()
            subclass = RE_SEP.sub('', name).title()
            block_mgr.register(
                type(
                    subclass,
                    (Details,),
                    {
                        'OPTIONS': {'open': [False, type_boolean]},
                        'NAME': name,
                        'DEF_TITLE': title,
                        'DEF_CLASS': class_name
                    }
                ),
                {}
            )


def makeExtension(*args, **kwargs):
    """Return extension."""

    return DetailsExtension(*args, **kwargs)
