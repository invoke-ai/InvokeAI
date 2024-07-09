"""Admonitions."""
import xml.etree.ElementTree as etree
from .block import Block, type_html_identifier
from .. blocks import BlocksExtension
import re

RE_SEP = re.compile(r'[_-]+')
RE_VALID_NAME = re.compile(r'[\w-]+')


class Admonition(Block):
    """
    Admonition.

    Arguments (1 optional):
    - A title.

    Options:
    - `type` (string): Attach a single special class for styling purposes. If more are needed,
      use the built-in `attributes` options to apply as many classes as desired.

    Content:
    Detail body.
    """

    NAME = 'admonition'
    ARGUMENT = None
    OPTIONS = {
        'type': ['', type_html_identifier],
    }
    DEF_TITLE = None
    DEF_CLASS = None

    def on_validate(self, parent):
        """Handle on validate event."""

        if self.NAME != 'admonition':
            self.options['type'] = {'name': self.NAME}
            if self.DEF_TITLE:
                self.options['type']['title'] = self.DEF_TITLE
            if self.DEF_TITLE:
                self.options['type']['class'] = self.DEF_CLASS
        return True

    def on_create(self, parent):
        """Create the element."""

        # Set classes
        classes = ['admonition']
        obj = self.options['type']
        atype = def_title = class_name = ''
        if isinstance(obj, dict):
            atype = obj['name']
            class_name = obj.get('class', atype)
            def_title = obj.get('title', RE_SEP.sub(' ', class_name).title())
        elif isinstance(obj, str):
            atype = obj
            class_name = atype
            def_title = RE_SEP.sub(' ', atype).title()

        if atype and atype != 'admonition':
            classes.append(class_name)

        # Create the admonition
        el = etree.SubElement(parent, 'div', {'class': ' '.join(classes)})

        # Create the title
        title = None
        if self.argument is None:
            if atype:
                title = def_title
        elif self.argument:
            title = self.argument

        if title is not None:
            ad_title = etree.SubElement(el, 'p', {'class': 'admonition-title'})
            ad_title.text = title

        return el


class AdmonitionExtension(BlocksExtension):
    """Admonition Blocks Extension."""

    def __init__(self, *args, **kwargs):
        """Initialize."""

        self.config = {
            "types": [
                ['note', 'attention', 'caution', 'danger', 'error', 'tip', 'hint', 'warning'],
                "Generate Admonition block extensions for the given types."
            ]
        }

        super().__init__(*args, **kwargs)

    def extendMarkdownBlocks(self, md, block_mgr):
        """Extend Markdown blocks."""

        block_mgr.register(Admonition, self.getConfigs())

        # Generate an admonition subclass based on the given names.
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
                    (Admonition,),
                    {'OPTIONS': {}, 'NAME': name, 'DEF_TITLE': title, 'DEF_CLASS': class_name}
                ),
                {}
            )


def makeExtension(*args, **kwargs):
    """Return extension."""

    return AdmonitionExtension(*args, **kwargs)
