"""
Tabbed.

pymdownx.tabbed

MIT license.

Copyright (c) 2017 Isaac Muse <isaacmuse@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from markdown import Extension
from markdown.blockprocessors import BlockProcessor
from markdown.treeprocessors import Treeprocessor
from markdown.extensions import toc
import xml.etree.ElementTree as etree
import re
import html

HEADERS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}


class TabbedProcessor(BlockProcessor):
    """Tabbed block processor."""

    START = re.compile(
        r'(?:^|\n)={3}(\+|\+!|!\+|!)? +"(.*?)" *(?:\n|$)'
    )
    COMPRESS_SPACES = re.compile(r' {2,}')

    def __init__(self, parser, config):
        """Initialize."""

        super().__init__(parser)
        self.tab_group_count = 0
        self.current_sibling = None
        self.content_indention = 0
        self.alternate_style = config['alternate_style']
        self.slugify = callable(config['slugify'])

    def detab_by_length(self, text, length):
        """Remove a tab from the front of each line of the given text."""

        newtext = []
        lines = text.split('\n')
        for line in lines:
            if line.startswith(' ' * length):
                newtext.append(line[length:])
            elif not line.strip():
                newtext.append('')  # pragma: no cover
            else:
                break
        return '\n'.join(newtext), '\n'.join(lines[len(newtext):])

    def parse_content(self, parent, block):
        """
        Get sibling tab.

        Retrieve the appropriate sibling element. This can get tricky when
        dealing with lists.

        """

        old_block = block
        non_tabs = ''
        tabbed_set = 'tabbed-set' if not self.alternate_style else 'tabbed-set tabbed-alternate'

        # We already acquired the block via test
        if self.current_sibling is not None:
            sibling = self.current_sibling
            block, non_tabs = self.detab_by_length(block, self.content_indent)
            self.current_sibling = None
            self.content_indent = 0
            return sibling, block, non_tabs

        sibling = self.lastChild(parent)

        if sibling is None or sibling.tag.lower() != 'div' or sibling.attrib.get('class', '') != tabbed_set:
            sibling = None
        else:
            # If the last child is a list and the content is indented sufficient
            # to be under it, then the content's is sibling is in the list.
            if self.alternate_style:
                last_child = self.lastChild(self.lastChild(sibling))
                tabbed_content = 'tabbed-block'
            else:
                last_child = self.lastChild(sibling)
                tabbed_content = 'tabbed-content'
            child_class = last_child.attrib.get('class', '') if last_child is not None else ''
            indent = 0
            while last_child is not None:
                if (
                    sibling is not None and block.startswith(' ' * self.tab_length * 2) and
                    last_child is not None and (
                        last_child.tag in ('ul', 'ol', 'dl') or
                        (
                            last_child.tag == 'div' and
                            child_class == tabbed_content
                        )
                    )
                ):

                    # Handle nested tabbed content
                    if last_child.tag == 'div' and child_class == tabbed_content:
                        temp_child = self.lastChild(last_child)
                        if temp_child is None or temp_child.tag not in ('ul', 'ol', 'dl'):
                            break
                        last_child = temp_child
                        child_class = last_child.attrib.get('class', '') if last_child is not None else ''

                    # The expectation is that we'll find an `<li>`.
                    # We should get it's last child as well.
                    sibling = self.lastChild(last_child)
                    last_child = self.lastChild(sibling) if sibling is not None else None
                    child_class = last_child.attrib.get('class', '') if last_child is not None else ''

                    # Context has been lost at this point, so we must adjust the
                    # text's indentation level so it will be evaluated correctly
                    # under the list.
                    block = block[self.tab_length:]
                    indent += self.tab_length
                else:
                    last_child = None

            if not block.startswith(' ' * self.tab_length):
                sibling = None

            if sibling is not None:
                indent += self.tab_length
                block, non_tabs = self.detab_by_length(old_block, indent)
                self.current_sibling = sibling
                self.content_indent = indent

        return sibling, block, non_tabs

    def test(self, parent, block):
        """Test block."""

        if self.START.search(block):
            return True
        else:
            return self.parse_content(parent, block)[0] is not None

    def run(self, parent, blocks):
        """Convert to tabbed block."""

        block = blocks.pop(0)
        m = self.START.search(block)
        tabbed_set = 'tabbed-set' if not self.alternate_style else 'tabbed-set tabbed-alternate'

        if m:
            # removes the first line
            if m.start() > 0:
                self.parser.parseBlocks(parent, [block[:m.start()]])
            block = block[m.end():]
            sibling = self.lastChild(parent)
            block, non_tabs = self.detab(block)
        else:
            sibling, block, non_tabs = self.parse_content(parent, block)

        if m:
            special = m.group(1) if m.group(1) else ''
            title = m.group(2) if m.group(2) else m.group(3)
            index = 0
            labels = None
            content = None

            if (
                sibling is not None and sibling.tag.lower() == 'div' and
                sibling.attrib.get('class', '') == tabbed_set and
                '!' not in special
            ):
                first = False
                tab_group = sibling
                if self.alternate_style:
                    index = [index for index, _ in enumerate(tab_group.findall('input'), 1)][-1]
                    for d in tab_group.findall('div'):
                        if d.attrib['class'] == 'tabbed-labels':
                            labels = d
                        elif d.attrib['class'] == 'tabbed-content':
                            content = d
                        if labels is not None and content is not None:
                            break
            else:
                first = True
                self.tab_group_count += 1
                tab_group = etree.SubElement(
                    parent,
                    'div',
                    {'class': tabbed_set, 'data-tabs': '%d:0' % self.tab_group_count}
                )
                if self.alternate_style:
                    labels = etree.SubElement(
                        tab_group,
                        'div',
                        {'class': 'tabbed-labels'}
                    )
                    content = etree.SubElement(
                        tab_group,
                        'div',
                        {'class': 'tabbed-content'}
                    )

            data = tab_group.attrib['data-tabs'].split(':')
            tab_set = int(data[0])
            tab_count = int(data[1]) + 1

            attributes = {
                "name": "__tabbed_%d" % tab_set,
                "type": "radio"
            }

            if not self.slugify:
                attributes['id'] = "__tabbed_%d_%d" % (tab_set, tab_count)

            if first or '+' in special:
                attributes['checked'] = 'checked'
                # Remove any previously assigned "checked states" to siblings
                for i in tab_group.findall('input'):
                    if i.attrib.get('name', '') == '__tabbed_{}'.format(tab_set):
                        if 'checked' in i.attrib:
                            del i.attrib['checked']

            attributes2 = {"for": "__tabbed_%d_%d" % (tab_set, tab_count)} if not self.slugify else {}

            if self.alternate_style:
                input_el = etree.Element(
                    'input',
                    attributes
                )
                tab_group.insert(index, input_el)
                lab = etree.SubElement(
                    labels,
                    "label",
                    attributes2
                )
                lab.text = title

                div = etree.SubElement(
                    content,
                    "div",
                    {'class': 'tabbed-block'}
                )
            else:
                etree.SubElement(
                    tab_group,
                    'input',
                    attributes
                )
                lab = etree.SubElement(
                    tab_group,
                    "label",
                    attributes2
                )
                lab.text = title

                div = etree.SubElement(
                    tab_group,
                    "div",
                    {
                        "class": "tabbed-content"
                    }
                )

            tab_group.attrib['data-tabs'] = '%d:%d' % (tab_set, tab_count)
        else:
            if sibling.tag in ('li', 'dd') and sibling.text:
                # Sibling is a list item, but we need to wrap it's content should be wrapped in <p>
                text = sibling.text
                sibling.text = ''
                p = etree.SubElement(sibling, 'p')
                p.text = text
                div = sibling
            elif sibling.tag == 'div' and sibling.attrib.get('class', '') == tabbed_set:
                # Get `tabbed-content` under `tabbed-set`
                if self.alternate_style:
                    div = self.lastChild(self.lastChild(sibling))
                else:
                    div = self.lastChild(sibling)
            else:
                # Pass anything else as the parent
                div = sibling

        self.parser.parseChunk(div, block)

        if non_tabs:
            # Insert the tabbed content back into blocks
            blocks.insert(0, non_tabs)


class TabbedTreeprocessor(Treeprocessor):
    """Tab tree processor."""

    def __init__(self, md, config):
        """Initialize."""

        super().__init__(md)

        self.slugify = config["slugify"]
        self.alternate = config["alternate_style"]
        self.sep = config["separator"]
        self.combine_header_slug = config["combine_header_slug"]

    def get_parent_header_slug(self, root, header_map, parent_map, el):
        """Attempt retrieval of parent header slug."""

        parent = el
        last_parent = parent
        while parent is not root:
            last_parent = parent
            parent = parent_map[parent]
            if parent in header_map:
                headers = header_map[parent]
                header = None
                for i in list(parent):
                    if i is el and header is None:
                        break
                    if i is last_parent:
                        return header.attrib.get("id", '')
                    if i in headers:
                        header = i
        return ''

    def run(self, doc):
        """Update tab IDs."""

        # Get a list of id attributes
        used_ids = set()
        parent_map = {}
        header_map = {}

        if self.combine_header_slug:
            parent_map = {c: p for p in doc.iter() for c in p}

        for el in doc.iter():
            if "id" in el.attrib:
                if self.combine_header_slug and el.tag in HEADERS:
                    parent = parent_map[el]
                    if parent in header_map:
                        header_map[parent].append(el)
                    else:
                        header_map[parent] = [el]
                used_ids.add(el.attrib["id"])

        for el in doc.iter():
            if isinstance(el.tag, str) and el.tag.lower() == 'div':
                classes = el.attrib.get('class', '').split()
                if 'tabbed-set' in classes and (not self.alternate or 'tabbed-alternate' in classes):
                    inputs = []
                    labels = []
                    if self.alternate:
                        for i in list(el):
                            if i.tag == 'input':
                                inputs.append(i)
                            if i.tag == 'div' and i.attrib.get('class', '') == 'tabbed-labels':
                                labels = [j for j in list(i) if j.tag == 'label']
                    else:
                        for i in list(el):
                            if i.tag == 'input':
                                inputs.append(i)
                            if i.tag == 'label':
                                labels.append(i)

                    # Generate slugged IDs
                    for inpt, label in zip(inputs, labels):
                        innerhtml = toc.render_inner_html(toc.remove_fnrefs(label), self.md)
                        innertext = html.unescape(toc.strip_tags(innerhtml))
                        if self.combine_header_slug:
                            parent_slug = self.get_parent_header_slug(doc, header_map, parent_map, el)
                        else:
                            parent_slug = ''
                        slug = self.slugify(innertext, self.sep)
                        if parent_slug:
                            slug = parent_slug + self.sep + slug
                        slug = toc.unique(slug, used_ids)
                        inpt.attrib["id"] = slug
                        label.attrib["for"] = slug


class TabbedExtension(Extension):
    """Add Tabbed extension."""

    def __init__(self, *args, **kwargs):
        """Initialize."""

        self.config = {
            'alternate_style': [False, "Use alternate style - Default: False"],
            'slugify': [0, "Slugify function used to create tab specific IDs - Default: None"],
            'combine_header_slug': [False, "Combine the tab slug with the slug of the parent header - Default: False"],
            'separator': ['-', "Slug separator - Default: '-'"]
        }

        super().__init__(*args, **kwargs)

    def extendMarkdown(self, md):
        """Add Tabbed to Markdown instance."""
        md.registerExtension(self)

        config = self.getConfigs()
        self.tab_processor = TabbedProcessor(md.parser, config)
        md.parser.blockprocessors.register(self.tab_processor, "tabbed", 105)
        if config['slugify']:
            slugs = TabbedTreeprocessor(md, config)
            md.treeprocessors.register(slugs, 'tab_slugs', 4)

    def reset(self):
        """Reset."""

        self.tab_processor.tab_group_count = 0


def makeExtension(*args, **kwargs):
    """Return extension."""

    return TabbedExtension(*args, **kwargs)
