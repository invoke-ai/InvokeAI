import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { rehypeTableScroll } from './rehype-table-scroll.mjs';

const element = (tagName, children = []) => ({ type: 'element', tagName, properties: {}, children });
const text = (value) => ({ type: 'text', value });

describe('rehypeTableScroll', () => {
  it('wraps every table, nested ones included, and nothing else', () => {
    const outer = element('table');
    const nested = element('table');
    const tree = { type: 'root', children: [element('p'), outer, element('aside', [nested])] };
    rehypeTableScroll()(tree);

    const [paragraph, wrapper, aside] = tree.children;
    assert.equal(paragraph.tagName, 'p');
    assert.deepEqual([wrapper.tagName, wrapper.properties.className, wrapper.children], ['div', ['table-scroll'], [outer]]);
    assert.deepEqual(aside.children[0].children, [nested]);
  });

  it("names each table by its caption, else the nearest heading before it, else 'Table'", () => {
    const tree = {
      type: 'root',
      children: [
        element('table'),
        element('h2', [text('Environment '), element('code', [text('variables')])]),
        element('aside', [element('table')]),
        element('table', [element('caption', [text('Support tiers')])]),
      ],
    };
    rehypeTableScroll()(tree);
    const labels = [tree.children[0], tree.children[2].children[0], tree.children[3]].map((w) => w.properties.dataLabel);
    assert.deepEqual(labels, ['Table', 'Environment variables', 'Support tiers']);
  });
});
