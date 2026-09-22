import assert from 'node:assert/strict';
import test from 'node:test';

import { rehypePrefixBaseToRootLinks } from './rehype-prefix-base-to-root-links.mjs';

const anchor = (href) => ({ type: 'element', tagName: 'a', properties: { href }, children: [] });

const jsxElement = (name, attributes) => ({
  type: 'mdxJsxFlowElement',
  name,
  attributes: attributes.map(([attrName, value]) => ({ type: 'mdxJsxAttribute', name: attrName, value })),
  children: [],
});

test('prefixes the base onto root-relative anchor hrefs', () => {
  const tree = {
    type: 'root',
    children: [
      anchor('/start-here/installation/'),
      anchor('/InvokeAI/concepts/'),
      anchor('https://example.com/'),
      anchor('#section'),
      anchor('//cdn.example.com/asset.js'),
    ],
  };

  rehypePrefixBaseToRootLinks({ base: '/InvokeAI' })(tree);

  assert.equal(tree.children[0].properties.href, '/InvokeAI/start-here/installation/');
  assert.equal(tree.children[1].properties.href, '/InvokeAI/concepts/');
  assert.equal(tree.children[2].properties.href, 'https://example.com/');
  assert.equal(tree.children[3].properties.href, '#section');
  assert.equal(tree.children[4].properties.href, '//cdn.example.com/asset.js');
});

test('prefixes the base onto MDX component href and link attributes', () => {
  const tree = {
    type: 'root',
    children: [
      jsxElement('LinkButton', [['href', '/download/']]),
      jsxElement('LinkCard', [['link', '/de/features/gallery/']]),
      jsxElement('LinkCard', [['href', '/InvokeAI/troubleshooting/faq/']]),
      jsxElement('Image', [['src', '/assets/splash.png']]),
    ],
  };

  rehypePrefixBaseToRootLinks({ base: '/InvokeAI' })(tree);

  assert.equal(tree.children[0].attributes[0].value, '/InvokeAI/download/');
  assert.equal(tree.children[1].attributes[0].value, '/InvokeAI/de/features/gallery/');
  // Already based — must not double up.
  assert.equal(tree.children[2].attributes[0].value, '/InvokeAI/troubleshooting/faq/');
  // Only href/link are link attributes; src is left to Astro's asset pipeline.
  assert.equal(tree.children[3].attributes[0].value, '/assets/splash.png');
});

test('leaves expression-valued attributes alone', () => {
  // `href={withBase('/download/', BASE_URL)}` parses to an expression, not a string, and the
  // component already applies the base itself.
  const expressionAttribute = {
    type: 'mdxJsxAttribute',
    name: 'href',
    value: { type: 'mdxJsxAttributeValueExpression', value: "withBase('/download/')" },
  };
  const tree = {
    type: 'root',
    children: [{ type: 'mdxJsxFlowElement', name: 'LinkButton', attributes: [expressionAttribute], children: [] }],
  };

  rehypePrefixBaseToRootLinks({ base: '/InvokeAI' })(tree);

  assert.deepEqual(tree.children[0].attributes[0].value, {
    type: 'mdxJsxAttributeValueExpression',
    value: "withBase('/download/')",
  });
});

test('is a no-op when no base is configured', () => {
  const tree = {
    type: 'root',
    children: [anchor('/start-here/installation/'), jsxElement('LinkButton', [['href', '/download/']])],
  };

  rehypePrefixBaseToRootLinks({ base: '/' })(tree);

  assert.equal(tree.children[0].properties.href, '/start-here/installation/');
  assert.equal(tree.children[1].attributes[0].value, '/download/');
});

test('rewrites links nested inside MDX components', () => {
  const tree = {
    type: 'root',
    children: [
      {
        type: 'mdxJsxFlowElement',
        name: 'Card',
        attributes: [],
        children: [anchor('/concepts/prompting-guide/')],
      },
    ],
  };

  rehypePrefixBaseToRootLinks({ base: '/InvokeAI' })(tree);

  assert.equal(tree.children[0].children[0].properties.href, '/InvokeAI/concepts/prompting-guide/');
});
