import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { remarkLocalizeContent, testing } from './remark-localize-content.mjs';

const docsRoot = fileURLToPath(new URL('../src/content/docs/', import.meta.url));

test('localizes links and reuses source assets in translated MDX', () => {
  const tree = {
    type: 'root',
    children: [
      { type: 'link', url: '/concepts/prompting-guide/', children: [] },
      { type: 'link', url: 'https://example.com/', children: [] },
      { type: 'image', url: './assets/gallery.png', children: [] },
      {
        type: 'mdxJsxFlowElement',
        children: [
          { type: 'mdxJsxAttribute', name: 'href', value: '/troubleshooting/faq/' },
          { type: 'mdxJsxAttribute', name: 'href', value: '/download/' },
        ],
      },
    ],
  };

  remarkLocalizeContent({ locales: ['de', 'es', 'hi'] })(tree, {
    path: `${docsRoot}es/features/gallery.mdx`,
  });

  assert.equal(tree.children[0].url, '/es/concepts/prompting-guide/');
  assert.equal(tree.children[1].url, 'https://example.com/');
  assert.equal(tree.children[2].url, '../../features/assets/gallery.png');
  assert.equal(tree.children[3].children[0].value, '/es/troubleshooting/faq/');
  assert.equal(tree.children[3].children[1].value, '/download/');
});

test('rewrites relative MDX asset imports to the English source tree', () => {
  const tree = {
    type: 'root',
    children: [
      {
        type: 'mdxjsEsm',
        value: "import splashImage from './assets/invoke-webui-canvas.png';",
        children: [],
      },
    ],
  };

  remarkLocalizeContent({ locales: ['de', 'es', 'hi'] })(tree, {
    path: `${docsRoot}hi/index.mdx`,
  });

  assert.equal(
    tree.children[0].value,
    "import splashImage from '../assets/invoke-webui-canvas.png';",
  );
});

test('leaves English source files unchanged', () => {
  const tree = {
    type: 'root',
    children: [{ type: 'link', url: '/configuration/docker/', children: [] }],
  };

  remarkLocalizeContent({ locales: ['de', 'es', 'hi'] })(tree, {
    path: `${docsRoot}index.mdx`,
  });

  assert.equal(tree.children[0].url, '/configuration/docker/');
  assert.equal(
    testing.getLocalizedFileContext(`${docsRoot}it/index.mdx`, new Set(['de', 'es', 'hi'])),
    undefined,
  );
});
