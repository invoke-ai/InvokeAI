import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { after, describe, it } from 'node:test';

import { remarkLocalizeAssets } from './remark-localize-assets.mjs';

const docsDir = mkdtempSync(path.join(tmpdir(), 'remark-localize-assets-'));
after(() => rmSync(docsDir, { recursive: true, force: true }));

for (const file of [
  'Users Guide/06.Gallery/assets/board.png',
  'Users Guide/assets/shared.webp',
  'de/Users Guide/06.Gallery/assets/board.png',
]) {
  mkdirSync(path.dirname(path.join(docsDir, file)), { recursive: true });
  writeFileSync(path.join(docsDir, file), '');
}

/**
 * @param {string} source
 * @param {string} [type] the module statement
 */
const importNode = (source, type = 'ImportDeclaration') => ({
  type: 'mdxjsEsm',
  data: { estree: { body: [{ type, source: { type: 'Literal', value: source, raw: `'${source}'` } }] } },
});

/**
 * @param {string} file the page, relative to the docs directory
 * @param {object[]} children
 */
const transform = (file, children) => {
  const tree = { type: 'root', children };
  remarkLocalizeAssets({ docsDir })(tree, { history: [path.join(docsDir, file)] });
  return tree.children;
};

describe('remarkLocalizeAssets', () => {
  it("points a translated page's images, imports and exports at the English page's files", () => {
    const [image, definition, esm, exportAll, exportNamed] = transform('es/Users Guide/06.Gallery/boards.mdx', [
      { type: 'paragraph', children: [{ type: 'image', url: './assets/board.png' }] },
      { type: 'definition', identifier: 'board', url: './assets/board.png#large' },
      importNode('../assets/shared.webp?url'),
      importNode('../assets/shared.webp', 'ExportAllDeclaration'),
      importNode('./assets/board.png', 'ExportNamedDeclaration'),
    ]);
    assert.equal(image.children[0].url, '../../../Users Guide/06.Gallery/assets/board.png');
    assert.equal(definition.url, '../../../Users Guide/06.Gallery/assets/board.png#large');
    assert.deepEqual(esm.data.estree.body[0].source, {
      type: 'Literal',
      value: '../../../Users Guide/assets/shared.webp?url',
      raw: '"../../../Users Guide/assets/shared.webp?url"',
    });
    assert.equal(exportAll.data.estree.body[0].source.value, '../../../Users Guide/assets/shared.webp');
    assert.equal(exportNamed.data.estree.body[0].source.value, '../../../Users Guide/06.Gallery/assets/board.png');
  });

  it("keeps a translation's own files, and paths that are not files of the English page", () => {
    const [own, link, directory, missing, localExport] = transform('de/Users Guide/06.Gallery/boards.mdx', [
      { type: 'image', url: './assets/board.png' },
      { type: 'definition', identifier: 'intro', url: '../introduction' },
      // A relative page link that names a directory.
      { type: 'definition', identifier: 'assets', url: './assets/' },
      importNode('./assets/missing.png'),
      // `export const`, without a source.
      { type: 'mdxjsEsm', data: { estree: { body: [{ type: 'ExportNamedDeclaration', source: null }] } } },
    ]);
    assert.equal(own.url, './assets/board.png');
    assert.equal(link.url, '../introduction');
    assert.equal(directory.url, './assets/');
    assert.equal(missing.data.estree.body[0].source.value, './assets/missing.png');
    assert.equal(localExport.data.estree.body[0].source, null);
  });

  it('leaves English pages and unknown locale directories alone', () => {
    for (const file of ['Users Guide/06.Gallery/boards.mdx', 'it/Users Guide/06.Gallery/boards.mdx']) {
      const [image] = transform(file, [{ type: 'image', url: './assets/board.png' }]);
      assert.equal(image.url, './assets/board.png');
    }
  });
});
