import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { describe, it } from 'node:test';

import {
  ALL_CATEGORIES,
  codeSpans,
  gitCloneUrl,
  packMatches,
  paragraphs,
  searchText,
  sortNodePacks,
} from './community-nodes.mjs';

/** @param {string} id @param {string} title */
const entry = (id, title) => ({ id, data: { title } });

describe('sortNodePacks', () => {
  it('sorts by title ignoring case, then by id', () => {
    const packs = [entry('c', 'curves'), entry('b', 'BiRefNet'), entry('a2', 'Average'), entry('a1', 'average')];
    assert.deepEqual(
      sortNodePacks(packs).map(({ id }) => id),
      ['a1', 'a2', 'b', 'c']
    );
  });

  it('leaves the given array as it was', () => {
    const packs = [entry('z', 'Zeta'), entry('a', 'Alpha')];
    sortNodePacks(packs);
    assert.deepEqual(
      packs.map(({ id }) => id),
      ['z', 'a']
    );
  });
});

describe('gitCloneUrl', () => {
  it('appends .git to a repository page URL', () => {
    assert.equal(gitCloneUrl('https://github.com/JPPhoto/curves-node'), 'https://github.com/JPPhoto/curves-node.git');
  });

  it('drops a trailing slash and keeps an existing .git', () => {
    assert.equal(gitCloneUrl('https://gitlab.com/srcrr/shift3d/'), 'https://gitlab.com/srcrr/shift3d.git');
    assert.equal(gitCloneUrl('https://github.com/o/r.git'), 'https://github.com/o/r.git');
  });
});

describe('codeSpans', () => {
  it('marks backtick spans as code', () => {
    assert.deepEqual(codeSpans('Use `collect` inside `iterate`.'), [
      { code: false, text: 'Use ' },
      { code: true, text: 'collect' },
      { code: false, text: ' inside ' },
      { code: true, text: 'iterate' },
      { code: false, text: '.' },
    ]);
  });

  it('keeps an unpaired backtick and markup as plain text', () => {
    assert.deepEqual(codeSpans('a ` b <i>c</i>'), [{ code: false, text: 'a ` b <i>c</i>' }]);
  });
});

describe('paragraphs', () => {
  it('splits at blank lines and folds line breaks', () => {
    assert.deepEqual(paragraphs('One\nline.\n\n  Two.  \n'), ['One line.', 'Two.']);
  });

  it('is empty for a missing description', () => {
    assert.deepEqual(paragraphs(undefined), []);
  });
});

describe('packMatches', () => {
  const pack = {
    text: searchText(
      {
        title: 'Clothing Mask',
        author: 'VeyDlin',
        summary: 'Segment clothing with a U²-Net model.',
        nodes: [{ name: 'Clothing Mask', description: 'Masks garments' }],
      },
      ['Masks & backgrounds']
    ),
    categories: ['masks'],
  };

  it('shows every pack for an empty query in all categories', () => {
    assert.equal(packMatches(pack, { query: '  ', category: ALL_CATEGORIES }), true);
  });

  it('needs every word, in any order, ignoring case and punctuation', () => {
    assert.equal(packMatches(pack, { query: 'MASK, clothing', category: ALL_CATEGORIES }), true);
    assert.equal(packMatches(pack, { query: 'clothing pose', category: ALL_CATEGORIES }), false);
  });

  it('matches the author, node descriptions and category names', () => {
    assert.equal(packMatches(pack, { query: 'veydlin', category: ALL_CATEGORIES }), true);
    assert.equal(packMatches(pack, { query: 'garments', category: ALL_CATEGORIES }), true);
    assert.equal(packMatches(pack, { query: 'backgrounds', category: ALL_CATEGORIES }), true);
  });

  it('ignores accents in the query and the text', () => {
    const accented = { text: searchText({ title: 'Café Filter', author: 'x', summary: '' }, []), categories: [] };
    assert.equal(packMatches(accented, { query: 'cafe', category: ALL_CATEGORIES }), true);
  });

  it('shows only packs in the chosen category', () => {
    assert.equal(packMatches(pack, { query: '', category: 'masks' }), true);
    assert.equal(packMatches(pack, { query: 'clothing', category: 'prompts' }), false);
  });
});

describe('src/content/community-nodes', () => {
  // The collection loads `*.yaml` only, so a `.yml` pack would vanish from the page without an error.
  // A thumbnail its pack doesn't name would sit in the repository unused.
  it('holds only packs, the thumbnails they name and its README', () => {
    const directory = new URL('../content/community-nodes/', import.meta.url);
    const files = readdirSync(directory);
    const thumbnailOf = (id) =>
      /^thumbnail:\s*\.\/(\S+)\s*$/m.exec(readFileSync(new URL(`${id}.yaml`, directory), 'utf8'))?.[1];
    assert.deepEqual(
      files.filter((name) => {
        if (name === 'README.md') return false;
        const match = /^([a-z0-9]+(?:-[a-z0-9]+)*)\.(yaml|webp)$/.exec(name);
        if (!match) return true;
        return match[2] === 'webp' && (!files.includes(`${match[1]}.yaml`) || thumbnailOf(match[1]) !== name);
      }),
      []
    );
  });
});
