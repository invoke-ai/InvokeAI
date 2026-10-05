import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  addCrossListings,
  CROSS_LISTED_ATTR,
  fileDirectory,
  makeHrefToId,
  paginate,
  selectCrossListings,
  sortAndRelabel,
} from './sidebar.mjs';
import { generateDocsId } from './sort-prefix.mjs';

const hrefToId = makeHrefToId('/');

/** @param {string} id */
const link = (id, label = id.split('/').pop(), isCurrent = false) => ({
  type: 'link',
  href: `/${id}/`,
  label,
  isCurrent,
  attrs: {},
});

/** @param {string} label @param {string} directory @param {object[]} entries */
const group = (label, directory, entries) => ({ type: 'group', label, autogenerate: { directory }, entries });

/** @param {object[]} entries */
const labels = (entries) => entries.map((entry) => entry.label);

describe('generateDocsId', () => {
  it('strips sort prefixes from every path segment', () => {
    assert.equal(
      generateDocsId({ entry: 'Users Guide/04.Image Generation/01.introduction.mdx', data: {} }),
      'users-guide/image-generation/introduction'
    );
  });

  it('prefers a frontmatter slug, but ignores an empty one', () => {
    assert.equal(generateDocsId({ entry: '01.a.mdx', data: { slug: 'custom/path' } }), 'custom/path');
    assert.equal(generateDocsId({ entry: '01.a.mdx', data: { slug: '' } }), 'a');
  });
});

describe('sortAndRelabel', () => {
  it('orders prefixed siblings by prefix, keeps unprefixed ones after them in order, and strips group labels', () => {
    const fileNames = new Map([
      ['guide/b', 'b.mdx'],
      ['guide/intro', '01.intro.mdx'],
      ['guide/a', 'a.mdx'],
      ['guide/setup', '02.setup.mdx'],
    ]);
    const entries = [
      { ...link('guide/b'), autogenerate: { directory: 'guide' } },
      { ...link('guide/setup'), autogenerate: { directory: 'guide' } },
      { ...link('guide/a'), autogenerate: { directory: 'guide' } },
      { ...link('guide/intro'), autogenerate: { directory: 'guide' } },
    ];
    assert.equal(sortAndRelabel(entries, fileNames, hrefToId), true);
    assert.deepEqual(labels(entries), ['intro', 'setup', 'b', 'a']);
  });

  it('sorts prefixed folders and recurses into them', () => {
    const fileNames = new Map([
      ['models/introduction', '01.introduction.mdx'],
      ['models/zoo', 'zoo.mdx'],
    ]);
    const models = group('03.Models', 'Users Guide', [
      { ...link('models/zoo'), autogenerate: { directory: 'models' } },
      { ...link('models/introduction'), autogenerate: { directory: 'models' } },
    ]);
    const workbench = group('01.Workbench', 'Users Guide', []);
    const entries = [models, workbench];
    sortAndRelabel(entries, fileNames, hrefToId);
    assert.deepEqual(labels(entries), ['Workbench', 'Models']);
    assert.deepEqual(labels(models.entries), ['introduction', 'zoo']);
  });

  it('reports no change when the order is already right', () => {
    const entries = [{ ...link('x/a'), autogenerate: { directory: 'x' } }];
    assert.equal(sortAndRelabel(entries, new Map([['x/a', '01.a.mdx']]), hrefToId), false);
  });
});

describe('fileDirectory', () => {
  it('maps a file to its directory URL, independent of slugs and index pages', () => {
    assert.equal(fileDirectory('src/content/docs/Users Guide/03.Models/Local Models/krea-2.mdx'), 'users-guide/models/local-models');
    assert.equal(fileDirectory('src/content/docs/development/Documentation/index.mdx'), 'development/documentation');
    assert.equal(fileDirectory('src/content/docs/index.mdx'), '');
  });
});

describe('selectCrossListings', () => {
  const docs = [
    { id: 'video/wan', data: { alsoListedIn: ['/models/local/'] } },
    { id: 'video/hidden', data: { alsoListedIn: ['models/local'], sidebar: { hidden: true } } },
    { id: 'video/draft', data: { alsoListedIn: ['models/local'], draft: true } },
    { id: 'video/plain', data: {} },
  ];

  it('normalizes directories and skips hidden pages', () => {
    assert.deepEqual(selectCrossListings(docs, false), [
      { id: 'video/wan', directory: 'models/local' },
      { id: 'video/draft', directory: 'models/local' },
    ]);
  });

  it('also skips drafts in production, where Starlight leaves them out of the sidebar', () => {
    assert.deepEqual(selectCrossListings(docs, true), [{ id: 'video/wan', directory: 'models/local' }]);
  });
});

describe('addCrossListings', () => {
  const makeSidebar = () => [
    group('Models', 'models', [
      link('models/local/anima', 'Anima'),
      link('models/local/sdxl', 'SDXL'),
      group('Nested', 'models/local/nested', [link('models/local/nested/x', 'X')]),
    ]),
    group('Video', 'video', [link('video/ltx', 'LTX-2', true), link('video/wan', 'Wan 2.2')]),
  ];
  const directories = new Map([
    ['models/local/anima', 'models/local'],
    ['models/local/sdxl', 'models/local'],
    ['models/local/nested/x', 'models/local/nested'],
    ['video/ltx', 'video'],
    ['video/wan', 'video'],
  ]);

  it('inserts copies alphabetically, before subgroups, without the current-page state', () => {
    const sidebar = makeSidebar();
    const listings = [
      { id: 'video/wan', directory: 'models/local' },
      { id: 'video/ltx', directory: 'models/local' },
    ];
    assert.equal(addCrossListings(sidebar, listings, directories, hrefToId), true);
    const models = sidebar[0].entries;
    assert.deepEqual(labels(models), ['Anima', 'LTX-2', 'SDXL', 'Wan 2.2', 'Nested']);
    const ltxCopy = models[1];
    assert.equal(ltxCopy.isCurrent, false);
    assert.ok(CROSS_LISTED_ATTR in ltxCopy.attrs);
    // The original keeps its state and gains no marker.
    assert.equal(sidebar[1].entries[0].isCurrent, true);
    assert.ok(!(CROSS_LISTED_ATTR in sidebar[1].entries[0].attrs));
  });

  it('matches the target group by file directory, not by URL', () => {
    // A custom slug moves the page's URL out of its folder; the folder still decides its group.
    const sidebar = [group('Docs', 'docs', [link('elsewhere/page', 'Page')]), group('Video', 'video', [link('video/wan', 'Wan')])];
    const dirs = new Map([
      ['elsewhere/page', 'docs/folder'],
      ['video/wan', 'video'],
    ]);
    addCrossListings(sidebar, [{ id: 'video/wan', directory: 'docs/folder' }], dirs, hrefToId);
    assert.deepEqual(labels(sidebar[0].entries), ['Page', 'Wan']);
  });

  it('does nothing without listings', () => {
    assert.equal(addCrossListings(makeSidebar(), [], directories, hrefToId), false);
  });

  it('fails loudly on misconfiguration', () => {
    assert.throws(
      () => addCrossListings(makeSidebar(), [{ id: 'video/missing', directory: 'models/local' }], directories, hrefToId),
      /has no sidebar link/
    );
    assert.throws(
      () => addCrossListings(makeSidebar(), [{ id: 'video/wan', directory: 'video' }], directories, hrefToId),
      /its own directory/
    );
    assert.throws(
      () => addCrossListings(makeSidebar(), [{ id: 'video/wan', directory: 'models' }], directories, hrefToId),
      /no sidebar group with pages/
    );
  });

  it('honours a deployment base path', () => {
    const baseHrefToId = makeHrefToId('/InvokeAI/');
    const sidebar = [
      group('Models', 'models', [{ ...link('models/local/sdxl', 'SDXL'), href: '/InvokeAI/models/local/sdxl/' }]),
      group('Video', 'video', [{ ...link('video/wan', 'Wan'), href: '/InvokeAI/video/wan/' }]),
    ];
    addCrossListings(sidebar, [{ id: 'video/wan', directory: 'models/local' }], directories, baseHrefToId);
    assert.deepEqual(labels(sidebar[0].entries), ['SDXL', 'Wan']);
  });
});

describe('paginate', () => {
  const sidebar = () => {
    const entries = [
      group('Models', 'models', [link('models/a', 'A'), link('models/b', 'B')]),
      group('Video', 'video', [link('video/wan', 'Wan', true), link('video/ltx', 'LTX')]),
    ];
    // Cross-listed copies right before and after the current page, which pagination must step over.
    const copy = (id, label) => ({ ...link(id, label), attrs: { [CROSS_LISTED_ATTR]: '' } });
    entries[1].entries.splice(0, 0, copy('models/a', 'A'));
    entries[1].entries.splice(2, 0, copy('models/b', 'B'));
    return entries;
  };

  it('follows sidebar order and skips cross-listed copies', () => {
    const { prev, next } = paginate(sidebar(), {});
    assert.equal(prev.href, '/models/b/');
    assert.ok(!(CROSS_LISTED_ATTR in prev.attrs));
    assert.equal(next.label, 'LTX');
  });

  it('applies prev/next frontmatter', () => {
    const { prev, next } = paginate(sidebar(), { prev: false, next: 'Onward' });
    assert.equal(prev, undefined);
    assert.equal(next.label, 'Onward');
    assert.equal(next.href, '/video/ltx/');
    const custom = paginate(sidebar(), { next: { link: '/elsewhere/', label: 'Elsewhere' } });
    assert.equal(custom.next.href, '/elsewhere/');
  });

  it('returns null when the current page is not in the sidebar', () => {
    assert.equal(paginate([group('Models', 'models', [link('models/a')])], {}), null);
  });
});
