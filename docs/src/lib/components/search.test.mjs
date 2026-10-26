import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  RECENT_LIMIT,
  addRecent,
  groupByTopic,
  matchSettings,
  parseRecent,
  resolveScope,
  toPageRows,
} from './search.mjs';

describe('resolveScope', () => {
  const models = { id: 'models', title: 'Models' };
  const pressed = (state) => state.chips.filter((chip) => chip.pressed).map((chip) => chip.scope);

  it('starts in the page topic, with its chip pressed and its filter applied', () => {
    const state = resolveScope(models);
    assert.equal(state.scope, 'topic');
    assert.deepEqual(pressed(state), ['topic']);
    assert.deepEqual(state.filters, { topic: 'models' });
    assert.equal(state.showSettings, false);
    assert.equal(state.announcement, 'Searching in Models');
  });

  it('widens to All docs without a filter, showing settings', () => {
    const state = resolveScope(models, 'all');
    assert.deepEqual(pressed(state), ['all']);
    assert.equal(state.filters, undefined);
    assert.equal(state.showSettings, true);
    assert.equal(state.announcement, 'Searching all docs');
  });

  it('offers only All docs on pages outside every topic, whatever was asked for', () => {
    for (const state of [resolveScope(undefined), resolveScope(undefined, 'topic')]) {
      assert.equal(state.scope, 'all');
      assert.deepEqual(pressed(state), ['all']);
      assert.deepEqual(
        state.chips.filter((chip) => !chip.hidden).map((chip) => chip.scope),
        ['all'],
      );
      assert.equal(state.filters, undefined);
    }
  });

  it('shows settings in the scope of the topic that documents them', () => {
    const state = resolveScope({ id: 'configure', title: 'Configure & Deploy' });
    assert.deepEqual(state.filters, { topic: 'configure' });
    assert.equal(state.showSettings, true);
  });
});

const setting = (key, description = '') => ({ key, type: 'bool', default: 'false', description });

describe('matchSettings', () => {
  const settings = [
    setting('fp8_compute_full_precision_hints', 'Honor per-layer full-precision flags.'),
    setting('ram', 'Maximum memory amount used by the model cache.'),
    setting('fp8_compute', 'Run scaled-fp8 checkpoints on fp8 tensor cores.'),
    setting('max_cache_ram_gb', 'The maximum amount of CPU RAM to use for model caching.'),
    setting('lazy_offload', 'Keep models in VRAM until their space is needed.'),
    setting('port', 'Port to bind to.'),
  ];
  const keys = (query) => matchSettings(settings, query).map((match) => match.key);

  it('ranks an exact key, then key prefixes, then keys containing the query', () => {
    assert.deepEqual(keys('ram'), ['ram', 'max_cache_ram_gb']);
    assert.deepEqual(keys('fp8_compute'), ['fp8_compute', 'fp8_compute_full_precision_hints']);
    assert.deepEqual(keys('cache_ram'), ['max_cache_ram_gb', 'ram']);
  });

  it('treats spaces, hyphens and underscores alike, in any case', () => {
    assert.deepEqual(keys('FP8 compute'), keys('fp8_compute'));
    assert.deepEqual(keys('lazy-offload'), ['lazy_offload']);
  });

  it('finds a setting from its environment variable', () => {
    assert.deepEqual(keys('INVOKEAI_PORT'), ['port']);
  });

  it('matches every word of the query in the key before falling back to descriptions', () => {
    assert.deepEqual(keys('ram max'), ['max_cache_ram_gb', 'ram']);
    assert.deepEqual(keys('vram'), ['lazy_offload']);
    assert.deepEqual(keys('scaled fp8'), ['fp8_compute']);
    assert.deepEqual(keys('tensor cores'), ['fp8_compute']);
  });

  it('ignores queries too short to be useful and words found nowhere', () => {
    assert.deepEqual(keys(' '), []);
    assert.deepEqual(keys('p'), []);
    // Two characters match keys, not descriptions.
    assert.deepEqual(keys('to'), []);
    assert.deepEqual(keys('ram banana'), []);
  });
});

describe('toPageRows', () => {
  const page = {
    url: '/fp8/',
    excerpt: 'page excerpt',
    meta: { title: 'FP8 Storage' },
    filters: { topic: ['configure'] },
    sub_results: [
      { title: 'FP8 Storage', url: '/fp8/', excerpt: 'intro', locations: [1] },
      { title: 'Hardware', url: '/fp8/#hardware', excerpt: 'hw', anchor: { id: 'hardware' }, locations: [5, 6, 7] },
      { title: 'Enabling', url: '/fp8/#enabling', excerpt: 'on', anchor: { id: 'enabling' }, locations: [9, 10] },
    ],
  };

  it('keeps the sections with the most matches, in page order, linking to their anchors', () => {
    assert.deepEqual(toPageRows(page, 2), [
      { url: '/fp8/#hardware', page: 'FP8 Storage', section: 'Hardware', excerpt: 'hw', topic: 'configure' },
      { url: '/fp8/#enabling', page: 'FP8 Storage', section: 'Enabling', excerpt: 'on', topic: 'configure' },
    ]);
  });

  it('shows a match before the first heading as the page itself', () => {
    const [intro] = toPageRows({ ...page, sub_results: page.sub_results.slice(0, 1) }, 2);
    assert.deepEqual(intro, { url: '/fp8/', page: 'FP8 Storage', section: undefined, excerpt: 'intro', topic: 'configure' });
  });

  it('falls back to the page when Pagefind has no sections, and leaves pages outside every topic untagged', () => {
    assert.deepEqual(toPageRows({ url: '/', excerpt: 'home', meta: { title: 'Invoke' } }, 2), [
      { url: '/', page: 'Invoke', excerpt: 'home', topic: undefined },
    ]);
  });
});

describe('groupByTopic', () => {
  it('orders groups by their best row, keeps row order within each group and titles them by topic', () => {
    const titles = new Map([
      ['configure', 'Configure & Deploy'],
      ['models', 'Models'],
    ]);
    const rows = [
      { url: '/a/', topic: 'configure' },
      { url: '/b/', topic: 'models' },
      { url: '/c/' },
      { url: '/d/', topic: 'configure' },
    ];
    assert.deepEqual(
      groupByTopic(rows, titles, 'Other pages').map(({ label, rows: groupRows }) => [
        label,
        groupRows.map((row) => row.url),
      ]),
      [
        ['Configure & Deploy', ['/a/', '/d/']],
        ['Models', ['/b/']],
        ['Other pages', ['/c/']],
      ],
    );
  });
});

describe('recent searches', () => {
  it('puts the latest opened result first, without duplicates, up to the limit', () => {
    let list = [];
    for (let index = 0; index <= RECENT_LIMIT; index++) list = addRecent(list, { title: `Page ${index}`, url: `/${index}/` });
    list = addRecent(list, { title: 'Page 3', section: 'Setup', url: '/3/' });
    assert.deepEqual(
      list.map((entry) => entry.url),
      ['/3/', '/5/', '/4/', '/2/', '/1/'],
    );
    assert.equal(list[0].section, 'Setup');
  });

  it('reads back what it stores and drops malformed storage', () => {
    const stored = [{ title: 'Low VRAM mode', url: '/low-vram/' }, { title: 'ram', section: 'invokeai.yaml', url: '/y/' }];
    assert.deepEqual(parseRecent(JSON.stringify(stored)), stored);
    for (const junk of [null, '', '{', '{"title":"x"}', '[1, {"title": 2, "url": "/"}, {"url": "/"}]']) {
      assert.deepEqual(parseRecent(junk), []);
    }
    assert.deepEqual(parseRecent(JSON.stringify([{ title: 'A', url: '/a/', extra: 1 }, 'x'])), [{ title: 'A', url: '/a/' }]);
  });
});
