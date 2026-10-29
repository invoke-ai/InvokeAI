import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { buildAppSearchIndex, formatAppSearchIndex } from './app-search.mjs';

const topic = (label, group, directory, link = `/${directory}/`) => ({
  label,
  description: `${label} docs`,
  group,
  link,
  directory,
});

const topics = [
  topic('Get Started', 'Use', 'start-here', '/start-here/installation/'),
  topic('Guide', 'Use', 'users-guide'),
  topic('Models', 'Use', 'users-guide/models'),
  topic('Developers', 'Build', 'development'),
  topic('Troubleshooting', 'Help', 'troubleshooting', '/troubleshooting/faq/'),
];

const doc = (id, filePath, data = {}) => ({
  id,
  filePath: `src/content/docs/${filePath}`,
  data: { title: id, ...data },
});

const build = (docs, headingsById = {}) =>
  buildAppSearchIndex(docs, { topics, getHeadings: async (entry) => headingsById[entry.id] ?? [] });

describe('buildAppSearchIndex', () => {
  it('leaves out drafts, pages the site search skips, the 404 page, splash pages and developer topics', async () => {
    const { pages } = await build([
      doc('start-here/installation', 'start-here/01.installation.mdx'),
      doc('start-here/draft', 'start-here/draft.md', { draft: true }),
      doc('start-here/hidden', 'start-here/hidden.md', { pagefind: false }),
      doc('404', '404.md'),
      doc('', 'index.mdx', { template: 'splash' }),
      doc('development/architecture', 'development/architecture.md'),
    ]);
    assert.deepEqual(
      pages.map((page) => page.s),
      ['start-here/installation'],
    );
  });

  it('lists the included topics in order with their landing pages', async () => {
    const index = await build([]);
    assert.deepEqual(index.topics, [
      { l: 'Get Started', d: 'Get Started docs', s: 'start-here/installation' },
      { l: 'Guide', d: 'Guide docs', s: 'users-guide' },
      { l: 'Models', d: 'Models docs', s: 'users-guide/models' },
      { l: 'Troubleshooting', d: 'Troubleshooting docs', s: 'troubleshooting/faq' },
    ]);
  });

  it('keeps h2 and h3 headings in document order as [anchor, text]', async () => {
    const {
      pages: [page],
    } = await build([doc('start-here/installation', 'start-here/01.installation.mdx')], {
      'start-here/installation': [
        { depth: 1, slug: 'installation', text: 'Installation' },
        { depth: 2, slug: 'requirements', text: 'Requirements' },
        { depth: 4, slug: 'linux-notes', text: 'Linux notes' },
        { depth: 3, slug: 'gpu', text: 'GPU' },
        { depth: 2, slug: 'launcher', text: 'Launcher' },
      ],
    });
    assert.deepEqual(page.h, [
      ['requirements', 'Requirements'],
      ['gpu', 'GPU'],
      ['launcher', 'Launcher'],
    ]);
  });

  it('points each page at the most specific topic of its directory, and none outside every topic', async () => {
    const { topics: indexTopics, pages } = await build([
      doc('users-guide/models/loras', 'Users Guide/03.Models/02.loras.mdx'),
      doc('users-guide/canvas/layers', 'Users Guide/02.Canvas/layers.mdx'),
      doc('users-guide-extra', 'users-guide-extra/index.md'),
    ]);
    assert.deepEqual(
      Object.fromEntries(pages.map((page) => [page.s, page.c === undefined ? undefined : indexTopics[page.c].l])),
      {
        'users-guide-extra': undefined,
        'users-guide/canvas/layers': 'Guide',
        'users-guide/models/loras': 'Models',
      },
    );
  });

  it('orders pages by topic, then id, whatever the collection order, and omits empty optional fields', async () => {
    const docs = [
      doc('troubleshooting/faq', 'troubleshooting/faq.md', { title: 'FAQ' }),
      doc('users-guide/models/loras', 'Users Guide/03.Models/loras.md', { title: 'LoRAs' }),
      doc('start-here/upgrading', 'start-here/upgrading.md', { title: 'Upgrading', description: 'Move to a release.' }),
      doc('start-here/installation', 'start-here/installation.md', { title: 'Installation', description: '' }),
    ];
    const forward = await build(docs);
    const reversed = await build([...docs].reverse());
    assert.deepEqual(forward.pages, [
      { s: 'start-here/installation', t: 'Installation', c: 0 },
      { s: 'start-here/upgrading', t: 'Upgrading', d: 'Move to a release.', c: 0 },
      { s: 'users-guide/models/loras', t: 'LoRAs', c: 2 },
      { s: 'troubleshooting/faq', t: 'FAQ', c: 3 },
    ]);
    assert.equal(formatAppSearchIndex(reversed), formatAppSearchIndex(forward));
  });
});

describe('formatAppSearchIndex', () => {
  it('writes one topic or page per line as valid JSON', () => {
    const index = {
      topics: [{ l: 'Guide', d: 'How to', s: 'guide' }],
      pages: [
        { s: 'a', t: 'A' },
        { s: 'b', t: 'B', h: [['x', 'X']] },
      ],
    };
    const text = formatAppSearchIndex(index);
    assert.equal(
      text,
      '{"topics":[\n{"l":"Guide","d":"How to","s":"guide"}\n],\n"pages":[\n{"s":"a","t":"A"},\n{"s":"b","t":"B","h":[["x","X"]]}\n]}\n',
    );
    assert.deepEqual(JSON.parse(text), index);
  });
});
