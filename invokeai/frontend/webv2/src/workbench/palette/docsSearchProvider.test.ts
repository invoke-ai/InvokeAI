import type { TFunction } from 'i18next';

import { afterEach, describe, expect, it, vi } from 'vitest';

const index = vi.hoisted(() => ({
  value: {
    pages: [
      {
        c: 0,
        h: [
          ['docker', 'Docker'],
          ['launcher', 'Launcher'],
        ],
        s: 'start-here/installation',
        t: 'Installation',
      },
      {
        c: 2,
        d: 'Families, LoRAs and installs.',
        h: [['starter-models', 'Starter models']],
        s: 'users-guide/models/introduction',
        t: 'Models',
      },
      {
        c: 2,
        h: [
          ['weights', 'Weights'],
          ['training-loras', 'Training LoRAs'],
        ],
        s: 'users-guide/models/loras',
        t: 'LoRAs',
      },
      {
        c: 1,
        h: [
          ['lora', 'LoRA'],
          ['setup', 'Setup'],
        ],
        s: 'users-guide/canvas/control-layers',
        t: 'Control layers',
      },
      { s: 'troubleshooting/simple-setup', t: 'Simple Setup' },
      {
        h: Array.from({ length: 30 }, (_, index) => [`step-${index}`, `Step ${index}`]),
        s: 'troubleshooting/many-steps',
        t: 'Many steps',
      },
    ],
    topics: [
      { d: 'Install, requirements', l: 'Get Started', s: 'start-here/installation' },
      { d: 'Workbench, canvas', l: 'Guide', s: 'users-guide/workbench/introduction' },
      { d: 'Families, installs', l: 'Models', s: 'users-guide/models/introduction' },
    ],
  },
}));

vi.mock('./docsSearchIndex.json', () => ({ default: index.value }));

import { createDocsProvider } from './docsSearchProvider';

const t = ((key: string) => key) as TFunction;
const search = (text: string) => createDocsProvider({ t }).search({ text }, { signal: new AbortController().signal });
const titles = async (text: string) => (await search(text)).map((entry) => entry.title);

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('docs palette provider', () => {
  it('ranks title matches by closeness, pages before headings, and description-only pages last', async () => {
    expect(await titles('lora')).toEqual([
      // Page title prefix.
      'LoRAs',
      // Heading prefix, then a heading with every term at a word start.
      'LoRA',
      'Training LoRAs',
      // Matched only through the page title.
      'Weights',
      // Matched only through the description.
      'Models',
    ]);
  });

  it('puts a page whose title has the query as a word above a heading that starts with it', async () => {
    expect(await titles('setup')).toEqual(['Simple Setup', 'Setup']);
  });

  it('requires every term, letting a heading combine its text with its page title', async () => {
    expect(await titles('install docker')).toEqual(['Docker']);
    expect(await titles('install launcher')).toEqual(['Launcher']);
    expect(await titles('install kubernetes')).toEqual([]);
  });

  it('matches a page by its topic label', async () => {
    expect(await titles('get started')).toEqual(['Installation', 'Docker', 'Launcher']);
  });

  it('browses the topics in order when its own scope opens with an empty query', async () => {
    const entries = await search('  ');

    expect(entries.map(({ subtitle, title }) => ({ subtitle, title }))).toEqual([
      { subtitle: 'Install, requirements', title: 'Get Started' },
      { subtitle: 'Workbench, canvas', title: 'Guide' },
      { subtitle: 'Families, installs', title: 'Models' },
    ]);
    expect(entries.every((entry) => entry.opensExternally)).toBe(true);
  });

  it('caps the results', async () => {
    expect(await search('step')).toHaveLength(20);
  });

  it('opens the page or heading anchor on the docs site in a new tab without an opener', async () => {
    const open = vi.fn(() => null);
    vi.stubGlobal('window', { open });
    const [page, heading] = await search('lora');

    expect(page).toMatchObject({ group: 'Docs', opensExternally: true, subtitle: 'Models', title: 'LoRAs' });
    expect(heading).toMatchObject({ opensExternally: true, subtitle: 'Control layers', title: 'LoRA' });

    page!.run();
    heading!.run();

    expect(open.mock.calls).toEqual([
      ['https://v7.invoke.ai/users-guide/models/loras/', '_blank', 'noopener,noreferrer'],
      ['https://v7.invoke.ai/users-guide/canvas/control-layers/#lora', '_blank', 'noopener,noreferrer'],
    ]);
  });
});

describe('docs index loading', () => {
  it('loads the index again after a failed chunk load', async () => {
    let failLoads = 1;
    vi.resetModules();
    // A factory call stands for the index chunk loading; the first one fails like a dropped network.
    vi.doMock('./docsSearchIndex.json', () => {
      if (failLoads > 0) {
        failLoads -= 1;
        throw new Error('Failed to fetch dynamically imported module');
      }

      return { default: index.value };
    });
    const { createDocsProvider: createFreshProvider } = await import('./docsSearchProvider');
    const provider = createFreshProvider({ t });
    const context = { signal: new AbortController().signal };

    await expect(provider.search({ text: 'docker' }, context)).rejects.toThrow();
    expect((await provider.search({ text: 'docker' }, context)).map((entry) => entry.title)).toEqual(['Docker']);
  });
});
