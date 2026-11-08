import type { TFunction } from 'i18next';

import { DOCS_URL } from '@platform/runtime/appMetadata';

import type { PaletteEntry, PaletteSearchProvider } from './entries';

import { getPaletteContributionKey } from './contributionKey';

/**
 * Docs search for both palettes. A leaf on purpose: the Launchpad palette imports it, so it must not reach editor
 * modules, and the index loads as its own chunk on the first search.
 */

/**
 * Mirrors `AppSearchIndex` in `docs/src/lib/app-search.mjs`. `docsSearchIndex.json` is generated from the docs build
 * (`cd docs && pnpm build && pnpm run sync-app-search-index`); docs CI fails when it drifts.
 */
interface DocsSearchIndex {
  /** User-facing topics in the docs' order. */
  topics: DocsSearchTopic[];
  pages: DocsSearchPage[];
}

interface DocsSearchTopic {
  /** Label. */
  l: string;
  /** Description. */
  d: string;
  /** Landing page id. */
  s: string;
}

interface DocsSearchPage {
  /** Page id: the URL path relative to the docs root, without slashes. */
  s: string;
  /** Page title. */
  t: string;
  /** Page description. */
  d?: string;
  /** Index into `topics`. */
  c?: number;
  /** h2/h3 headings in document order. */
  h?: [anchor: string, text: string][];
}

interface DocsSearchRow {
  anchor?: string;
  pageId: string;
  pageTitle: string;
  topic?: string;
  title: string;
  titleText: string;
  titleWords: string[];
  /** Title, page title and topic: what every row can match on. */
  contextText: string;
  descriptionText: string;
}

interface LoadedDocsIndex {
  rows: DocsSearchRow[];
  topics: DocsSearchTopic[];
}

const DOCS_RESULT_LIMIT = 20;

const toRows = ({ pages, topics }: DocsSearchIndex): DocsSearchRow[] =>
  pages.flatMap((page) => {
    const topic = page.c === undefined ? undefined : topics[page.c]?.l;
    const row = (title: string, anchor?: string, description = ''): DocsSearchRow => {
      const titleText = title.toLowerCase();

      return {
        anchor,
        contextText: [titleText, page.t.toLowerCase(), topic?.toLowerCase() ?? ''].join('\n'),
        descriptionText: description.toLowerCase(),
        pageId: page.s,
        pageTitle: page.t,
        title,
        titleText,
        titleWords: titleText.split(/[^\p{L}\p{N}]+/u).filter(Boolean),
        topic,
      };
    };

    return [row(page.t, undefined, page.d), ...(page.h ?? []).map(([anchor, text]) => row(text, anchor))];
  });

let indexPromise: Promise<LoadedDocsIndex> | undefined;

// A failed chunk load is not cached, so the palette's Retry can load it again.
const loadIndex = (): Promise<LoadedDocsIndex> =>
  (indexPromise ??= import('./docsSearchIndex.json').then(
    ({ default: json }) => {
      // The JSON infers loose tuple types; docs CI guarantees the shape.
      const index = json as unknown as DocsSearchIndex;

      return { rows: toRows(index), topics: index.topics };
    },
    (error: unknown) => {
      indexPromise = undefined;
      throw error;
    }
  ));

/**
 * Lower is better; undefined is no match. Every term must match. A page whose title starts with the query or has every
 * term at a word start leads, as pages are the main destinations; then headings matched the same way; then any title
 * containing every term; then rows that need their page title or topic; last, pages matched only through their
 * description.
 */
const rankRow = (row: DocsSearchRow, phrase: string, terms: readonly string[]): number | undefined => {
  const headingOffset = row.anchor === undefined ? 0 : 2;

  if (row.titleText.startsWith(phrase)) {
    return headingOffset;
  }
  if (terms.every((term) => row.titleWords.some((word) => word.startsWith(term)))) {
    return headingOffset + 1;
  }
  if (terms.every((term) => row.titleText.includes(term))) {
    return 4;
  }
  if (terms.every((term) => row.contextText.includes(term))) {
    return 5;
  }
  if (terms.every((term) => row.contextText.includes(term) || row.descriptionText.includes(term))) {
    return 6;
  }

  return undefined;
};

const searchDocsRows = (rows: readonly DocsSearchRow[], terms: readonly string[]): DocsSearchRow[] => {
  const phrase = terms.join(' ');

  return rows
    .flatMap((row, order) => {
      const rank = rankRow(row, phrase, terms);

      return rank === undefined ? [] : [{ order, rank, row }];
    })
    .sort(
      (left, right) =>
        left.rank - right.rank ||
        // Within a tier, pages lead their headings, and a shorter title is a closer match.
        Number(left.row.anchor !== undefined) - Number(right.row.anchor !== undefined) ||
        left.row.title.length - right.row.title.length ||
        left.order - right.order
    )
    .slice(0, DOCS_RESULT_LIMIT)
    .map(({ row }) => row);
};

export const createDocsProvider = ({ t }: { t: TFunction }): PaletteSearchProvider => {
  const toEntry = ({
    anchor,
    pageId,
    subtitle,
    title,
  }: {
    anchor?: string;
    pageId: string;
    subtitle?: string;
    title: string;
  }): PaletteEntry => {
    const url = new URL(`${pageId ? `${pageId}/` : ''}${anchor ? `#${anchor}` : ''}`, DOCS_URL).href;

    return {
      group: 'Docs',
      groupLabel: t('commandPalette.groups.docs'),
      id: `docs:${url}`,
      isPersistentRecent: false,
      opensExternally: true,
      run: () => window.open(url, '_blank', 'noopener,noreferrer'),
      subtitle,
      title,
    };
  };

  return {
    contextKey: 'global',
    label: t('commandPalette.providers.docs'),
    providerKey: getPaletteContributionKey('provider', 'docs'),
    search: async (query, { signal }) => {
      const { rows, topics } = await loadIndex();

      signal.throwIfAborted();

      const terms = query.text.trim().toLowerCase().split(/\s+/).filter(Boolean);

      // An empty query only reaches the provider in its own scope; browse the topics there instead of an empty list.
      if (terms.length === 0) {
        return topics.map((topic) => toEntry({ pageId: topic.s, subtitle: topic.d, title: topic.l }));
      }

      return searchDocsRows(rows, terms).map((row) =>
        toEntry({
          anchor: row.anchor,
          pageId: row.pageId,
          subtitle: row.anchor ? row.pageTitle : row.topic,
          title: row.title,
        })
      );
    },
  };
};
