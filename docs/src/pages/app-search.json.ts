// The docs index the app's command palette searches. The app commits a copy, so resync it with
// `pnpm build && pnpm run sync-app-search-index` after adding, removing, moving or renaming a page, changing a page's
// title, description, `draft` or `pagefind` frontmatter, any h2 or h3 heading, or a topic in `src/config/topics.ts`.
import type { APIRoute } from 'astro';
import { getCollection, render } from 'astro:content';

import { getPathLocale } from '../config/locales.mjs';
import { docsTopics, topicDescription, topicLabel } from '../config/topics';
import { buildAppSearchIndex } from '../lib/app-search.mjs';

export const GET: APIRoute = async ({ locals: { t } }) => {
  // Headings come from Astro's own render, so their anchors are the ids the pages carry.
  // The app links to the English pages.
  const index = await buildAppSearchIndex(await getCollection('docs', ({ id }) => !getPathLocale(id)), {
    // In English, the root locale this route renders in.
    topics: docsTopics.map((topic) => ({
      ...topic,
      label: topicLabel(t, topic),
      description: topicDescription(t, topic),
    })),
    getHeadings: async (doc) => (await render(doc)).headings,
  });
  return new Response(JSON.stringify(index), { headers: { 'Content-Type': 'application/json' } });
};
