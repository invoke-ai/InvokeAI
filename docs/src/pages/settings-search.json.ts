// The compact settings index that the search modal fetches when it first opens (`src/lib/components/Search.astro`).
import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';

import { getPathLocale } from '../config/locales.mjs';
import settingsData from '../generated/settings.json';
import { buildSettingsSearchIndex } from '../lib/components/settings-reference.mjs';

export const GET: APIRoute = async () => {
  // Settings results link into whichever English page renders the settings reference; search adds the reader's locale.
  const docs = await getCollection('docs', ({ id }) => !getPathLocale(id));
  const reference = docs.find((doc) => doc.body?.includes('<SettingsDocs'));
  if (!reference) throw new Error('No docs page renders <SettingsDocs />, so settings search results have no target.');

  return new Response(JSON.stringify(buildSettingsSearchIndex(settingsData.settings, `/${reference.id}/`)), {
    headers: { 'Content-Type': 'application/json' },
  });
};
