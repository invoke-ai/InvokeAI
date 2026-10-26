// The compact settings index that the search modal fetches when it first opens (`src/lib/components/Search.astro`).
import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';

import settingsData from '../generated/settings.json';
import { withBase } from '../lib/base-path';
import { buildSettingsSearchIndex } from '../lib/components/settings-reference.mjs';

export const GET: APIRoute = async () => {
  // Settings results link into whichever page renders the settings reference.
  const docs = await getCollection('docs');
  const reference = docs.find((doc) => doc.body?.includes('<SettingsDocs'));
  if (!reference) throw new Error('No docs page renders <SettingsDocs />, so settings search results have no target.');

  const page = withBase(`/${reference.id}/`, import.meta.env.BASE_URL);
  return new Response(JSON.stringify(buildSettingsSearchIndex(settingsData.settings, page)), {
    headers: { 'Content-Type': 'application/json' },
  });
};
