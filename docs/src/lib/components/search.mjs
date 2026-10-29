// The search modal's data handling (`Search.astro`): matching the settings index, shaping Pagefind results into rows,
// and the recent searches list. Plain JavaScript so `node --test` can cover it.

/** Settings results show in the All docs scope, and in this topic's scope, which documents the settings. */
export const SETTINGS_TOPIC_ID = 'configure';
export const RECENT_STORAGE_KEY = 'invoke-docs:recent-searches';
export const RECENT_LIMIT = 5;

const ENV_PREFIX = 'invokeai_';
// A description match on a shorter query mostly finds noise.
const MIN_DESCRIPTION_QUERY_LENGTH = 3;

/**
 * @typedef {{ id: string, title: string }} SearchTopic
 * @typedef {'topic' | 'all'} SearchScope
 */

/**
 * Everything that follows from the search scope, so the chips, the Pagefind filter and the settings results never
 * disagree. Without a scope, the current topic's is the default; without a topic, only All docs exists.
 * @param {SearchTopic | undefined} topic the page's topic, from its `docs-topic` meta tag
 * @param {SearchScope} [requested]
 */
export const resolveScope = (topic, requested = topic ? 'topic' : 'all') => {
  const scoped = requested === 'topic' && topic ? topic : undefined;
  /** @type {SearchScope} */
  const scope = scoped ? 'topic' : 'all';
  return {
    scope,
    /** The topic results are limited to, if any. */
    scoped,
    filters: scoped ? { topic: scoped.id } : undefined,
    showSettings: !scoped || scoped.id === SETTINGS_TOPIC_ID,
    chips: [
      { scope: 'topic', hidden: !topic, pressed: scope === 'topic' },
      { scope: 'all', hidden: false, pressed: scope === 'all' },
    ],
  };
};

/**
 * Lowercases text and treats runs of spaces, underscores and hyphens as one `_`, so `max cache`, `max-cache` and
 * `max_cache` read the same.
 * @param {string} text
 */
export const normalizeTerm = (text) =>
  text
    .toLowerCase()
    .trim()
    .replace(/[\s_-]+/g, '_');

/**
 * @typedef {import('./settings-reference.mjs').SettingsSearchEntry} SettingsSearchEntry
 */

/**
 * The settings that match a query, best first: an exact key, then keys starting with the query, keys containing it,
 * keys containing every word of it, and last keys and descriptions with words starting with every word of it.
 * `INVOKEAI_PORT` finds `port`.
 * @param {SettingsSearchEntry[]} settings
 * @param {string} query
 * @returns {SettingsSearchEntry[]}
 */
export const matchSettings = (settings, query) => {
  let term = normalizeTerm(query).replace(/^_+|_+$/g, '');
  if (term.startsWith(ENV_PREFIX) && term.length > ENV_PREFIX.length) term = term.slice(ENV_PREFIX.length);
  if (term.length < 2) return [];
  const words = term.split('_').filter(Boolean);

  /** @param {SettingsSearchEntry} setting */
  const rank = (setting) => {
    const key = normalizeTerm(setting.key);
    if (key === term) return 0;
    if (key.startsWith(term)) return 1;
    if (key.includes(term)) return 2;
    if (words.length > 1 && words.every((word) => key.includes(word))) return 3;
    if (term.length < MIN_DESCRIPTION_QUERY_LENGTH) return undefined;
    // Words match at word starts, so `ram` does not find `vram`.
    const textWords = normalizeTerm(`${setting.key} ${setting.description}`).split(/[^a-z0-9]+/);
    return words.every((word) => textWords.some((textWord) => textWord.startsWith(word))) ? 4 : undefined;
  };

  return settings
    .map((setting, index) => ({ setting, index, rank: rank(setting) }))
    .filter((match) => match.rank !== undefined)
    .sort((a, b) => a.rank - b.rank || a.setting.key.length - b.setting.key.length || a.index - b.index)
    .map((match) => match.setting);
};

/**
 * @typedef {{ title: string, url: string, excerpt: string, anchor?: unknown, locations?: number[] }} PagefindSubResult
 * @typedef {{
 *   url: string,
 *   excerpt: string,
 *   meta?: { title?: string },
 *   filters?: Record<string, string[]>,
 *   sub_results?: PagefindSubResult[],
 * }} PagefindResultData
 * @typedef {{ url: string, page: string, section?: string, excerpt: string, topic?: string }} PageRow `topic` is the
 *   page's topic id, from its Pagefind `topic` filter
 */

/**
 * One row per matching section of a Pagefind result, for at most `maxSections` sections: those with the most matches,
 * in page order. A match outside every heading's section links to the page itself.
 * @param {PagefindResultData} data
 * @param {number} maxSections
 * @returns {PageRow[]}
 */
export const toPageRows = (data, maxSections) => {
  const page = data.meta?.title || data.url;
  const topic = data.filters?.topic?.[0];
  const sections = data.sub_results ?? [];
  if (sections.length === 0) return [{ url: data.url, page, excerpt: data.excerpt, topic }];

  const best = new Set(
    [...sections].sort((a, b) => (b.locations?.length ?? 0) - (a.locations?.length ?? 0)).slice(0, maxSections),
  );
  return sections
    .filter((section) => best.has(section))
    .map((section) => ({
      url: section.url,
      page,
      section: section.anchor && section.title !== page ? section.title : undefined,
      excerpt: section.excerpt,
      topic,
    }));
};

/**
 * Groups rows under their topic's title, keeping the rank order: groups are ordered by their best row.
 * @param {PageRow[]} rows
 * @param {Map<string, string>} titles topic id -> title
 * @param {string} untitled the group label for pages outside every topic
 * @returns {{ label: string, rows: PageRow[] }[]}
 */
export const groupByTopic = (rows, titles, untitled) => {
  /** @type {Map<string, PageRow[]>} */
  const groups = new Map();
  for (const row of rows) {
    const label = (row.topic && titles.get(row.topic)) || untitled;
    const group = groups.get(label);
    if (group) group.push(row);
    else groups.set(label, [row]);
  }
  return [...groups].map(([label, groupRows]) => ({ label, rows: groupRows }));
};

/**
 * @typedef {{ title: string, section?: string, url: string }} RecentEntry
 */

/**
 * The recent searches list from its stored JSON, dropping anything malformed.
 * @param {string | null} raw
 * @returns {RecentEntry[]}
 */
export const parseRecent = (raw) => {
  let value;
  try {
    value = JSON.parse(raw ?? '[]');
  } catch {
    return [];
  }
  if (!Array.isArray(value)) return [];
  return value
    .filter(
      (entry) =>
        typeof entry?.title === 'string' &&
        typeof entry.url === 'string' &&
        (entry.section === undefined || typeof entry.section === 'string'),
    )
    .map(({ title, section, url }) => (section === undefined ? { title, url } : { title, section, url }))
    .slice(0, RECENT_LIMIT);
};

/**
 * Puts an opened result first, without duplicates.
 * @param {RecentEntry[]} list
 * @param {RecentEntry} entry
 * @returns {RecentEntry[]}
 */
export const addRecent = (list, entry) =>
  [entry, ...list.filter((existing) => existing.url !== entry.url)].slice(0, RECENT_LIMIT);
