// @ts-check
// The community node packs on the Community Nodes page (`components/CommunityNodes.astro`). Each pack is a YAML file in
// `src/content/community-nodes/`, whose README.md explains the fields. Plain JavaScript so `node --test` can cover it
// and `content.config.ts` can share the category ids.

/**
 * The categories a pack can list, in the order the filter shows them. A pack's first category picks its placeholder
 * icon when it has no images. `labelKey` is the category's UI string in `src/content/i18n/en.json`.
 */
export const NODE_PACK_CATEGORIES = /** @type {const} */ ([
  { id: 'image-effects', labelKey: 'communityNodes.category.imageEffects' },
  { id: 'masks', labelKey: 'communityNodes.category.masks' },
  { id: 'compositing', labelKey: 'communityNodes.category.compositing' },
  { id: 'prompts', labelKey: 'communityNodes.category.prompts' },
  { id: 'language-models', labelKey: 'communityNodes.category.languageModels' },
  { id: 'control', labelKey: 'communityNodes.category.control' },
  { id: 'video-3d', labelKey: 'communityNodes.category.video3d' },
  { id: 'sizing', labelKey: 'communityNodes.category.sizing' },
  { id: 'utility', labelKey: 'communityNodes.category.utility' },
]);

/** @typedef {(typeof NODE_PACK_CATEGORIES)[number]['id']} NodePackCategory */

/** The filter value that shows every category. */
export const ALL_CATEGORIES = 'all';

/**
 * Alphabetical by title, ignoring case and accents, then by id, so builds are stable.
 * @template {{ id: string, data: { title: string } }} T
 * @param {readonly T[]} packs
 * @returns {T[]}
 */
export const sortNodePacks = (packs) =>
  [...packs].sort(
    (a, b) => a.data.title.localeCompare(b.data.title, 'en', { sensitivity: 'base' }) || a.id.localeCompare(b.id)
  );

/**
 * The URL to paste into the Nodes manager or `git clone`: the repository page's URL with `.git`, as the manager's own
 * example shows. The manager names the pack's folder after the URL's last segment, without `.git`.
 * @param {string} repo the repository's web page, such as `https://github.com/owner/name`
 */
export const gitCloneUrl = (repo) => {
  const url = repo.replace(/\/+$/, '');
  return url.endsWith('.git') ? url : `${url}.git`;
};

/**
 * Splits text at Markdown code spans, so descriptions can name nodes and fields as `code` without allowing any other
 * markup. An unpaired backtick stays text.
 * @param {string} text
 * @returns {{ code: boolean, text: string }[]}
 */
export const codeSpans = (text) => {
  /** @type {{ code: boolean, text: string }[]} */
  const parts = [];
  let last = 0;
  for (const match of text.matchAll(/`([^`]+)`/g)) {
    if (match.index > last) parts.push({ code: false, text: text.slice(last, match.index) });
    parts.push({ code: true, text: match[1] });
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push({ code: false, text: text.slice(last) });
  return parts;
};

/**
 * A description's paragraphs: blocks separated by a blank line, each with its line breaks folded into spaces.
 * @param {string | undefined} text
 */
export const paragraphs = (text) =>
  (text ?? '')
    .split(/\n\s*\n/)
    .map((block) => block.replace(/\s+/g, ' ').trim())
    .filter(Boolean);

/**
 * Lowercase words without accents or punctuation, one space apart, for matching searches.
 * @param {string} text
 */
export const normalizeSearchText = (text) =>
  text
    .normalize('NFKD')
    .replace(/\p{M}/gu, '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim();

/**
 * What a search matches a pack against: its title, author, summary, description, node names and descriptions, and
 * the names of its categories in the page's language.
 * @param {{ title: string, author: string, summary: string, description?: string,
 *   nodes?: readonly { name: string, description?: string }[] }} pack
 * @param {readonly string[]} categoryLabels
 */
export const searchText = (pack, categoryLabels) =>
  normalizeSearchText(
    [
      pack.title,
      pack.author,
      pack.summary,
      pack.description ?? '',
      ...(pack.nodes ?? []).flatMap((node) => [node.name, node.description ?? '']),
      ...categoryLabels,
    ].join(' ')
  );

/**
 * Whether a pack shows for a search and a category. Every word of the query must appear somewhere in the pack's
 * search text, in any order and as part of a longer word ("mask" finds "masks"); an empty query matches every pack.
 * @param {{ text: string, categories: readonly string[] }} pack `text` from `searchText`
 * @param {{ query: string, category: string }} filter `category` is a category id or `ALL_CATEGORIES`
 */
export const packMatches = (pack, { query, category }) => {
  if (category !== ALL_CATEGORIES && !pack.categories.includes(category)) return false;
  const words = normalizeSearchText(query).split(' ').filter(Boolean);
  return words.every((word) => pack.text.includes(word));
};
