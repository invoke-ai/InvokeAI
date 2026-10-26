// Shared by the `invokeai.yaml` settings reference (`SettingsDocs.astro`) and the compact settings index that the
// search modal loads (`src/pages/settings-search.json.ts`), so both show a setting the same way and agree on its
// anchor.

const DESCRIPTION_MAX_LENGTH = 160;

/**
 * The id of a setting's entry in the settings reference.
 * @param {string} name
 */
export const settingAnchorId = (name) => `setting-${name}`;

/**
 * A setting's default value as shown to readers.
 * @param {unknown} value
 */
export const formatSettingValue = (value) => {
  if (value === null) return 'null';
  if (Array.isArray(value) || typeof value === 'object') return JSON.stringify(value);
  return String(value);
};

/**
 * Cleans up a Python type representation: `<class 'pathlib.Path'>` -> `Path`,
 * `typing.Optional[pathlib.Path]` -> `Optional[Path]`.
 * @param {string} typeStr
 */
export const formatSettingType = (typeStr) => {
  const classMatch = typeStr.match(/^<class '(.+)'>$/);
  if (classMatch) return classMatch[1].split('.').pop();
  return typeStr.replace(/typing\./g, '').replace(/[a-z_][a-z0-9_.]*\.([A-Z][A-Za-z]*)/g, '$1');
};

/**
 * The first sentence of a description, cut at a word boundary when it is still long.
 * @param {string} text
 */
export const summarizeDescription = (text) => {
  const trimmed = text.trim();
  // A sentence ends at `.`, `!` or `?` followed by the end or by a space and what starts the next sentence, so
  // "e.g. `/invoke`" and version numbers stay intact.
  const sentence = trimmed.match(/^.*?(?<!\b(?:e\.g|i\.e))[.!?](?=$|\s+[A-Z('"])/s)?.[0] ?? trimmed;
  if (sentence.length <= DESCRIPTION_MAX_LENGTH) return sentence;
  const cut = sentence.slice(0, DESCRIPTION_MAX_LENGTH - 1);
  return `${cut.slice(0, Math.max(cut.lastIndexOf(' '), 1)).replace(/(?:[\s,;:]|\b(?:e\.g|i\.e)\.)+$/, '')}…`;
};

/**
 * @typedef {{ name: string, type: string, default: unknown, description: string, env_var?: string | null }} Setting
 * @typedef {{ key: string, type: string, default: string, description: string, env?: string }} SettingsSearchEntry
 */

/**
 * The compact settings index the search modal fetches: the page that documents the settings, and what each search
 * result row shows.
 * @param {Setting[]} settings
 * @param {string} page the settings reference page's URL, including the deployment base path
 * @returns {{ page: string, settings: SettingsSearchEntry[] }}
 */
export const buildSettingsSearchIndex = (settings, page) => ({
  page,
  settings: settings.map((setting) => ({
    key: setting.name,
    type: formatSettingType(setting.type),
    default: formatSettingValue(setting.default),
    description: summarizeDescription(setting.description ?? ''),
    ...(setting.env_var ? { env: setting.env_var } : {}),
  })),
});
