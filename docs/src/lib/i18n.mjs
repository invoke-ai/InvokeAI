// UI string helpers beyond `Astro.locals.t`: messages with markup tags (`components/Trans.astro`) and messages that
// client scripts finish themselves. Plain JavaScript so `node --test` can cover it.

const PLURAL_CATEGORIES = ['zero', 'one', 'two', 'few', 'many', 'other'];

/** The CLDR plural-category suffix of an i18next plural form's key, as in `search.resultCount_one`. */
export const PLURAL_SUFFIX = new RegExp(`_(${PLURAL_CATEGORIES.join('|')})$`);

/** An i18next `{{name}}` placeholder; the first group is the name. Global, so use it with `matchAll` or `replace`. */
export const PLACEHOLDER = /\{\{\s*(\w+)\s*\}\}/g;

/**
 * @typedef {{ text: string } | { tag: string, text: string }} MessagePart
 * @typedef {{ (key: string): string, exists: (key: string) => boolean }} Translate
 */

/**
 * Splits a message into plain text and `<name>text</name>` parts, react-i18next style, so translators can place markup
 * such as a link anywhere in the sentence. Tags hold plain text; unclosed and mismatched tags stay literal text.
 * @param {string} message
 * @returns {MessagePart[]}
 */
export const parseTags = (message) => {
  /** @type {MessagePart[]} */
  const parts = [];
  let last = 0;
  for (const match of message.matchAll(/<([A-Za-z][\w-]*)>([^<]*)<\/\1>/g)) {
    const [whole, tag, text] = match;
    if (match.index > last) parts.push({ text: message.slice(last, match.index) });
    parts.push({ tag, text });
    last = match.index + whole.length;
  }
  if (last < message.length) parts.push({ text: message.slice(last) });
  return parts;
};

/**
 * The plural forms of `key` (`key_one`, `key_other`, …) for a client script to pick from with `formatPlural`, keyed
 * by plural category. Forms the locale lacks come from English, as with `t`.
 * @param {Translate} t
 * @param {string} key
 * @returns {Record<string, string>}
 */
export const pluralForms = (t, key) =>
  Object.fromEntries(
    PLURAL_CATEGORIES.filter((category) => t.exists(`${key}_${category}`)).map((category) => [
      category,
      t(`${key}_${category}`),
    ])
  );

/**
 * Fills i18next-style `{{name}}` placeholders. Placeholders without a value stay as they are.
 * @param {string} template
 * @param {Record<string, string | number>} values
 */
export const interpolate = (template, values) =>
  template.replace(PLACEHOLDER, (placeholder, name) =>
    Object.hasOwn(values, name) ? String(values[name]) : placeholder
  );

/**
 * The form of `forms` (from `pluralForms`) that `lang` uses for `count`, with `{{count}}` filled in. Falls back to
 * `other`, the one form every language has.
 * @param {Record<string, string>} forms
 * @param {number} count
 * @param {string} lang
 */
export const formatPlural = (forms, count, lang) =>
  interpolate(forms[new Intl.PluralRules(lang).select(count)] ?? forms.other ?? '', { count });
