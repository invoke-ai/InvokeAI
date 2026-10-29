import { localizePath, splitSitePath } from '../config/locales.mjs';

// Keeps readers in their language: Markdown links are root-relative English paths, and a page without a translation
// renders the English page's HTML, so the links are moved into the page's locale after rendering
// (`components/MarkdownContent.astro`). Plain JavaScript so `node --test` can cover it.

/**
 * Comments, script and style elements (whose text may hold markup-like strings), and start tags, whose quoted
 * attribute values may hold `>` and `<`, as Expressive Code's `data-code` does. Consuming every start tag keeps an
 * `<a` inside another element's attribute value from reading as a tag.
 */
const TAG_BODY = `(?:[^>"']|"[^"]*"|'[^']*')*`;
const tokenRegEx = new RegExp(
  [
    '<!--[\\s\\S]*?-->',
    `<(script|style)\\b${TAG_BODY}>[\\s\\S]*?</\\1\\s*>`,
    `<([a-z][^\\s/>]*)(${TAG_BODY})>`,
  ].join('|'),
  'gi'
);
/** One attribute of a start tag; the double-quoted value, as Astro writes `href`, is the second group. */
const attributeRegEx = /([^\s"'>/=]+)(?:\s*=\s*(?:"([^"]*)"|'[^']*'|[^\s"'=<>`]+))?/g;

/**
 * Moves the root-relative `href` values of anchors in rendered HTML into `locale`. Hrefs already carry the base path
 * (`plugins/rehype-prefix-base-to-root-links.mjs`); links outside it or already in a locale, and relative,
 * protocol-relative and external links, stay as they are.
 * @param {string} html
 * @param {{ base: string, locale: string }} options `base` is Astro's `BASE_URL`, such as `/` or `/InvokeAI`
 */
export const localizeHtmlLinks = (html, { base, locale }) => {
  const basePath = base.replace(/\/+$/, '');
  /** @param {string} href */
  const localizeHref = (href) => {
    if (!href.startsWith('/') || href.startsWith('//')) return href;
    const site = splitSitePath(href, base);
    return site && !site.locale ? `${basePath}${localizePath(site.path, locale)}` : href;
  };
  return html.replace(tokenRegEx, (token, _rawText, tagName, attributes) =>
    tagName?.toLowerCase() === 'a'
      ? `<${tagName}${attributes.replace(attributeRegEx, (attribute, name, value) =>
          name.toLowerCase() === 'href' && value !== undefined ? `${name}="${localizeHref(value)}"` : attribute
        )}>`
      : token
  );
};
