// The announcement pill above the home page's title. It fills the image slot of Starlight's hero, which takes raw
// HTML, so the route middleware (`src/route-data.ts`) builds it here with its text from the UI strings rather than the
// page's frontmatter, where translators would see markup. `HomePage.astro` styles it.

/** @param {string} text */
const escapeHtml = (text) =>
  text.replace(/[&<>"']/g, (character) => `&#${character.charCodeAt(0)};`);

/** Lucide's `arrow-right`. */
const ARROW =
  '<svg class="home-announcement__arrow" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>';

/**
 * The pill's markup, with `text` and `href` escaped.
 * @param {{ text: string, href: string }} announcement
 */
export const announcementHtml = ({ text, href }) =>
  `<a class="home-announcement" href="${escapeHtml(href)}" aria-label="${escapeHtml(text)}"><span>` +
  `<span class="home-announcement__dot" aria-hidden="true"></span>${escapeHtml(text)}</span>${ARROW}</a>`;
