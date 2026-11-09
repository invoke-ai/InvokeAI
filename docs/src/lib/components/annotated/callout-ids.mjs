// @ts-check

// Ids of an `Annotated` figure's legend entries, which their pins link to. They depend only on the page's own
// content (`callout-<figure>-<entry>`, both counted from 1 in page order), so a deep link to an entry keeps working
// across builds and dev reloads. A callout cannot know its figure or its place in it, so it writes a unique marker;
// the figure numbers the markers in its rendered legend.

const MARKER = /annotated-callout-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/g;

/** A placeholder for one callout's id: write it everywhere the id goes (the entry's `id`, the pin's `href`). */
export const calloutMarker = () => `annotated-callout-${crypto.randomUUID()}`;

/** @type {WeakMap<object, number>} */
const figuresOnPage = new WeakMap();

/**
 * The figure's number on its page, from 1. `Astro.locals` is created for each page render, so it keys the count.
 * @param {object} locals the page's `Astro.locals`
 */
export const nextFigureIndex = (locals) => {
  const index = (figuresOnPage.get(locals) ?? 0) + 1;
  figuresOnPage.set(locals, index);
  return index;
};

/**
 * Replaces each callout marker in a figure's rendered legend with its id: the n-th distinct marker becomes
 * `callout-<figure>-<n>`, and repeats of a marker get the same id.
 * @param {string} html
 * @param {number} figure
 */
export const numberCallouts = (html, figure) => {
  /** @type {Map<string, string>} */
  const ids = new Map();
  return html.replace(MARKER, (marker) => {
    if (!ids.has(marker)) ids.set(marker, `callout-${figure}-${ids.size + 1}`);
    return /** @type {string} */ (ids.get(marker));
  });
};
