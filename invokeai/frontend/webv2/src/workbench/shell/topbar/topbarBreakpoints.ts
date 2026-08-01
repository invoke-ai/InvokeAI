/**
 * The top bar's own widths, as raw media queries.
 *
 * The theme's breakpoint scale (992 / 1280 / 1536) is about page layout; this
 * bar degrades at the points where its *own* content stops fitting, which are
 * not the same numbers. Rather than bend the shared scale for one component, the
 * three thresholds live here as conditions.
 *
 * Order of sacrifice: the ⌘↵ hint, then the preset labels, then the project
 * name. The routing indicator and the queue readout never collapse at any width
 * — the first is the only thing telling a user what ⌘↵ is about to run, and the
 * second is the only thing telling them whether it started.
 */
const hideBelow = (px: number) => ({ [`@media (max-width: ${px - 1}px)`]: { display: 'none' } }) as const;

/** The `⌘↵` hint on the Invoke button face; the tooltip keeps it below this. */
export const HIDE_BELOW_HINT_WIDTH = hideBelow(1440);

/** Preset labels; the names move to their tooltips below this. */
export const HIDE_BELOW_PRESET_LABEL_WIDTH = hideBelow(1280);

/** The project name; the count badge and selector glyph always stay. */
export const HIDE_BELOW_PROJECT_NAME_WIDTH = hideBelow(1024);
