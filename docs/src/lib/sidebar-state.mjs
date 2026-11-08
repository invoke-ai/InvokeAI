// Keeps the sidebar's sections open or closed as the reader left them, across pages of the session. Starlight's own
// persistence keeps one state for one sidebar and only restores it on wide screens, so every topic switch, and every
// page on a phone, reset the sections; this keys the state by topic and section directory instead, which every locale
// shares. Plain JavaScript so `node --test` can exercise it; `Sidebar.astro` runs `mountSidebarState` as an inline
// script.

/** @param {string} topicId */
export const sidebarStateKey = (topicId) => `docs-sidebar:${topicId}`;

/**
 * Whether a section opens on arrival: the section holding the current page always does; otherwise the reader's last
 * choice wins over the sidebar's default.
 * @param {{ rendered: boolean, containsCurrent: boolean, stored: unknown }} section
 */
export const resolveOpen = ({ rendered, containsCurrent, stored }) =>
  containsCurrent || (typeof stored === 'boolean' ? stored : rendered);

/**
 * The scroll position to remember, or `undefined` while the sidebar is hidden (the closed mobile drawer), whose
 * position of 0 would overwrite the one kept from a wider window.
 * @param {{ clientHeight: number, scrollTop: number } | null} scroller
 */
export const visibleScrollTop = (scroller) =>
  scroller && scroller.clientHeight > 0 ? scroller.scrollTop : undefined;

/**
 * Restores the sections and scroll position of the sidebar tree, then records the reader's changes. Runs inline right
 * after the tree, before it paints, so sections never animate into place. Serialized with `toString()`: it must only
 * use its arguments and globals. Storage may be unavailable; the sidebar then keeps its defaults.
 * @param {Element} tree
 * @param {string} key
 * @param {typeof resolveOpen} resolve
 * @param {typeof visibleScrollTop} scrollTopToKeep
 */
export const mountSidebarState = (tree, key, resolve, scrollTopToKeep) => {
  const scroller = document.getElementById('starlight__sidebar');
  /** A section's directory (`SidebarTree.astro`), the same in every locale; state kept under other keys is ignored. */
  const sectionOf = (/** @type {HTMLElement} */ details) => details.dataset.section ?? '';

  /** @type {{ open: Record<string, boolean>, scroll?: number }} */
  let state = { open: {} };
  try {
    const stored = JSON.parse(sessionStorage.getItem(key) || '{}');
    if (stored && typeof stored.open === 'object' && stored.open) state = stored;
  } catch {}
  const save = () => {
    try {
      sessionStorage.setItem(key, JSON.stringify(state));
    } catch {}
  };

  for (const details of tree.querySelectorAll('details')) {
    details.open = resolve({
      rendered: details.open,
      containsCurrent: details.querySelector('[aria-current="page"]') !== null,
      stored: state.open[sectionOf(details)],
    });
  }
  if (scroller && typeof state.scroll === 'number') scroller.scrollTop = state.scroll;

  // A click on a summary toggles its section after the event, so record the state it is about to take.
  tree.addEventListener('click', (event) => {
    const details = event.target instanceof Element ? event.target.closest('summary')?.parentElement : null;
    if (!(details instanceof HTMLDetailsElement)) return;
    state.open[sectionOf(details)] = !details.open;
    save();
  });
  addEventListener('pagehide', () => {
    const scroll = scrollTopToKeep(scroller);
    if (scroll === undefined) return;
    state.scroll = scroll;
    save();
  });
};
