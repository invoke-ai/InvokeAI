/**
 * Registers the app-shell service worker (`sw.js`, emitted by
 * `scripts/service-worker-plugin.mjs`). Hashed assets become cache-first so
 * repeat boots skip the network; navigations stay network-first so a plain
 * reload always picks up a new deploy; the shell keeps working offline.
 *
 * Production only: the dev server never emits `sw.js`, and a worker caching
 * unbundled dev modules would only get in the way. Registration waits for
 * `load` so it cannot compete with boot-critical fetches, and
 * `updateViaCache: 'none'` makes every navigation check for a new build.
 */
export const registerServiceWorker = (): void => {
  if (!import.meta.env.PROD || typeof navigator === 'undefined' || !('serviceWorker' in navigator)) {
    return;
  }

  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js', { updateViaCache: 'none' }).catch(() => {
      // Registration failing (private mode, unsupported scheme) just means
      // the app keeps loading straight from the network.
    });
  });
};
