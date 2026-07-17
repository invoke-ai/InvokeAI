/**
 * Derives the deployment base URL (origin + optional sub-path prefix, WITHOUT a trailing slash)
 * from the URL of the app's entry module and a fallback origin.
 *
 * Because Vite is configured with `base: './'`, the entry chunk is always emitted under
 * `<deploy-root>/assets/`, so the deployment root is everything before `/assets/`:
 *
 *   https://example.com/invoke/assets/index-abc.js  -> https://example.com/invoke
 *   https://example.com/assets/index-abc.js          -> https://example.com
 *
 * In dev mode (Vite serves `/src/...`, no `/assets/` segment) and when served from the domain
 * root, the prefix is empty, so the result is just the origin - identical to the previous
 * behavior, leaving existing installations unaffected.
 *
 * Exported separately from {@link getDeploymentBaseUrl} so it can be unit-tested without relying
 * on `import.meta.url`.
 */
export const deriveDeploymentBaseUrl = (moduleUrl: string, fallbackOrigin: string): string => {
  try {
    const url = new URL(moduleUrl);
    // If the bundle is served from a different origin than the page (e.g. CDN-hosted assets in
    // `package` mode), sub-path detection on that foreign origin is meaningless - API/websocket/locale
    // requests must target the app origin, not the asset origin. Fall back to the app origin.
    if (typeof window !== 'undefined' && url.origin !== window.location.origin) {
      return fallbackOrigin;
    }
    // `lastIndexOf` rather than `indexOf` in case the deployment prefix itself contains an `/assets/`
    // segment (e.g. `/foo/assets/bar/`); Vite emits a single flat `assets/` dir at the deploy root.
    const assetsIdx = url.pathname.lastIndexOf('/assets/');
    const prefix = assetsIdx >= 0 ? url.pathname.slice(0, assetsIdx) : '';
    return `${url.origin}${prefix}`;
  } catch {
    return fallbackOrigin;
  }
};

let _cachedBaseUrl: string | undefined;

/**
 * Returns the deployment base URL: origin + optional sub-path prefix, WITHOUT a trailing slash.
 * See {@link deriveDeploymentBaseUrl} for how the prefix is determined.
 */
export const getDeploymentBaseUrl = (): string => {
  if (_cachedBaseUrl === undefined) {
    _cachedBaseUrl = deriveDeploymentBaseUrl(import.meta.url, window.location.origin);
  }
  return _cachedBaseUrl;
};

/**
 * Returns just the normalized sub-path prefix: `''` at the domain root, or e.g. `'/invoke'`
 * (leading slash, no trailing slash). Used for the React Router `basename` and the socket.io `path`.
 */
export const getBasePath = (): string => {
  const { pathname } = new URL(getDeploymentBaseUrl());
  return pathname === '/' ? '' : pathname.replace(/\/$/, '');
};
