/**
 * App-shell service worker. The manifest read below is a placeholder that
 * `scripts/service-worker-plugin.mjs` replaces at build time with
 * `{ version, assets }` — the build's content hash and its emitted asset file
 * names. The placeholder must appear exactly once; the plugin enforces this.
 *
 * Strategy:
 * - Navigations are network-first with the cached shell as offline fallback,
 *   so a plain reload always picks up a new deploy — no update UI needed.
 * - Hashed `assets/*` files are cache-first: their names change when their
 *   content does, so a cache hit is always correct and repeat boots skip the
 *   network entirely.
 * - Locale files are stale-while-revalidate: instant from cache, refreshed in
 *   the background.
 * - Everything else (API, sockets) is untouched.
 *
 * Deliberately no `skipWaiting()`: a new worker prunes assets its manifest no
 * longer lists, and a still-open old tab may yet lazy-load one of them. The
 * new worker activates once the old tabs are gone.
 */

const MANIFEST = self.__SW_MANIFEST__;

const SHELL_CACHE = `invokeai-shell-${MANIFEST.version}`;
const ASSET_CACHE = 'invokeai-assets';
const RUNTIME_CACHE = 'invokeai-runtime';

const INDEX_URL = new URL('index.html', self.registration.scope).href;
const ASSETS_PREFIX = new URL('assets/', self.registration.scope).href;
const LOCALES_PREFIX = new URL('locales/', self.registration.scope).href;
const ASSET_URLS = new Set(MANIFEST.assets.map((fileName) => new URL(fileName, self.registration.scope).href));

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(SHELL_CACHE).then((cache) => cache.add(new Request(INDEX_URL, { cache: 'no-cache' }))));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();

      await Promise.all(
        names
          .filter((name) => name.startsWith('invokeai-shell-') && name !== SHELL_CACHE)
          .map((name) => caches.delete(name))
      );

      const assetCache = await caches.open(ASSET_CACHE);
      const cachedRequests = await assetCache.keys();

      await Promise.all(
        cachedRequests.filter((request) => !ASSET_URLS.has(request.url)).map((request) => assetCache.delete(request))
      );

      await self.clients.claim();
    })()
  );
});

const respondToNavigation = async (request) => {
  try {
    const response = await fetch(request);

    if (response.ok) {
      const cache = await caches.open(SHELL_CACHE);

      void cache.put(INDEX_URL, response.clone());
    }

    return response;
  } catch (error) {
    const cached = await caches.match(INDEX_URL);

    if (cached) {
      return cached;
    }

    throw error;
  }
};

const respondCacheFirst = async (request) => {
  const cached = await caches.match(request, { cacheName: ASSET_CACHE });

  if (cached) {
    return cached;
  }

  const response = await fetch(request);

  // Only current-manifest assets are written back, which keeps the cache
  // bounded; the activate handler prunes the previous build's entries.
  if (response.ok && ASSET_URLS.has(request.url)) {
    const cache = await caches.open(ASSET_CACHE);

    void cache.put(request, response.clone());
  }

  return response;
};

const respondStaleWhileRevalidate = async (request) => {
  const cache = await caches.open(RUNTIME_CACHE);
  const cached = await cache.match(request);
  const refresh = fetch(request)
    .then((response) => {
      if (response.ok) {
        void cache.put(request, response.clone());
      }

      return response;
    })
    .catch(() => undefined);

  if (cached) {
    return cached;
  }

  const response = await refresh;

  if (!response) {
    throw new TypeError(`Offline and not cached: ${request.url}`);
  }

  return response;
};

self.addEventListener('fetch', (event) => {
  const { request } = event;

  if (request.method !== 'GET' || !request.url.startsWith(self.location.origin)) {
    return;
  }

  if (request.mode === 'navigate') {
    event.respondWith(respondToNavigation(request));
    return;
  }

  if (request.url.startsWith(ASSETS_PREFIX)) {
    event.respondWith(respondCacheFirst(request));
    return;
  }

  if (request.url.startsWith(LOCALES_PREFIX)) {
    event.respondWith(respondStaleWhileRevalidate(request));
  }
});
