/**
 * The plugin injects one {version, assets} manifest. Navigation is network-first, hashed assets and
 * content-versioned locales cache-first; API/socket traffic is untouched. Do not skipWaiting: old tabs may still
 * need lazy assets the new worker prunes.
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

  // Cache only current-manifest assets; activation prunes old entries.
  if (response.ok && ASSET_URLS.has(request.url)) {
    const cache = await caches.open(ASSET_CACHE);

    void cache.put(request, response.clone());
  }

  return response;
};

const respondVersionedLocale = async (request) => {
  const cache = await caches.open(RUNTIME_CACHE);
  const cached = await cache.match(request);

  if (cached) {
    return cached;
  }

  const response = await fetch(request);

  const contentType = response.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase();

  // Store JSON only: the backend's SPA fallback answers a missing file with a 200 index.html. Keep one entry per
  // locale file: drop every other version of this path, unversioned ones included.
  if (response.ok && contentType === 'application/json') {
    const copy = response.clone();

    void cache.keys(request, { ignoreSearch: true, ignoreVary: true }).then(async (entries) => {
      await Promise.all(entries.filter((entry) => entry.url !== request.url).map((entry) => cache.delete(entry)));
      await cache.put(request, copy);
    });
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

  // `platform/i18n/client.ts` adds the locale content version as `v`. Unversioned requests go to the network, so
  // none can be pinned in the cache.
  if (request.url.startsWith(LOCALES_PREFIX) && new URL(request.url).searchParams.has('v')) {
    event.respondWith(respondVersionedLocale(request));
  }
});
