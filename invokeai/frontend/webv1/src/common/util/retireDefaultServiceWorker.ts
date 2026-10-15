/** Release the default UI's worker before legacy loads its differently shaped locale files. */
export const retireDefaultServiceWorker = async (deploymentBaseUrl: string): Promise<boolean> => {
  if (!('serviceWorker' in navigator)) {
    return false;
  }
  const scriptUrl = `${deploymentBaseUrl}/sw.js`;
  const registration = await navigator.serviceWorker.getRegistration(`${deploymentBaseUrl}/`);
  if (registration?.active?.scriptURL !== scriptUrl) {
    return false;
  }
  await registration.unregister();
  // Asset caches are disposable. Never clear localStorage or IndexedDB recovery drafts.
  if ('caches' in window) {
    const names = await caches.keys();
    const indexUrl = `${deploymentBaseUrl}/index.html`;
    const assetPrefix = `${deploymentBaseUrl}/assets/`;
    const localePrefix = `${deploymentBaseUrl}/locales/`;
    await Promise.all(
      names
        .filter(
          (name) => name.startsWith('invokeai-shell-') || name === 'invokeai-assets' || name === 'invokeai-runtime'
        )
        .map(async (name) => {
          const cache = await caches.open(name);
          const requests = await cache.keys();
          // Cache names are origin-wide, so another deployment may share each bucket.
          await Promise.all(
            requests
              .filter(({ url }) => url === indexUrl || url.startsWith(assetPrefix) || url.startsWith(localePrefix))
              .map((request) => cache.delete(request))
          );
        })
    );
  }
  // Unregistration leaves the current document controlled until its next navigation.
  if (navigator.serviceWorker.controller?.scriptURL === scriptUrl) {
    window.location.reload();
    return true;
  }
  return false;
};
