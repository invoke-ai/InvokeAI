/** Register only in production after load to avoid boot contention; bypass the HTTP cache for worker updates. */
export const registerServiceWorker = (): void => {
  if (!import.meta.env.PROD || typeof navigator === 'undefined' || !('serviceWorker' in navigator)) {
    return;
  }

  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js', { updateViaCache: 'none' }).catch(() => {
      // Registration failure leaves ordinary network loading available.
    });
  });
};
