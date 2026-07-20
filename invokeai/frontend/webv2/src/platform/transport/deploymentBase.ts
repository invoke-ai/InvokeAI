/**
 * Derives the backend deployment root (origin plus an optional reverse-proxy
 * prefix) from the emitted module URL. Vite emits production chunks beneath
 * `<deployment-root>/assets/`; development modules have no `/assets/` segment
 * and therefore keep the existing origin-root behavior.
 */
export const deriveDeploymentBaseUrl = (
  moduleUrl: string,
  fallbackOrigin: string,
  appOrigin = typeof window === 'undefined' ? fallbackOrigin : window.location.origin
): string => {
  try {
    const url = new URL(moduleUrl);

    if (url.origin !== appOrigin) {
      return fallbackOrigin;
    }

    const assetsIndex = url.pathname.lastIndexOf('/assets/');
    const prefix = assetsIndex >= 0 ? url.pathname.slice(0, assetsIndex) : '';

    return `${url.origin}${prefix}`;
  } catch {
    return fallbackOrigin;
  }
};

let cachedDeploymentBaseUrl: string | undefined;

/** Deployment origin plus prefix, without a trailing slash. */
export const getDeploymentBaseUrl = (): string => {
  cachedDeploymentBaseUrl ??= deriveDeploymentBaseUrl(import.meta.url, window.location.origin);

  return cachedDeploymentBaseUrl;
};

/** Empty at the domain root, otherwise a leading-slash prefix without a trailing slash. */
export const getDeploymentBasePath = (): string => {
  const pathname = new URL(getDeploymentBaseUrl()).pathname;

  return pathname === '/' ? '' : pathname.replace(/\/$/, '');
};
