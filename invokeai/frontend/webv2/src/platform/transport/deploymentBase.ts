/** Production modules resolve the deployment prefix before /assets/; development falls back to the origin root. */
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

const getRuntimeOrigin = (): string => (typeof window === 'undefined' ? 'http://localhost' : window.location.origin);

/** Deployment origin plus prefix, without a trailing slash. */
export const getDeploymentBaseUrl = (): string => {
  const runtimeOrigin = getRuntimeOrigin();
  cachedDeploymentBaseUrl ??= deriveDeploymentBaseUrl(import.meta.url, runtimeOrigin, runtimeOrigin);

  return cachedDeploymentBaseUrl;
};

/** Empty at the domain root, otherwise a leading-slash prefix without a trailing slash. */
export const getDeploymentBasePath = (): string => {
  const pathname = new URL(getDeploymentBaseUrl()).pathname;

  return pathname === '/' ? '' : pathname.replace(/\/$/, '');
};
