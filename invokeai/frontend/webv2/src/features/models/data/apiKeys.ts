import { getUserStorageScope } from '@features/identity';

/**
 * HuggingFace credentials live server-side; Civitai keys are account-scoped browser storage and become
 * access_token only on matching install URLs.
 */

const CIVITAI_KEY_BASE_STORAGE_KEY = 'invokeai-webv2-civitai-api-key';

const getStorageKey = (): string => `${CIVITAI_KEY_BASE_STORAGE_KEY}${getUserStorageScope()}`;

export const getCivitaiApiKey = (): string | null => {
  try {
    return window.localStorage.getItem(getStorageKey());
  } catch {
    return null;
  }
};

export const setCivitaiApiKey = (key: string): void => {
  try {
    window.localStorage.setItem(getStorageKey(), key);
  } catch {
    // Storage unavailable (private mode/quota) — the key just is not persisted.
  }
};

export const clearCivitaiApiKey = (): void => {
  try {
    window.localStorage.removeItem(getStorageKey());
  } catch {
    // Ignore: nothing to clear if storage is unavailable.
  }
};

export const isCivitaiUrl = (source: string): boolean => {
  try {
    const host = new URL(source).hostname;

    return host === 'civitai.com' || host.endsWith('.civitai.com');
  } catch {
    return false;
  }
};

/** The access token to use for a given install source, if any is saved. */
export const getAccessTokenForSource = (source: string): string | undefined =>
  isCivitaiUrl(source) ? (getCivitaiApiKey() ?? undefined) : undefined;
