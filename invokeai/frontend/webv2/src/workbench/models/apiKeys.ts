/**
 * Client-side API key storage for remote model sources. The HuggingFace token
 * lives on the backend (`/api/v2/models/hf_login`); the Civitai key has no
 * backend home, so it is kept in this browser's localStorage and passed as the
 * `access_token` query param when installing from a matching URL.
 */

const CIVITAI_KEY_STORAGE_KEY = 'invokeai-webv2-civitai-api-key';

export const getCivitaiApiKey = (): string | null => {
  try {
    return window.localStorage.getItem(CIVITAI_KEY_STORAGE_KEY);
  } catch {
    return null;
  }
};

export const setCivitaiApiKey = (key: string): void => {
  try {
    window.localStorage.setItem(CIVITAI_KEY_STORAGE_KEY, key);
  } catch {
    // Storage unavailable (private mode/quota) — the key just is not persisted.
  }
};

export const clearCivitaiApiKey = (): void => {
  try {
    window.localStorage.removeItem(CIVITAI_KEY_STORAGE_KEY);
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
