const AUTH_GENERATION_KEY = 'auth_generation';
const MEDIA_AUTH_LOCK = 'invokeai-media-auth';

const getAuthGeneration = () => {
  const value = Number(localStorage.getItem(AUTH_GENERATION_KEY) ?? 0);
  return Number.isSafeInteger(value) && value >= 0 ? value : 0;
};

export const captureAuthGeneration = () => getAuthGeneration();

export const beginAuthTransition = () => {
  const next = getAuthGeneration() + 1;
  localStorage.setItem(AUTH_GENERATION_KEY, String(next));
  return next;
};

export const shouldAcceptRefreshedToken = (requestToken: string, requestGeneration: number) =>
  getAuthGeneration() === requestGeneration && localStorage.getItem('auth_token') === requestToken;

let fallbackLock = Promise.resolve();

export const runWithMediaAuthLock = <T>(callback: () => T | PromiseLike<T>): Promise<T> => {
  if (typeof navigator !== 'undefined' && navigator.locks) {
    return navigator.locks.request(MEDIA_AUTH_LOCK, callback) as Promise<T>;
  }

  const result = fallbackLock.then(callback, callback);
  fallbackLock = result.then(
    () => undefined,
    () => undefined
  );
  return result;
};
