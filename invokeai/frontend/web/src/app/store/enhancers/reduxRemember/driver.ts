import { logger } from 'app/logging/logger';
import { StorageError } from 'app/store/enhancers/reduxRemember/errors';
import { $projectId } from 'app/store/nanostores/projectId';
import type { Driver as ReduxRememberDriver } from 'redux-remember';
import { getBaseUrl } from 'services/api';
import { buildAppInfoUrl } from 'services/api/endpoints/appInfo';

const log = logger('system');

const getUrl = (key?: string) => {
  const baseUrl = getBaseUrl();
  const query: Record<string, string> = {};
  if (key) {
    query['key'] = key;
  }
  const path = buildAppInfoUrl('client_state', query);
  const url = `${baseUrl}/${path}`;
  return url;
};

const defaultGetItem = async (key: string): Promise<string | undefined> => {
  const url = getUrl(key);
  const res = await fetch(url, { method: 'GET' });
  if (!res.ok) {
    throw new Error(`Response status: ${res.status}`);
  }
  return res.json();
};

const defaultSetItem = async (key: string, value: string): Promise<string> => {
  const url = getUrl(key);
  const res = await fetch(url, { method: 'POST', body: value });
  if (!res.ok) {
    throw new Error(`Response status: ${res.status}`);
  }
  return res.json();
};

const defaultClear = async (): Promise<void> => {
  const url = getUrl();
  const res = await fetch(url, { method: 'DELETE' });
  if (!res.ok) {
    throw new Error(`Response status: ${res.status}`);
  }
};

export const buildStorage = (api?: {
  getItem: (key: string) => Promise<string | undefined>;
  setItem: (key: string, value: string) => Promise<string>;
  clear: () => Promise<void>;
}): {
  reduxRememberDriver: ReduxRememberDriver;
  clearStorage: () => Promise<void>;
  registerListeners: () => () => void;
} => {
  const _api = api ?? {
    getItem: defaultGetItem,
    setItem: defaultSetItem,
    clear: defaultClear,
  };
  // Persistence happens per slice. To track when persistence is in progress, maintain a ref count, incrementing
  // it when a slice is being persisted and decrementing it when the persistence is done.
  let persistRefCount = 0;

  // Keep track of the last persisted state for each key to avoid unnecessary network requests.
  //
  // `redux-remember` persists individual slices of state, so we can implicity denylist a slice by not giving it a
  // persist config.
  //
  // However, we may need to avoid persisting individual _fields_ of a slice. `redux-remember` does not provide a
  // way to do this directly.
  //
  // To accomplish this, we add a layer of logic on top of the `redux-remember`. In the state serializer function
  // provided to `redux-remember`, we can omit certain fields from the state that we do not want to persist. See
  // the implementation in `store.ts` for this logic.
  //
  // This logic is unknown to `redux-remember`. When an omitted field changes, it will still attempt to persist the
  // whole slice, even if the final, _serialized_ slice value is unchanged.
  //
  // To avoid unnecessary network requests, we keep track of the last persisted state for each key. If the value to
  // be persisted is the same as the last persisted value, we can skip the network request.
  const lastPersistedState = new Map<string, string | undefined>();

  const reduxRememberDriver: ReduxRememberDriver = {
    getItem: async (key) => {
      try {
        const value = await _api.getItem(key);
        lastPersistedState.set(key, value);
        log.trace({ key, last: lastPersistedState.get(key), next: value }, `Getting state for ${key}`);
        return value;
      } catch (originalError) {
        throw new StorageError({
          key,
          projectId: $projectId.get(),
          originalError,
        });
      }
    },
    setItem: async (key, value) => {
      try {
        persistRefCount++;
        if (lastPersistedState.get(key) === value) {
          log.trace(
            { key, last: lastPersistedState.get(key), next: value },
            `Skipping persist for ${key} as value is unchanged`
          );
          return value;
        }
        log.trace({ key, last: lastPersistedState.get(key), next: value }, `Persisting state for ${key}`);
        const resultValue = await _api.setItem(key, value);
        lastPersistedState.set(key, resultValue);
        return resultValue;
      } catch (originalError) {
        throw new StorageError({
          key,
          value,
          projectId: $projectId.get(),
          originalError,
        });
      } finally {
        persistRefCount--;
        if (persistRefCount < 0) {
          log.trace('Persist ref count is negative, resetting to 0');
          persistRefCount = 0;
        }
      }
    },
  };

  const clearStorage = async () => {
    try {
      persistRefCount++;
      await _api.clear();
    } catch {
      log.error('Failed to reset client state');
    } finally {
      persistRefCount--;
      lastPersistedState.clear();
      if (persistRefCount < 0) {
        log.trace('Persist ref count is negative, resetting to 0');
        persistRefCount = 0;
      }
    }
  };

  const registerListeners = () => {
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (persistRefCount > 0) {
        e.preventDefault();
      }
    };
    window.addEventListener('beforeunload', onBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', onBeforeUnload);
    };
  };

  return { reduxRememberDriver, clearStorage, registerListeners };
};
