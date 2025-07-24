/* eslint-disable @typescript-eslint/no-explicit-any */

import { logger } from 'app/logging/logger';
import { StorageError } from 'app/store/enhancers/reduxRemember/errors';
import { $projectId } from 'app/store/nanostores/projectId';
import type { Driver as ReduxRememberDriver } from 'redux-remember';
import { getBaseUrl } from 'services/api';
import { buildAppInfoUrl } from 'services/api/endpoints/appInfo';

const log = logger('system');

const buildOSSServerBackedDriver = (): {
  reduxRememberDriver: ReduxRememberDriver;
  clearStorage: () => Promise<void>;
  registerListeners: () => () => void;
} => {
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
  const lastPersistedState = new Map<string, unknown>();

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

  const reduxRememberDriver: ReduxRememberDriver = {
    getItem: async (key) => {
      try {
        const url = getUrl(key);
        const res = await fetch(url, { method: 'GET' });
        if (!res.ok) {
          throw new Error(`Response status: ${res.status}`);
        }
        const json = await res.json();
        return json;
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
          log.trace(`Skipping persist for key "${key}" as value is unchanged.`);
          return value;
        }
        const url = getUrl(key);
        const headers = new Headers({
          'Content-Type': 'application/json',
        });
        const res = await fetch(url, { method: 'POST', headers, body: value });
        if (!res.ok) {
          throw new Error(`Response status: ${res.status}`);
        }
        lastPersistedState.set(key, value);
        return value;
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
      const url = getUrl();
      const res = await fetch(url, { method: 'DELETE' });
      if (!res.ok) {
        throw new Error(`Response status: ${res.status}`);
      }
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

const buildCustomDriver = (api: {
  getItem: (key: string) => Promise<any>;
  setItem: (key: string, value: any) => Promise<any>;
  clear: () => Promise<void>;
}): {
  reduxRememberDriver: ReduxRememberDriver;
  clearStorage: () => Promise<void>;
  registerListeners: () => () => void;
} => {
  // See the comment in `buildOSSServerBackedDriver` for an explanation of this variable.
  let persistRefCount = 0;

  // See the comment in `buildOSSServerBackedDriver` for an explanation of this variable.
  const lastPersistedState = new Map<string, unknown>();

  const reduxRememberDriver: ReduxRememberDriver = {
    getItem: async (key) => {
      try {
        log.trace(`Getting client state for key "${key}"`);
        return await api.getItem(key);
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
          log.trace(`Skipping setting client state for key "${key}" as value is unchanged`);
          return value;
        }
        log.trace(`Setting client state for key "${key}", ${value}`);
        await api.setItem(key, value);
        lastPersistedState.set(key, value);
        return value;
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
      log.trace('Clearing client state');
      await api.clear();
    } catch {
      log.error('Failed to clear client state');
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

export const buildStorageApi = (api?: {
  getItem: (key: string) => Promise<any>;
  setItem: (key: string, value: any) => Promise<any>;
  clear: () => Promise<void>;
}) => {
  if (api) {
    return buildCustomDriver(api);
  } else {
    return buildOSSServerBackedDriver();
  }
};
