import { logger } from 'app/logging/logger';
import { StorageError } from 'app/store/enhancers/reduxRemember/errors';
import { $authToken } from 'app/store/nanostores/authToken';
import { $projectId } from 'app/store/nanostores/projectId';
import { $queueId } from 'app/store/nanostores/queueId';
import type { UseStore } from 'idb-keyval';
import { createStore as idbCreateStore, del as idbDel, get as idbGet } from 'idb-keyval';
import type { Driver } from 'redux-remember';
import { serializeError } from 'serialize-error';
import { buildV1Url, getBaseUrl } from 'services/api';
import type { JsonObject } from 'type-fest';

const log = logger('system');

const getUrl = (endpoint: 'get_by_key' | 'set_by_key' | 'delete', key?: string) => {
  const baseUrl = getBaseUrl();
  const query: Record<string, string> = {};
  if (key) {
    query['key'] = key;
  }

  const path = buildV1Url(`client_state/${$queueId.get()}/${endpoint}`, query);
  const url = `${baseUrl}/${path}`;
  return url;
};

const getHeaders = () => {
  const headers = new Headers();
  const authToken = $authToken.get();
  const projectId = $projectId.get();
  if (authToken) {
    headers.set('Authorization', `Bearer ${authToken}`);
  }
  if (projectId) {
    headers.set('project-id', projectId);
  }
  return headers;
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
// To avoid unnecessary network requests, we keep track of the last persisted state for each key in this map.
// If the value to be persisted is the same as the last persisted value, we will skip the network request.
const lastPersistedState = new Map<string, string | undefined>();

// As of v6.3.0, we use server-backed storage for client state. This replaces the previous IndexedDB-based storage,
// which was implemented using `idb-keyval`.
//
// To facilitate a smooth transition, we implement a migration strategy that attempts to retrieve values from IndexedDB
// and persist them to the new server-backed storage. This is done on a best-effort basis.

// These constants were used in the previous IndexedDB-based storage implementation.
const IDB_DB_NAME = 'invoke';
const IDB_STORE_NAME = 'invoke-store';
const IDB_STORAGE_PREFIX = '@@invokeai-';

// Lazy store creation
let _idbKeyValStore: UseStore | null = null;
const getIdbKeyValStore = () => {
  if (_idbKeyValStore === null) {
    _idbKeyValStore = idbCreateStore(IDB_DB_NAME, IDB_STORE_NAME);
  }
  return _idbKeyValStore;
};

const getIdbKey = (key: string) => {
  return `${IDB_STORAGE_PREFIX}${key}`;
};

const getItem = async (key: string) => {
  try {
    const url = getUrl('get_by_key', key);
    const headers = getHeaders();
    const res = await fetch(url, { method: 'GET', headers });
    if (!res.ok) {
      throw new Error(`Response status: ${res.status}`);
    }
    const value = await res.json();

    // Best-effort migration from IndexedDB to the new storage system
    log.trace({ key, value }, 'Server-backed storage value retrieved');

    if (!value) {
      const idbKey = getIdbKey(key);
      try {
        // It's a bit tricky to query IndexedDB directly to check if value exists, so we use `idb-keyval` to do it.
        // Thing is, `idb-keyval` requires you to create a store to query it. End result - we are creating a store
        // even if we don't use it for anything besides checking if the key is present.
        const idbKeyValStore = getIdbKeyValStore();
        const idbValue = await idbGet(idbKey, idbKeyValStore);
        if (idbValue) {
          log.debug(
            { key, idbKey, idbValue },
            'No value in server-backed storage, but found value in IndexedDB - attempting migration'
          );
          await idbDel(idbKey, idbKeyValStore);
          await setItem(key, idbValue);
          log.debug({ key, idbKey, idbValue }, 'Migration successful');
          return idbValue;
        }
      } catch (error) {
        // Just log if IndexedDB retrieval fails - this is a best-effort migration.
        log.debug(
          { key, idbKey, error: serializeError(error) } as JsonObject,
          'Error checking for or migrating from IndexedDB'
        );
      }
    }

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
};

const setItem = async (key: string, value: string) => {
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
    const url = getUrl('set_by_key', key);
    const headers = getHeaders();
    const res = await fetch(url, { method: 'POST', headers, body: value });
    if (!res.ok) {
      throw new Error(`Response status: ${res.status}`);
    }
    const resultValue = await res.json();
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
};

export const reduxRememberDriver: Driver = { getItem, setItem };

export const clearStorage = async () => {
  try {
    persistRefCount++;
    const url = getUrl('delete');
    const headers = getHeaders();
    const res = await fetch(url, { method: 'POST', headers });
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

export const addStorageListeners = () => {
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
