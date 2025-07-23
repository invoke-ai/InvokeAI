import { logger } from 'app/logging/logger';
import { StorageError } from 'app/store/enhancers/reduxRemember/errors';
import { $authToken } from 'app/store/nanostores/authToken';
import { $projectId } from 'app/store/nanostores/projectId';
import { $queueId } from 'app/store/nanostores/queueId';
import { atom } from 'nanostores';
import type { Driver } from 'redux-remember';
import { getBaseUrl } from 'services/api';
import { buildAppInfoUrl } from 'services/api/endpoints/appInfo';

export type StorageDriverApi = {
  getItem: (key: string) => Promise<any>;
  setItem: (key: string, value: any) => Promise<any>;
  clear: () => Promise<void>;
};

const log = logger('system');

// Persistence happens per slice. To track when persistence is in progress, maintain a ref count, incrementing
// it when a slice is being persisted and decrementing it when the persistence is done.
let persistRefCount = 0;

// Keep track of the last persisted state for each key to avoid unnecessary network requests.
const lastPersistedState = new Map<string, unknown>();

const getUrl = (key?: string) => {
  const baseUrl = getBaseUrl();
  const query: Record<string, string> = {};
  if (key) {
    query['key'] = key;
  }
  const queueId = $queueId.get();
  if (queueId) {
    query['queueId'] = queueId;
  }
  const path = buildAppInfoUrl('client_state', query);
  const url = `${baseUrl}/${path}`;
  return url;
};

const getHeaders = (extra?: Record<string, string>) => {
  const headers = new Headers();
  const authToken = $authToken.get();
  if (authToken) {
    headers.set('Authorization', `Bearer ${authToken}`);
  }
  const projectId = $projectId.get();
  if (projectId) {
    headers.set('project-id', projectId);
  }
  for (const [key, value] of Object.entries(extra ?? {})) {
    headers.set(key, value);
  }
  return headers;
};

export const buildDriver = (api?: StorageDriverApi): Driver => {
  return {
    getItem: async (key) => {
      try {
        if (api) {
          log.trace(`Using provided API to get item for key "${key}"`);
          return await api.getItem(key);
        }
        const url = getUrl(key);
        const headers = getHeaders();
        const res = await fetch(url, { headers, method: 'GET' });
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
        if (api) {
          log.trace(`Using provided API to get item for key "${key}"`);
          return await api.setItem(key, value);
        }
        // Deep equality check to avoid noop persist network requests.
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
        // be persisted is the same as the last persisted value, we skip the network request.
        if (lastPersistedState.get(key) === value) {
          log.trace(`Skipping persist for key "${key}" as value is unchanged.`);
          return value;
        }
        const url = getUrl(key);
        const headers = getHeaders({ 'content-type': 'application/json' });
        const res = await fetch(url, { headers, method: 'POST', body: JSON.stringify(value) });
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
          log.warn('Persist ref count is negative, resetting to 0');
          persistRefCount = 0;
        }
      }
    },
  };
};

export const $resetClientState = atom(() => {});

export const buildResetClientState = (api?: StorageDriverApi) => async () => {
  try {
    persistRefCount++;
    if (api) {
      log.trace('Using provided API to reset client state');
      await api.clear();
      return;
    }
    const url = getUrl();
    const headers = getHeaders();
    const res = await fetch(url, { headers, method: 'DELETE' });
    if (!res.ok) {
      throw new Error(`Response status: ${res.status}`);
    }
  } catch {
    log.error('Failed to reset client state');
  } finally {
    persistRefCount--;
    lastPersistedState.clear();
    if (persistRefCount < 0) {
      log.warn('Persist ref count is negative, resetting to 0');
      persistRefCount = 0;
    }
  }
};

window.addEventListener('beforeunload', (e) => {
  if (persistRefCount > 0) {
    e.preventDefault();
  }
});
