import { StorageError } from 'app/store/enhancers/reduxRemember/errors';
import { $authToken } from 'app/store/nanostores/authToken';
import { $projectId } from 'app/store/nanostores/projectId';
import type { UseStore } from 'idb-keyval';
import { clear, createStore as createIDBKeyValStore, get, set } from 'idb-keyval';
import { atom } from 'nanostores';
import type { Driver } from 'redux-remember';
import { getBaseUrl } from 'services/api';
import { buildAppInfoUrl } from 'services/api/endpoints/appInfo';

// Create a custom idb-keyval store (just needed to customize the name)
const $idbKeyValStore = atom<UseStore>(createIDBKeyValStore('invoke', 'invoke-store'));

export const clearIdbKeyValStore = () => {
  clear($idbKeyValStore.get());
};

// Create redux-remember driver, wrapping idb-keyval
export const idbKeyValDriver: Driver = {
  getItem: (key) => {
    try {
      return get(key, $idbKeyValStore.get());
    } catch (originalError) {
      throw new StorageError({
        key,
        projectId: $projectId.get(),
        originalError,
      });
    }
  },
  setItem: (key, value) => {
    try {
      return set(key, value, $idbKeyValStore.get());
    } catch (originalError) {
      throw new StorageError({
        key,
        value,
        projectId: $projectId.get(),
        originalError,
      });
    }
  },
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

export const serverBackedDriver: Driver = {
  getItem: async (key) => {
    try {
      const baseUrl = getBaseUrl();
      const path = buildAppInfoUrl('client_state', { key });
      const url = `${baseUrl}/${path}`;
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
      const baseUrl = getBaseUrl();
      const path = buildAppInfoUrl('client_state');
      const url = `${baseUrl}/${path}`;
      const headers = getHeaders({ 'content-type': 'application/json' });
      const res = await fetch(url, { headers, method: 'POST', body: JSON.stringify({ key, value }) });
      if (!res.ok) {
        throw new Error(`Response status: ${res.status}`);
      }
      return value;
    } catch (originalError) {
      throw new StorageError({
        key,
        value,
        projectId: $projectId.get(),
        originalError,
      });
    }
  },
};

export const resetClientState = async () => {
  const baseUrl = getBaseUrl();
  const path = buildAppInfoUrl('client_state');
  const url = `${baseUrl}/${path}`;
  const headers = getHeaders();
  const res = await fetch(url, { headers, method: 'DELETE' });
  if (!res.ok) {
    throw new Error(`Response status: ${res.status}`);
  }
};
