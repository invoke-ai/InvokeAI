import { StorageError } from 'app/store/enhancers/reduxRemember/errors';
import { $authToken } from 'app/store/nanostores/authToken';
import { $projectId } from 'app/store/nanostores/projectId';
import { $queueId } from 'app/store/nanostores/queueId';
import { $isPendingPersist } from 'app/store/store';
import type { Driver } from 'redux-remember';
import { getBaseUrl } from 'services/api';
import { buildAppInfoUrl } from 'services/api/endpoints/appInfo';

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

export const serverBackedDriver: Driver = {
  getItem: async (key) => {
    try {
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
      const url = getUrl(key);
      const headers = getHeaders({ 'content-type': 'application/json' });
      const res = await fetch(url, { headers, method: 'POST', body: JSON.stringify(value) });
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
  const url = getUrl();
  const headers = getHeaders();
  const res = await fetch(url, { headers, method: 'DELETE' });
  if (!res.ok) {
    throw new Error(`Response status: ${res.status}`);
  }
};

window.addEventListener('beforeunload', (e) => {
  if ($isPendingPersist.get()) {
    e.preventDefault();
  }
});
