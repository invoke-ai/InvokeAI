import { logger } from 'app/logging/logger';
import { useAppDispatch, useAppStore } from 'app/store/storeHooks';
import { canvasSnapshotRestored } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { zCanvasState } from 'features/controlLayers/store/types';
import { useCallback, useEffect, useState } from 'react';
import { serializeError } from 'serialize-error';
import { buildV1Url, getBaseUrl } from 'services/api';
import type { JsonObject } from 'type-fest';

const log = logger('canvas');

const SNAPSHOT_PREFIX = 'canvas_snapshot:';

const getAuthHeaders = (): Record<string, string> => {
  const headers: Record<string, string> = {};
  if (typeof window !== 'undefined' && window.localStorage) {
    const token = localStorage.getItem('auth_token');
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
  }
  return headers;
};

const getUrl = (endpoint: string, query?: Record<string, string>) => {
  const baseUrl = getBaseUrl();
  const path = buildV1Url(`client_state/default/${endpoint}`, query);
  return `${baseUrl}/${path}`;
};

export type SnapshotInfo = {
  key: string;
  name: string;
};

export const useCanvasSnapshots = () => {
  const dispatch = useAppDispatch();
  const store = useAppStore();
  const [snapshots, setSnapshots] = useState<SnapshotInfo[]>([]);

  const fetchSnapshots = useCallback(async () => {
    try {
      const url = getUrl('get_keys_by_prefix', { prefix: SNAPSHOT_PREFIX });
      const res = await fetch(url, { method: 'GET', headers: getAuthHeaders() });
      if (!res.ok) {
        throw new Error(`Response status: ${res.status}`);
      }
      const keys: string[] = await res.json();
      setSnapshots(
        keys.map((key) => ({
          key,
          name: key.slice(SNAPSHOT_PREFIX.length),
        }))
      );
    } catch (e) {
      log.error({ error: serializeError(e) } as JsonObject, 'Failed to fetch snapshots');
    }
  }, []);

  const saveSnapshot = useCallback(
    async (name: string) => {
      try {
        const state = selectCanvasSlice(store.getState());
        const value = JSON.stringify(state);
        const key = `${SNAPSHOT_PREFIX}${name}`;
        const url = getUrl('set_by_key', { key });
        const res = await fetch(url, {
          method: 'POST',
          body: value,
          headers: getAuthHeaders(),
        });
        if (!res.ok) {
          throw new Error(`Response status: ${res.status}`);
        }
        await fetchSnapshots();
        return true;
      } catch (e) {
        log.error({ error: serializeError(e) } as JsonObject, 'Failed to save snapshot');
        return false;
      }
    },
    [store, fetchSnapshots]
  );

  const restoreSnapshot = useCallback(
    async (key: string) => {
      try {
        const url = getUrl('get_by_key', { key });
        const res = await fetch(url, { method: 'GET', headers: getAuthHeaders() });
        if (!res.ok) {
          throw new Error(`Response status: ${res.status}`);
        }
        const raw = await res.json();
        const parsed = JSON.parse(raw);
        const canvasState = zCanvasState.parse(parsed);
        dispatch(canvasSnapshotRestored(canvasState));
        return true;
      } catch (e) {
        log.error({ error: serializeError(e) } as JsonObject, 'Failed to restore snapshot');
        return false;
      }
    },
    [dispatch]
  );

  const deleteSnapshot = useCallback(
    async (key: string) => {
      try {
        const url = getUrl('delete_by_key', { key });
        const res = await fetch(url, { method: 'POST', headers: getAuthHeaders() });
        if (!res.ok) {
          throw new Error(`Response status: ${res.status}`);
        }
        await fetchSnapshots();
        return true;
      } catch (e) {
        log.error({ error: serializeError(e) } as JsonObject, 'Failed to delete snapshot');
        return false;
      }
    },
    [fetchSnapshots]
  );

  useEffect(() => {
    fetchSnapshots();
  }, [fetchSnapshots]);

  return {
    snapshots,
    saveSnapshot,
    restoreSnapshot,
    deleteSnapshot,
  };
};
