/** Per-layer detection status for control layers; React-free and `useSyncExternalStore`-compatible. */

import type { ControlDetectionStatus } from '@workbench/canvas-operations/operationTypes';

export interface ControlDetectionStatusStore {
  /** The layer's status; a layer without an entry is idle. Snapshots are stable until the next `set`. */
  get(layerId: string): ControlDetectionStatus;
  /** Setting idle removes the entry, so the store only holds layers with something to show. */
  set(layerId: string, status: ControlDetectionStatus): void;
  subscribe(layerId: string, listener: () => void): () => void;
  /** Layer ids that currently hold a non-idle status. */
  layerIds(): string[];
  /** Resets every layer to idle, notifying their listeners. */
  clear(): void;
}

export const IDLE_CONTROL_DETECTION_STATUS: ControlDetectionStatus = Object.freeze({ status: 'idle' });

const sameStatus = (a: ControlDetectionStatus, b: ControlDetectionStatus): boolean => {
  if (a.status !== b.status) {
    return false;
  }
  if (a.status === 'processing' && b.status === 'processing') {
    return a.detector === b.detector;
  }
  if (a.status === 'error' && b.status === 'error') {
    return a.error.code === b.error.code && a.error.detail === b.error.detail;
  }
  return true;
};

export const createControlDetectionStatusStore = (): ControlDetectionStatusStore => {
  const statuses = new Map<string, ControlDetectionStatus>();
  const listeners = new Map<string, Set<() => void>>();

  const notify = (layerId: string): void => {
    for (const listener of listeners.get(layerId) ?? []) {
      listener();
    }
  };

  const set = (layerId: string, status: ControlDetectionStatus): void => {
    const previous = statuses.get(layerId) ?? IDLE_CONTROL_DETECTION_STATUS;
    if (sameStatus(previous, status)) {
      return;
    }
    if (status.status === 'idle') {
      statuses.delete(layerId);
    } else {
      statuses.set(layerId, status);
    }
    notify(layerId);
  };

  return {
    clear: () => {
      const ids = [...statuses.keys()];
      statuses.clear();
      for (const layerId of ids) {
        notify(layerId);
      }
    },
    get: (layerId) => statuses.get(layerId) ?? IDLE_CONTROL_DETECTION_STATUS,
    layerIds: () => [...statuses.keys()],
    set,
    subscribe: (layerId, listener) => {
      let layerListeners = listeners.get(layerId);
      if (!layerListeners) {
        layerListeners = new Set();
        listeners.set(layerId, layerListeners);
      }
      layerListeners.add(listener);
      return () => {
        const current = listeners.get(layerId);
        current?.delete(listener);
        if (current?.size === 0) {
          listeners.delete(layerId);
        }
      };
    },
  };
};
