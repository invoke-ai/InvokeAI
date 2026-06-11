import { useSyncExternalStore } from 'react';

/**
 * Minimal factory for module-level stores backed by `useSyncExternalStore`.
 * For backend-owned or session-lived state shared across widget surfaces
 * (model library, install jobs, starter catalog) where the workbench reducer
 * is the wrong home. See `models/modelsStore.ts` for the canonical usage.
 */

/** A bare listener channel, for auxiliary data living beside a snapshot. */
export interface ListenerChannel {
  subscribe: (listener: () => void) => () => void;
  notify: () => void;
}

export const createListenerChannel = (): ListenerChannel => {
  const listeners = new Set<() => void>();

  return {
    notify: () => {
      for (const listener of listeners) {
        listener();
      }
    },
    subscribe: (listener) => {
      listeners.add(listener);

      return () => {
        listeners.delete(listener);
      };
    },
  };
};

export interface ExternalStore<Snapshot extends object> extends ListenerChannel {
  getSnapshot: () => Snapshot;
  /** Replace the snapshot and notify subscribers. */
  setSnapshot: (next: Snapshot) => void;
  /** Merge a partial snapshot and notify subscribers. */
  patchSnapshot: (next: Partial<Snapshot>) => void;
  /** Replace without notifying — for data read imperatively, never rendered. */
  setSnapshotSilently: (next: Snapshot) => void;
  /** Subscribe the calling component to the snapshot. */
  useSnapshot: () => Snapshot;
}

export const createExternalStore = <Snapshot extends object>(initialSnapshot: Snapshot): ExternalStore<Snapshot> => {
  let snapshot = initialSnapshot;
  const channel = createListenerChannel();

  const getSnapshot = (): Snapshot => snapshot;

  const setSnapshot = (next: Snapshot): void => {
    snapshot = next;
    channel.notify();
  };

  const useSnapshot = (): Snapshot => useSyncExternalStore(channel.subscribe, getSnapshot);

  return {
    ...channel,
    getSnapshot,
    patchSnapshot: (next) => setSnapshot({ ...snapshot, ...next }),
    setSnapshot,
    setSnapshotSilently: (next) => {
      snapshot = next;
    },
    useSnapshot,
  };
};
