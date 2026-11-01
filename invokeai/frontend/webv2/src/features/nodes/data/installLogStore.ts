import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

export interface CustomNodeInstallLogEntry {
  id: number;
  name: string;
  status: 'installing' | 'completed' | 'error' | 'uninstalled';
  /** The Git URL an install was submitted with, so a failed one can be retried; absent for uninstalls. */
  source?: string;
  message?: string;
  timestamp: number;
}

const LOG_LIMIT = 50;

const store = createExternalStore<{ log: CustomNodeInstallLogEntry[] }>({ log: [] });

let nextId = 1;

export const addCustomNodeInstallLogEntry = (
  entry: Omit<CustomNodeInstallLogEntry, 'id' | 'timestamp'>
): CustomNodeInstallLogEntry => {
  const nextEntry: CustomNodeInstallLogEntry = {
    ...entry,
    id: nextId,
    timestamp: Date.now(),
  };

  nextId += 1;
  store.patchSnapshot({ log: [nextEntry, ...store.getSnapshot().log].slice(0, LOG_LIMIT) });

  return nextEntry;
};

/** Resolve an entry in place (installing -> completed/error) so the queue row settles with its install. */
export const updateCustomNodeInstallLogEntry = (
  id: number,
  patch: Partial<Pick<CustomNodeInstallLogEntry, 'message' | 'name' | 'status'>>
): void => {
  store.patchSnapshot({
    log: store.getSnapshot().log.map((entry) => (entry.id === id ? { ...entry, ...patch } : entry)),
  });
};

export const removeCustomNodeInstallLogEntry = (id: number): void => {
  store.patchSnapshot({ log: store.getSnapshot().log.filter((entry) => entry.id !== id) });
};

/** Drop settled entries; an install still running keeps its row so its outcome has somewhere to land. */
export const clearFinishedCustomNodeInstalls = (): void => {
  store.patchSnapshot({ log: store.getSnapshot().log.filter((entry) => entry.status === 'installing') });
};

registerAccountOwnedResource({
  clear: () => {
    nextId = 1;
    store.patchSnapshot({ log: [] });
  },
  name: 'custom-node-install-log',
});

export const useCustomNodeInstallLog = (): CustomNodeInstallLogEntry[] => store.useSelector((snapshot) => snapshot.log);

/** Installs run one at a time: the backend clones synchronously and reports nothing until it finishes. */
export const useIsCustomNodeInstallRunning = (): boolean =>
  store.useSelector((snapshot) => snapshot.log.some((entry) => entry.status === 'installing'));

export const getCustomNodeInstallLogForTests = (): CustomNodeInstallLogEntry[] => store.getSnapshot().log;
