import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

export interface CustomNodeInstallLogEntry {
  id: number;
  name: string;
  status: 'installing' | 'completed' | 'error' | 'uninstalled';
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

/** Resolve an entry in place (installing -> completed/error) so the activity badge can settle. */
export const updateCustomNodeInstallLogEntry = (
  id: number,
  patch: Partial<Pick<CustomNodeInstallLogEntry, 'message' | 'name' | 'status'>>
): void => {
  store.patchSnapshot({
    log: store.getSnapshot().log.map((entry) => (entry.id === id ? { ...entry, ...patch } : entry)),
  });
};

export const clearCustomNodeInstallLog = (): void => {
  store.patchSnapshot({ log: [] });
};

registerAccountOwnedResource({
  clear: () => {
    nextId = 1;
    clearCustomNodeInstallLog();
  },
  name: 'custom-node-install-log',
});

export const useCustomNodeInstallLog = (): CustomNodeInstallLogEntry[] => store.useSelector((snapshot) => snapshot.log);

export const getCustomNodeInstallLogForTests = (): CustomNodeInstallLogEntry[] => store.getSnapshot().log;
