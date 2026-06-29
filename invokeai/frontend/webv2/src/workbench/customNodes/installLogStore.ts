import { createExternalStore } from '@workbench/externalStore';

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

export const clearCustomNodeInstallLog = (): void => {
  store.patchSnapshot({ log: [] });
};

export const useCustomNodeInstallLog = (): CustomNodeInstallLogEntry[] => store.useSelector((snapshot) => snapshot.log);
