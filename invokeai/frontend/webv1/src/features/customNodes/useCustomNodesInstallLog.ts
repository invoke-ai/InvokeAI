import { useStore } from '@nanostores/react';
import { atom } from 'nanostores';
import { useCallback } from 'react';

export type InstallLogEntry = {
  id: string;
  name: string;
  status: 'installing' | 'completed' | 'error' | 'uninstalled';
  message?: string;
  timestamp: number;
};

export const $installLog = atom<InstallLogEntry[]>([]);

let nextId = 0;

/**
 * Resets the internal ID counter. Only for testing.
 */
export const _resetIdCounter = () => {
  nextId = 0;
};

export const addInstallLogEntry = (entry: Omit<InstallLogEntry, 'id' | 'timestamp'>): InstallLogEntry => {
  const newEntry: InstallLogEntry = {
    ...entry,
    id: String(nextId++),
    timestamp: Date.now(),
  };
  $installLog.set([newEntry, ...$installLog.get()]);
  return newEntry;
};

export const clearInstallLog = () => {
  $installLog.set([]);
};

export const useCustomNodesInstallLog = () => {
  const log = useStore($installLog);

  const addLogEntry = useCallback((entry: Omit<InstallLogEntry, 'id' | 'timestamp'>) => {
    addInstallLogEntry(entry);
  }, []);

  return { log, addLogEntry, clearLog: clearInstallLog };
};
