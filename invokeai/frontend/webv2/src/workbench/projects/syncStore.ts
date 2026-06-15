import { createExternalStore } from '@workbench/externalStore';

/**
 * Read-only window into the project sync layer for shell surfaces (the
 * Project panel's debug section). Written exclusively by `syncedPersistence`
 * after each load/save pass; everything here is informational.
 */

export interface ProjectSyncInfo {
  /** Server revision the next save is based on; null = never reached the server. */
  revision: number | null;
  /** True when the local document differs from what the server acknowledged. */
  isPendingPush: boolean;
}

export interface ProjectSyncSnapshot {
  projects: Record<string, ProjectSyncInfo>;
  hasPendingChanges: boolean;
  lastSyncedAt: string | null;
}

const store = createExternalStore<ProjectSyncSnapshot>({
  hasPendingChanges: false,
  lastSyncedAt: null,
  projects: {},
});

export const useProjectSync = (): ProjectSyncSnapshot => store.useSnapshot();

export const reportProjectSync = (update: Omit<ProjectSyncSnapshot, 'lastSyncedAt'>): void => {
  store.setSnapshot({
    ...update,
    lastSyncedAt: update.hasPendingChanges ? store.getSnapshot().lastSyncedAt : new Date().toISOString(),
  });
};
