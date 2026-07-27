import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

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

const EMPTY_PROJECT_SYNC: ProjectSyncSnapshot = {
  hasPendingChanges: false,
  lastSyncedAt: null,
  projects: {},
};
const store = createExternalStore<ProjectSyncSnapshot>(EMPTY_PROJECT_SYNC);

registerAccountOwnedResource({
  clear: () => {
    store.setSnapshot(EMPTY_PROJECT_SYNC);
  },
  name: 'project-sync',
});

export const useProjectSync = (): ProjectSyncSnapshot => store.useSnapshot();

export const useProjectSyncSelector = store.useSelector;

export const reportProjectSync = (update: Omit<ProjectSyncSnapshot, 'lastSyncedAt'>): void => {
  store.setSnapshot({
    ...update,
    lastSyncedAt: update.hasPendingChanges ? store.getSnapshot().lastSyncedAt : new Date().toISOString(),
  });
};
