import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/**
 * The bridge between the project sync layer and everything outside the editor.
 *
 * Two things live here. {@link ProjectSyncSnapshot} is a read-only window for shell surfaces (the
 * Project panel's debug section), written by `syncedPersistence` after each load/save pass.
 * {@link OpenProjectHandle} is the other direction: a way for non-workbench code to mutate a project
 * the workbench is currently holding, *through* the sync engine rather than behind its back.
 *
 * ### One invariant, replacing three races
 *
 * A project the workbench holds is mutated only through the sync engine; every other project is
 * mutated over HTTP. The library used to GET-and-PUT unconditionally, which meant renaming an open
 * project landed a write beside its revision chain — the next autosave then saw a revision it did
 * not expect and forked the project into a conflict copy. Duplicating one copied whatever the
 * server last acknowledged, which is not what the person could see on screen. And deleting one
 * raced the autosave that was about to recreate it.
 *
 * Now that a project's board commits with its name, a stray write is worse still: it renames the
 * board too. So the branch is explicit, and there is exactly one of it per operation.
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

/**
 * What a mounted editor can do to a project it holds, for callers that are not the editor.
 *
 * Every method routes through the workbench reducer or the sync engine, so the project's document,
 * its revision chain and its board stay in step. The handle exists only while the project is open;
 * `getOpenProject` returning `null` is the signal to go over HTTP instead.
 */
export interface OpenProjectHandle {
  /** Close the tab, after the project has been deleted on the server. */
  close: () => void;
  /** Push the live document and wait for the server to acknowledge it. */
  flush: () => Promise<void>;
  /** Stop the autosave from recreating this project while it is being deleted. */
  markDeleted: () => void;
  /** Rename through the reducer, then flush — so the project and its board rename together. */
  rename: (name: string) => Promise<void>;
}

const openProjects = new Map<string, OpenProjectHandle>();

registerAccountOwnedResource({
  clear: () => {
    openProjects.clear();
  },
  name: 'open-project-handles',
});

export const registerOpenProject = (projectId: string, handle: OpenProjectHandle): void => {
  openProjects.set(projectId, handle);
};

export const unregisterOpenProject = (projectId: string): void => {
  openProjects.delete(projectId);
};

/** The mounted editor's handle on this project, or `null` when nothing holds it. */
export const getOpenProject = (projectId: string): OpenProjectHandle | null => openProjects.get(projectId) ?? null;

export const useProjectSync = (): ProjectSyncSnapshot => store.useSnapshot();

export const useProjectSyncSelector = store.useSelector;

export const reportProjectSync = (update: Omit<ProjectSyncSnapshot, 'lastSyncedAt'>): void => {
  store.setSnapshot({
    ...update,
    lastSyncedAt: update.hasPendingChanges ? store.getSnapshot().lastSyncedAt : new Date().toISOString(),
  });
};
