import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

import type { ProjectPushOutcome } from './projectFlush';

/**
 * The bridge between the project sync layer and everything outside the editor.
 * {@link ProjectSyncSnapshot} is a read-only window for shell surfaces; {@link OpenProjectHandle}
 * is the other direction.
 *
 * ### One invariant, replacing three races
 *
 * A project the workbench holds is mutated only through the sync engine; every other project over
 * HTTP. Unconditional GET-and-PUT meant renaming an open project forked it into a conflict copy,
 * duplicating one copied what the server last acknowledged rather than what was on screen, and
 * deleting one raced the autosave about to recreate it. Now that a board commits with its project's
 * name, a stray write renames the board too.
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
  /**
   * Delete the project on the server, from inside the sync engine's mutation queue.
   *
   * The queue is the point. Marking the project deleted stops a save that has not begun but says
   * nothing to a `PUT` already on the wire, which returns 404 once the delete commits — and the
   * engine's answer to a 404 is to fork the local edits into a new server-side project, a copy of
   * the thing just deleted. Deleting *through* the queue lets the in-flight push finish first.
   */
  deleteOnServer: () => Promise<void>;
  /**
   * Push the live document, and report whether the server actually took it.
   *
   * Deliberately an outcome rather than a rejection. A push that did not land is recoverable — the
   * document is cached and the next save retries — so a rename or a closing tab is right to ignore
   * it. A caller about to read the project back from the server is not: see `assertProjectFlushed`.
   */
  flush: () => Promise<ProjectPushOutcome>;
  /** Stop the autosave from recreating this project while it is being deleted. */
  markDeleted: () => void;
  /**
   * Undo {@link markDeleted} after a deletion that did not happen.
   *
   * Paired with it here rather than left to call sites, because a project left marked never
   * autosaves again for the rest of the session — a silent, unrecoverable failure to notice.
   */
  unmarkDeleted: () => void;
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
