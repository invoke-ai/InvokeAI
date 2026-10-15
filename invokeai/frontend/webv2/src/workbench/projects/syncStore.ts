import type { Project } from '@workbench/projectContracts';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

import type { DeleteProjectBoards } from './api';
import type { ProjectPushOutcome, ProjectSchemaRefusal } from './projectFlush';

/** Only the sync engine mutates open projects; closed projects use HTTP. */

export interface ProjectSyncInfo {
  /** Server revision the next save is based on; null = never reached the server. */
  revision: number | null;
  /** True when the local document differs from what the server acknowledged. */
  isPendingPush: boolean;
  /** Present when the server requires a newer canvas schema than this client supports. */
  schemaRefusal?: ProjectSchemaRefusal;
  /** Divergent local work awaits an explicit user decision. */
  conflict?: { detectedAt: string; kind: 'deleted' | 'revision'; serverRevision?: number };
}

export interface ProjectSyncSnapshot {
  projects: Record<string, ProjectSyncInfo>;
  hasPendingChanges: boolean;
  lastSyncedAt: string | null;
  localDraftStatus: 'ok' | 'unavailable';
  recoverableDrafts: Array<{
    editorSessionId: string;
    generation: number;
    projectId: string;
    updatedAt: number;
  }>;
}

const EMPTY_PROJECT_SYNC: ProjectSyncSnapshot = {
  hasPendingChanges: false,
  lastSyncedAt: null,
  localDraftStatus: 'ok',
  projects: {},
  recoverableDrafts: [],
};
const store = createExternalStore<ProjectSyncSnapshot>(EMPTY_PROJECT_SYNC);

registerAccountOwnedResource({
  clear: () => {
    store.setSnapshot(EMPTY_PROJECT_SYNC);
  },
  name: 'project-sync',
});

/** Registered handles own open-project revisions and board state for their lifetime; null permits HTTP fallback. */
export interface OpenProjectHandle {
  /** Close the tab, after the project has been deleted on the server. */
  close: () => void;
  /** Delete through the sync engine's mutation queue so in-flight saves finish first. */
  deleteOnServer: (boards?: DeleteProjectBoards) => Promise<void>;
  /** Flush returns an acknowledgement outcome; callers reading server bytes must assert success. */
  flush: () => Promise<ProjectPushOutcome>;
  /** Persists unsaved canvas pixels into the document; rejects when they cannot be saved. */
  flushPixels: () => Promise<void>;
  /** The live project as the editor holds it now. */
  current: () => Project | undefined;
  /** Stop the autosave from recreating this project while it is being deleted. */
  markDeleted: () => void;
  /** Unmark failed deletes so autosave can resume. */
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

export const getProjectSyncSnapshot = store.getSnapshot;

export const subscribeProjectSync = store.subscribe;

export const reportProjectSync = (update: Omit<ProjectSyncSnapshot, 'lastSyncedAt'>): void => {
  store.setSnapshot({
    ...update,
    lastSyncedAt: update.hasPendingChanges ? store.getSnapshot().lastSyncedAt : new Date().toISOString(),
  });
};

export const reportProjectSyncEntry = (
  projectId: string,
  info: ProjectSyncInfo,
  update: Pick<ProjectSyncSnapshot, 'hasPendingChanges' | 'localDraftStatus' | 'recoverableDrafts'>
): void => {
  const snapshot = store.getSnapshot();
  const projects = { ...snapshot.projects, [projectId]: info };
  const hasPendingChanges =
    update.hasPendingChanges ||
    Object.values(projects).some(
      (project) => project.isPendingPush || project.conflict !== undefined || project.schemaRefusal !== undefined
    );
  store.setSnapshot({
    ...snapshot,
    ...update,
    hasPendingChanges,
    projects,
    lastSyncedAt: hasPendingChanges ? snapshot.lastSyncedAt : new Date().toISOString(),
  });
};

export const resolveProjectSyncConflict = (
  projectId: string,
  replacement?: { projectId: string; revision: number },
  hasServicePendingChanges = false
): void => {
  const snapshot = store.getSnapshot();
  const projects = { ...snapshot.projects };
  delete projects[projectId];
  if (replacement) {
    projects[replacement.projectId] = { isPendingPush: false, revision: replacement.revision };
  }
  const hasPendingChanges =
    hasServicePendingChanges ||
    Object.values(projects).some(
      (project) => project.isPendingPush || project.conflict !== undefined || project.schemaRefusal !== undefined
    );
  store.setSnapshot({ ...snapshot, hasPendingChanges, projects });
};
