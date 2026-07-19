import type { HydratedWorkbenchSnapshot } from '@workbench/persistenceContracts';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { getUserStorageScope } from '@features/identity';
import { timeWorkbenchPerf } from '@workbench/performanceMarks';
import { localStorageWorkbenchPersistence, stripTransientWorkbenchState } from '@workbench/persistence';
import { createDraftProject, createInitialWorkbenchState } from '@workbench/workbenchState';

import {
  createProject as apiCreateProject,
  deleteClientStateValue,
  deleteProject as apiDeleteProject,
  getProject as apiGetProject,
  isProjectConflictError,
  isProjectNotFoundError,
  listProjects,
  setClientStateValue,
  updateProject as apiUpdateProject,
  type ProjectRecordDTO,
} from './api';
import { seedProjectLibrary, upsertProjectSummary } from './library';
import { fetchSessionBlob, serializeSessionBlob, SESSION_STATE_KEY } from './session';
import { reportProjectSync, type ProjectSyncInfo } from './syncStore';

/**
 * Backend-first workbench persistence (spec: Persistence Model).
 *
 * The backend database is the source of truth: each project is one
 * revision-versioned document on the server, and the small session blob
 * (open tabs + active project, with legacy account data) lives in the per-user
 * client-state KV. The user-scoped localStorage snapshot is kept as a
 * write-through cache so the workbench still loads and autosaves while the
 * backend is unreachable; un-pushed changes are replayed on the next save or
 * reconnect.
 *
 * Workbench state only ever holds the open projects (the session); the full
 * set of saved projects lives in the project library as summaries. Saving
 * pushes what is open and never deletes — projects leave the server only
 * through the library's explicit delete.
 *
 * Conflicts never lose work: when a save is based on a stale revision, the
 * server version wins for that project id and the local version is forked
 * into a "(recovered)" project beside it.
 */

const SYNC_MAP_BASE_KEY = 'invokeai:v7:webv2:workbench-sync';

interface SyncEntry {
  /** The server revision our next save is based on. */
  revision: number;
  /** Serialized form of the last document the server acknowledged. */
  pushedDoc: string | null;
}

export interface ProjectConflictResolution {
  projectId: string;
  /** The newer version that won on the server, now adopted locally. */
  serverProject: Project;
  /** The forked copy carrying the local edits that lost the race. */
  recoveredProject: Project;
}

export interface WorkbenchSaveResult {
  snapshot: HydratedWorkbenchSnapshot;
  conflicts: ProjectConflictResolution[];
  /** True when changes are cached locally but could not reach the backend. */
  hasPendingChanges: boolean;
}

export interface WorkbenchLoadOptions {
  /** Deep-linked project (/app?project=…) to include in the open set. */
  openProjectId?: string;
  /** Append a fresh draft project to the session (/app?new=1). */
  createNew?: boolean;
}

interface SyncedPersistenceState {
  /** Ids deleted in this runtime lifetime, guarding against racing saves. */
  deletedProjectIds: Set<string>;
  hasPending: boolean;
  lastPushedAccount: string | null;
  projectDocumentJsonCache: WeakMap<Project, { document: Record<string, unknown>; json: string }>;
  /** Server-known projects, keyed by project id. */
  syncEntries: Map<string, SyncEntry>;
}

const createSyncedPersistenceState = (): SyncedPersistenceState => ({
  deletedProjectIds: new Set(),
  hasPending: false,
  lastPushedAccount: null,
  projectDocumentJsonCache: new WeakMap(),
  syncEntries: new Map(),
});

/**
 * Undo/redo stacks are session-only (each entry is a full project snapshot,
 * far too heavy to autosave); everything else in the project document is the
 * project, verbatim.
 */
export const serializeProjectDocument = (project: Project): Record<string, unknown> => {
  const { undoRedo: _undoRedo, ...document } = project;

  return document;
};

const getSerializedProjectDocument = (
  syncState: SyncedPersistenceState,
  project: Project
): { document: Record<string, unknown>; json: string } => {
  const cached = syncState.projectDocumentJsonCache.get(project);

  if (cached) {
    return cached;
  }

  const document = serializeProjectDocument(project);
  const json = timeWorkbenchPerf(
    'workbench:project-document-stringify',
    { area: 'project-sync', kind: 'workbench', projectId: project.id },
    () => JSON.stringify(document)
  );
  const serialized = { document, json };

  syncState.projectDocumentJsonCache.set(project, serialized);

  return serialized;
};

const normalizeInvocationSourceId = (sourceId: unknown): unknown => {
  if (sourceId === 'project-graph') {
    return 'workflow';
  }

  if (sourceId === 'canvas-fill') {
    return 'canvas';
  }

  return sourceId;
};

const normalizeLegacyProjectDocument = (data: Record<string, unknown>): Record<string, unknown> => {
  const invocation = data.invocation;
  const queue = data.queue;

  return {
    ...data,
    invocation:
      invocation && typeof invocation === 'object'
        ? { ...invocation, sourceId: normalizeInvocationSourceId((invocation as { sourceId?: unknown }).sourceId) }
        : invocation,
    queue:
      queue && typeof queue === 'object' && Array.isArray((queue as { items?: unknown }).items)
        ? {
            ...queue,
            items: (queue as { items: unknown[] }).items.map((item) => {
              if (!item || typeof item !== 'object') {
                return item;
              }

              const snapshot = (item as { snapshot?: unknown }).snapshot;

              return {
                ...item,
                snapshot:
                  snapshot && typeof snapshot === 'object'
                    ? {
                        ...snapshot,
                        sourceId: normalizeInvocationSourceId((snapshot as { sourceId?: unknown }).sourceId),
                      }
                    : snapshot,
              };
            }),
          }
        : queue,
  };
};

export const deserializeProjectDocument = (data: Record<string, unknown>): Project | null => {
  const normalizedData = normalizeLegacyProjectDocument(data);

  if (
    typeof normalizedData.id !== 'string' ||
    typeof normalizedData.name !== 'string' ||
    typeof normalizedData.layout !== 'object' ||
    normalizedData.layout === null
  ) {
    return null;
  }

  return { ...normalizedData, undoRedo: { future: [], past: [] } } as unknown as Project;
};

const getSyncMapStorageKey = (): string => `${SYNC_MAP_BASE_KEY}${getUserStorageScope()}`;

/**
 * The revision map survives reloads so that, while offline, we can still tell
 * "this local project was synced before but is gone from the server (deleted
 * elsewhere — drop it)" apart from "this local project was created offline
 * (push it)".
 */
const persistSyncMap = (syncState: SyncedPersistenceState): void => {
  try {
    const revisions: Record<string, number> = {};

    for (const [projectId, entry] of syncState.syncEntries) {
      revisions[projectId] = entry.revision;
    }

    window.localStorage.setItem(getSyncMapStorageKey(), JSON.stringify({ revisions }));
  } catch {
    // Cache only; sync still works for this session.
  }
};

const loadPersistedRevisions = (): Record<string, number> => {
  try {
    const raw = window.localStorage.getItem(getSyncMapStorageKey());
    const parsed = raw ? (JSON.parse(raw) as { revisions?: Record<string, number> }) : null;

    return parsed?.revisions ?? {};
  } catch {
    return {};
  }
};

const createSnapshot = (state: WorkbenchState): HydratedWorkbenchSnapshot => ({
  savedAt: new Date().toISOString(),
  state: stripTransientWorkbenchState(state),
  version: 1,
});

/** Import a never-synced project to the server; returns false when it could not reach it. */
const pushNewProject = async (syncState: SyncedPersistenceState, project: Project): Promise<boolean> => {
  const document = serializeProjectDocument(project);

  try {
    const created = await apiCreateProject({ data: document, name: project.name, project_id: project.id });

    syncState.syncEntries.set(project.id, { pushedDoc: JSON.stringify(document), revision: created.revision });

    return true;
  } catch (error) {
    if (isProjectConflictError(error)) {
      // The id already exists server-side (e.g. a previous import raced a
      // reload). Adopt the server revision; the regular save path will PUT.
      try {
        const existing = await apiGetProject(project.id);

        syncState.syncEntries.set(project.id, {
          pushedDoc: JSON.stringify(existing.data),
          revision: existing.revision,
        });

        return true;
      } catch {
        return false;
      }
    }

    return false;
  }
};

/** Strip any number of stacked "(recovered)" suffixes left by older recoveries. */
const getRecoveryBaseName = (name: string): string => name.replace(/(\s*\((?:r|R)ecovered\))+$/u, '').trim() || name;

/**
 * Build the fork document for a conflicted project: lineage always points at
 * the root original (a recovery of a recovery still keys to the first
 * project), and the name never stacks suffixes.
 */
export const createRecoveredDocument = (
  project: Project,
  document: Record<string, unknown>
): { recoveredId: string; recoveredName: string; recoveredDocument: Record<string, unknown> } => {
  const recoveryOf = project.recoveryOf ?? project.id;
  const recoveredId = `${recoveryOf}-recovered-${Date.now().toString(36)}`;
  const recoveredName = `${getRecoveryBaseName(project.name)} (recovered)`;

  return {
    recoveredDocument: {
      ...document,
      id: recoveredId,
      name: recoveredName,
      recoveredAt: new Date().toISOString(),
      recoveryOf,
    },
    recoveredId,
    recoveredName,
  };
};

type ConflictOutcome =
  | { kind: 'adopted' }
  | { kind: 'retry' }
  | { kind: 'forked'; resolution: ProjectConflictResolution }
  | { kind: 'failed' };

/**
 * A save lost the revision race. Forking is the last resort — it only happens
 * when content actually diverged:
 *
 * - server content == what we tried to push → adopt the revision, done
 * - server content == the base this edit started from → revisions drifted
 *   without divergence (e.g. crash between an acknowledged PUT and the
 *   revision write); adopt the revision and retry the edit
 * - anything else → the server version keeps the id, the local edits fork
 *   into a "(recovered)" project so nothing is lost
 */
const recoverConflictingProject = async (
  syncState: SyncedPersistenceState,
  project: Project,
  document: Record<string, unknown>,
  documentJson: string,
  basePushedDoc: string | null
): Promise<ConflictOutcome> => {
  try {
    const server = await apiGetProject(project.id);
    const serverDocJson = JSON.stringify(server.data);

    syncState.syncEntries.set(project.id, { pushedDoc: serverDocJson, revision: server.revision });

    if (serverDocJson === documentJson) {
      return { kind: 'adopted' };
    }

    if (basePushedDoc !== null && serverDocJson === basePushedDoc) {
      return { kind: 'retry' };
    }

    const serverProject = deserializeProjectDocument(server.data);

    if (!serverProject) {
      return { kind: 'failed' };
    }

    const { recoveredDocument, recoveredId, recoveredName } = createRecoveredDocument(project, document);
    const recoveredProject = deserializeProjectDocument(recoveredDocument);

    if (!recoveredProject) {
      return { kind: 'failed' };
    }

    const created = await apiCreateProject({ data: recoveredDocument, name: recoveredName, project_id: recoveredId });

    syncState.syncEntries.set(recoveredId, {
      pushedDoc: JSON.stringify(recoveredDocument),
      revision: created.revision,
    });

    return { kind: 'forked', resolution: { projectId: project.id, recoveredProject, serverProject } };
  } catch {
    return { kind: 'failed' };
  }
};

const pushProject = async (
  syncState: SyncedPersistenceState,
  project: Project,
  conflicts: ProjectConflictResolution[]
): Promise<string> => {
  const { document, json: documentJson } = getSerializedProjectDocument(syncState, project);
  const entry = syncState.syncEntries.get(project.id);

  if (syncState.deletedProjectIds.has(project.id) || entry?.pushedDoc === documentJson) {
    return documentJson;
  }

  if (!entry) {
    if (!(await pushNewProject(syncState, project))) {
      syncState.hasPending = true;
    }

    return documentJson;
  }

  try {
    const updated = await apiUpdateProject(project.id, {
      data: document,
      expected_revision: entry.revision,
      name: project.name,
    });

    syncState.syncEntries.set(project.id, { pushedDoc: documentJson, revision: updated.revision });
  } catch (error) {
    if (isProjectConflictError(error)) {
      const outcome = await recoverConflictingProject(syncState, project, document, documentJson, entry.pushedDoc);

      if (outcome.kind === 'retry') {
        try {
          const baseRevision = syncState.syncEntries.get(project.id)?.revision ?? entry.revision;
          const retried = await apiUpdateProject(project.id, {
            data: document,
            expected_revision: baseRevision,
            name: project.name,
          });

          syncState.syncEntries.set(project.id, { pushedDoc: documentJson, revision: retried.revision });
        } catch {
          // A genuinely concurrent writer; the next save re-evaluates.
          syncState.hasPending = true;
        }
      } else if (outcome.kind === 'forked') {
        conflicts.push(outcome.resolution);
      } else if (outcome.kind === 'failed') {
        syncState.hasPending = true;
      }
    } else if (isProjectNotFoundError(error)) {
      // Deleted on another device while we held local edits: recreate rather
      // than drop the user's work.
      syncState.syncEntries.delete(project.id);

      if (!(await pushNewProject(syncState, project))) {
        syncState.hasPending = true;
      }
    } else {
      syncState.hasPending = true;
    }
  }

  return documentJson;
};

const pushSessionState = async (syncState: SyncedPersistenceState, state: WorkbenchState): Promise<void> => {
  const blob = serializeSessionBlob(state);

  if (blob === syncState.lastPushedAccount) {
    return;
  }

  try {
    await setClientStateValue(SESSION_STATE_KEY, blob);
    syncState.lastPushedAccount = blob;
  } catch {
    syncState.hasPending = true;
  }
};

const loadFromBackend = async (
  syncState: SyncedPersistenceState,
  local: HydratedWorkbenchSnapshot | null,
  options?: WorkbenchLoadOptions
): Promise<HydratedWorkbenchSnapshot> => {
  const [summaries, sessionBlob] = await Promise.all([listProjects(), fetchSessionBlob()]);
  const persistedRevisions = loadPersistedRevisions();

  seedProjectLibrary(summaries);

  // First contact: a backend with no projects adopts the browser's existing
  // workbench (one-time import of the pre-backend localStorage data).
  if (summaries.length === 0 && local && local.state.projects.length > 0) {
    for (const project of local.state.projects) {
      if (!(await pushNewProject(syncState, project))) {
        syncState.hasPending = true;
      }

      const entry = syncState.syncEntries.get(project.id);

      upsertProjectSummary({ id: project.id, name: project.name, revision: entry?.revision ?? null });
    }

    await pushSessionState(syncState, local.state);
    persistSyncMap(syncState);

    return local;
  }

  // The session blob says which projects are open as tabs; blobs from before
  // the library/session split have no open set, and for those every project
  // opens (exactly what that version of the app did). A deep-linked project
  // joins the set.
  const summaryIds = new Set(summaries.map((summary) => summary.project_id));
  const requestedIds = sessionBlob?.openProjectIds ?? summaries.map((summary) => summary.project_id);
  const openIds: string[] = [];

  for (const id of [...requestedIds, ...(options?.openProjectId ? [options.openProjectId] : [])]) {
    if (summaryIds.has(id) && !openIds.includes(id)) {
      openIds.push(id);
    }
  }

  // Only the open set is hydrated into full documents; everything else stays
  // a summary in the library. A project deleted between list and get is
  // simply dropped from the session.
  const records = await Promise.all(openIds.map((id) => apiGetProject(id).catch(() => null)));
  const serverProjects: Project[] = [];

  for (const record of records) {
    if (!record) {
      continue;
    }

    const project = deserializeProjectDocument(record.data);

    if (project) {
      serverProjects.push(project);
      syncState.syncEntries.set(record.project_id, {
        pushedDoc: JSON.stringify(record.data),
        revision: record.revision,
      });
    }
  }

  // Local projects the server does not have: keep the ones never synced
  // (created offline; the next save pushes them) and drop the ones with a
  // recorded revision (synced before, so they were deleted elsewhere).
  const serverIds = new Set(serverProjects.map((project) => project.id));
  const offlineCreated = (local?.state.projects ?? []).filter(
    (project) => !serverIds.has(project.id) && persistedRevisions[project.id] === undefined
  );

  if (offlineCreated.length > 0) {
    syncState.hasPending = true;
  }

  let projects = [...serverProjects, ...offlineCreated];

  if (sessionBlob) {
    syncState.lastPushedAccount = JSON.stringify(sessionBlob);
  }

  const base = local?.state ?? createInitialWorkbenchState();
  let activeProjectId =
    options?.openProjectId && projects.some((project) => project.id === options.openProjectId)
      ? options.openProjectId
      : sessionBlob && projects.some((project) => project.id === sessionBlob.activeProjectId)
        ? sessionBlob.activeProjectId
        : projects.some((project) => project.id === base.activeProjectId)
          ? base.activeProjectId
          : (projects[0]?.id ?? '');

  // An explicit "new project" request, or a session with nothing to open
  // (first run, or /app reached directly with an empty session): start a
  // fresh draft. The first autosave creates it server-side.
  if (options?.createNew || projects.length === 0) {
    const draft = createDraftProject(projects);

    projects = [...projects, draft];
    activeProjectId = draft.id;
  }

  const state: WorkbenchState = {
    ...base,
    account: sessionBlob?.account ?? base.account,
    activeProjectId,
    autosave: { status: 'idle' },
    backendConnection: { status: 'connecting' },
    notifications: [],
    projects,
  };

  if (serializeSessionBlob(state) !== syncState.lastPushedAccount) {
    syncState.hasPending = true;
  }

  reportProjectSync({
    hasPendingChanges: syncState.hasPending,
    projects: Object.fromEntries(
      projects.map((project) => {
        const entry = syncState.syncEntries.get(project.id);

        return [project.id, { isPendingPush: entry === undefined, revision: entry?.revision ?? null }];
      })
    ),
  });

  persistSyncMap(syncState);

  const snapshot = createSnapshot(state);

  // Refresh the offline cache with what the server gave us.
  await localStorageWorkbenchPersistence.saveWorkbench(state);

  return snapshot;
};

export interface SyncedWorkbenchPersistence {
  adoptProjectRecord(record: ProjectRecordDTO): Project | null;
  clearWorkbench(): Promise<void>;
  flushProjectToServer(project: Project): Promise<void>;
  hasPendingChanges(): boolean;
  hydrateProjectFromServer(projectId: string): Promise<Project | null>;
  loadWorkbench(options?: WorkbenchLoadOptions): Promise<HydratedWorkbenchSnapshot | null>;
  markProjectDeleted(projectId: string): void;
  persistEmptySession(state: WorkbenchState): Promise<void>;
  releaseProjectSync(projectId: string): void;
  saveWorkbench(state: WorkbenchState): Promise<WorkbenchSaveResult>;
  unmarkProjectDeleted(projectId: string): void;
}

/**
 * One-shot maintenance operation: deletes server projects and the session
 * blob, then clears the local cache, project library, and persisted sync map.
 * Independent of any mounted Workbench lifetime; callers are expected to
 * reload afterwards.
 */
export const clearAllWorkbenchData = async (): Promise<void> => {
  try {
    const summaries = await listProjects();

    await Promise.all(summaries.map((summary) => apiDeleteProject(summary.project_id)));
    await deleteClientStateValue(SESSION_STATE_KEY);
  } catch {
    // Backend unreachable; at least reset this browser.
  }

  seedProjectLibrary([]);

  try {
    window.localStorage.removeItem(getSyncMapStorageKey());
  } catch {
    // Nothing to clear if storage is unavailable.
  }

  await localStorageWorkbenchPersistence.clearWorkbench();
};

/** Construct one synchronization lifetime per mounted Workbench. */
export const createSyncedWorkbenchPersistence = (): SyncedWorkbenchPersistence => {
  const syncState = createSyncedPersistenceState();

  const adoptProjectRecord = (record: ProjectRecordDTO): Project | null => {
    const project = deserializeProjectDocument(record.data);

    if (!project) {
      return null;
    }

    syncState.syncEntries.set(record.project_id, {
      pushedDoc: JSON.stringify(record.data),
      revision: record.revision,
    });
    persistSyncMap(syncState);

    return project;
  };

  return {
    adoptProjectRecord,
    /** Clear everywhere: server projects + session blob, local cache, sync map, and this lifetime's sync state. */
    async clearWorkbench(): Promise<void> {
      await clearAllWorkbenchData();

      syncState.syncEntries.clear();
      syncState.deletedProjectIds.clear();
      syncState.lastPushedAccount = null;
      syncState.hasPending = false;
    },
    async flushProjectToServer(project): Promise<void> {
      const conflicts: ProjectConflictResolution[] = [];

      await pushProject(syncState, project, conflicts);
      persistSyncMap(syncState);
    },
    hasPendingChanges(): boolean {
      return syncState.hasPending;
    },
    async hydrateProjectFromServer(projectId): Promise<Project | null> {
      try {
        return adoptProjectRecord(await apiGetProject(projectId));
      } catch {
        return null;
      }
    },
    /**
     * Load from the backend, falling back to the localStorage cache when it is
     * unreachable. Returns null when there is nothing anywhere (first run with
     * no backend); the caller then keeps its default boot state.
     */
    async loadWorkbench(options?: WorkbenchLoadOptions): Promise<HydratedWorkbenchSnapshot | null> {
      let local: HydratedWorkbenchSnapshot | null = null;

      try {
        local = await localStorageWorkbenchPersistence.loadWorkbench();
      } catch {
        local = null;
      }

      try {
        return await loadFromBackend(syncState, local, options);
      } catch {
        // Backend unreachable: run from the cache; saves queue up locally and
        // replay on reconnect.
        syncState.hasPending = true;

        const persistedRevisions = loadPersistedRevisions();

        for (const [projectId, revision] of Object.entries(persistedRevisions)) {
          syncState.syncEntries.set(projectId, { pushedDoc: null, revision });
        }

        reportProjectSync({
          hasPendingChanges: true,
          projects: Object.fromEntries(
            (local?.state.projects ?? []).map((project) => [
              project.id,
              { isPendingPush: true, revision: persistedRevisions[project.id] ?? null },
            ])
          ),
        });

        // A cache holding an empty session (last tab closed offline) cannot
        // hydrate the editor; boot a fresh draft instead.
        return local && local.state.projects.length > 0 ? local : null;
      }
    },

    /**
     * Write-through save: localStorage cache always, then every dirty open
     * project and the session blob to the backend. Revision conflicts come
     * back as resolutions for the caller to apply to workbench state. Saving
     * never deletes anything: a project absent from state is merely closed,
     * and removal happens only through the library's explicit delete.
     */
    markProjectDeleted(projectId): void {
      syncState.deletedProjectIds.add(projectId);
      syncState.syncEntries.delete(projectId);
      persistSyncMap(syncState);
    },
    async persistEmptySession(state): Promise<void> {
      const emptied: WorkbenchState = { ...state, activeProjectId: '', projects: [] };

      await localStorageWorkbenchPersistence.saveWorkbench(emptied);

      try {
        const blob = serializeSessionBlob(emptied);

        await setClientStateValue(SESSION_STATE_KEY, blob);
        syncState.lastPushedAccount = blob;
      } catch {
        syncState.hasPending = true;
      }
    },
    releaseProjectSync(projectId): void {
      syncState.syncEntries.delete(projectId);
      persistSyncMap(syncState);
    },
    async saveWorkbench(state: WorkbenchState): Promise<WorkbenchSaveResult> {
      const snapshot = createSnapshot(state);

      await localStorageWorkbenchPersistence.saveWorkbench(state);

      syncState.hasPending = false;

      const conflicts: ProjectConflictResolution[] = [];
      const projectSyncInfos: Record<string, ProjectSyncInfo> = {};

      await pushSessionState(syncState, state);

      for (const project of state.projects) {
        const lastAckedDoc = syncState.syncEntries.get(project.id)?.pushedDoc ?? null;
        const documentJson = await pushProject(syncState, project, conflicts);
        const entry = syncState.syncEntries.get(project.id);

        projectSyncInfos[project.id] = {
          isPendingPush: entry?.pushedDoc !== documentJson,
          revision: entry?.revision ?? null,
        };

        // The server acknowledged new content for this project — keep the
        // library summary current without a refetch.
        if (entry && entry.pushedDoc === documentJson && lastAckedDoc !== documentJson) {
          upsertProjectSummary({ id: project.id, name: project.name, revision: entry.revision });
        }
      }

      persistSyncMap(syncState);
      reportProjectSync({ hasPendingChanges: syncState.hasPending, projects: projectSyncInfos });

      return { conflicts, hasPendingChanges: syncState.hasPending, snapshot };
    },
    unmarkProjectDeleted(projectId): void {
      syncState.deletedProjectIds.delete(projectId);
    },
  };
};
