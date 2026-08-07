import type { HydratedWorkbenchSnapshot } from '@workbench/persistenceContracts';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { assertAccountScopeCurrent, captureAccountScope, type AccountScope } from '@platform/state/accountLifecycle';
import { timeWorkbenchPerf } from '@workbench/performanceMarks';
import {
  createLocalStorageWorkbenchPersistence,
  stripTransientWorkbenchState,
  type WorkbenchPersistenceService,
} from '@workbench/persistence';
import {
  createDraftProject,
  createInitialWorkbenchState,
  normalizeWorkbenchProject,
  withAuthoritativeProjectBoard,
} from '@workbench/workbenchState';

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
import { recordProjectCover } from './covers';
import { seedProjectLibrary, upsertProjectSummary } from './library';
import { selectCoverImageName } from './projectAssets';
import {
  applyAuthoritativeProjectBoard,
  isProjectDocumentShape,
  normalizeLegacyProjectDocument,
  serializeProjectDocument,
} from './projectDocument';
import { fetchSessionBlob, serializeSessionBlob, SESSION_STATE_KEY } from './session';
import { reportProjectSync, type ProjectSyncInfo } from './syncStore';

export { serializeProjectDocument } from './projectDocument';

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

/** Local edits rescued from a project that was deleted on another device. */
export interface ProjectDeletionFork {
  projectId: string;
  /** The fork carrying the local edits, under a fresh id and its own board. */
  recoveredProject: Project;
}

/** A board the server assigned to a project this save created. */
export interface ProjectBoardAssignment {
  boardId: string;
  projectId: string;
}

export interface WorkbenchSaveResult {
  snapshot: HydratedWorkbenchSnapshot;
  conflicts: ProjectConflictResolution[];
  /** True when changes are cached locally but could not reach the backend. */
  hasPendingChanges: boolean;
  /**
   * Boards the server minted for projects created during this save. A draft is created by
   * persistence rather than by the editor, so the create response is the only place its board id
   * exists — discarding it would leave the project pointing at no board until the next reload.
   */
  projectBoardAssignments: ProjectBoardAssignment[];
  /**
   * Projects deleted elsewhere whose unsaved local edits were forked rather than re-created. The
   * deletion stands; the work does not.
   */
  deletedProjectForks: ProjectDeletionFork[];
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
  /**
   * Ids that have already been forked because the server no longer had them.
   *
   * The fork exists on the server from the moment it is made, but the original stays in the
   * aggregate until the reconciliation reaches it. In that window `pushProject` would find no sync
   * entry and re-create the project under its old id — undoing the very deletion the fork exists to
   * respect. So a forked id is never pushed again.
   */
  forkedProjectIds: Set<string>;
  hasPending: boolean;
  localPersistence: WorkbenchPersistenceService;
  lastPushedAccount: string | null;
  /** Immutable owner captured when this synchronization lifetime was constructed. */
  owner: AccountScope;
  /**
   * What the server said that the aggregate has not been told yet, drained by the next save.
   *
   * A push happens from a save *and* from a targeted flush, and only the save has somewhere to
   * return an outcome to. Holding them here rather than in the caller's local arrays is what makes
   * "the next save carries them" true rather than aspirational: a flush that forks a project or
   * resolves a conflict cannot silently drop the answer.
   */
  pendingBoardAssignments: ProjectBoardAssignment[];
  pendingConflicts: ProjectConflictResolution[];
  pendingDeletedForks: ProjectDeletionFork[];
  projectDocumentJsonCache: WeakMap<Project, { document: Record<string, unknown>; json: string }>;
  /** Server-known projects, keyed by project id. */
  syncEntries: Map<string, SyncEntry>;
}

const createSyncedPersistenceState = (owner: AccountScope): SyncedPersistenceState => ({
  deletedProjectIds: new Set(),
  forkedProjectIds: new Set(),
  hasPending: false,
  localPersistence: createLocalStorageWorkbenchPersistence(owner.storageSuffix),
  lastPushedAccount: null,
  owner,
  pendingBoardAssignments: [],
  pendingConflicts: [],
  pendingDeletedForks: [],
  projectDocumentJsonCache: new WeakMap(),
  syncEntries: new Map(),
});

const assertOwner = (syncState: SyncedPersistenceState): void => {
  assertAccountScopeCurrent(syncState.owner);
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
  // The cache is keyed by project identity, so this runs once per document
  // version — the only moments a project's cover can have changed. Recording it
  // here rather than after the push keeps one seam instead of three, and costs
  // nothing when the push fails: the cover names an image that exists either
  // way, and `recordProjectCover` is a no-op when the answer has not moved.
  recordProjectCover(project.id, selectCoverImageName(document), syncState.owner);

  return serialized;
};

/**
 * Rehydrate a *server record*, which knows the project's real board.
 *
 * The document's own `projectBoardId` is a cache and can be stale — written by another install, or
 * replaced by the migration when the board it named turned out to be ambiguous. Overwriting it
 * here means every path that reads a project from the server agrees on one answer, and the next
 * autosave writes the correction back. The saved destination is left alone: it is a deliberate
 * choice, and `getGallerySelectedBoardId` resolves it against the boards that actually exist.
 */
const deserializeProjectRecord = (record: ProjectRecordDTO): Project | null => {
  const project = deserializeProjectDocument(
    applyAuthoritativeProjectBoard(record.data, record.board_id, { selectBoard: false })
  );

  // Again after rehydration, because the document may have had no gallery values for the first
  // patch to land in — see `withAuthoritativeProjectBoard`.
  return project === null ? null : withAuthoritativeProjectBoard(project, record.board_id);
};

/**
 * Rehydrate a document into a live project. This is the half of the codec that
 * needs the aggregate reducer, so it stays here rather than in
 * `./projectDocument`; Launchpad callers reach it through a dynamic import.
 */
export const deserializeProjectDocument = (data: Record<string, unknown>): Project | null => {
  const normalizedData = normalizeLegacyProjectDocument(data);

  if (!isProjectDocumentShape(normalizedData)) {
    return null;
  }

  return normalizeWorkbenchProject({
    ...normalizedData,
    undoRedo: { future: [], past: [] },
  } as unknown as Project);
};

const getSyncMapStorageKey = (syncState: SyncedPersistenceState): string =>
  `${SYNC_MAP_BASE_KEY}${syncState.owner.storageSuffix}`;

/**
 * The revision map survives reloads so that, while offline, we can still tell
 * "this local project was synced before but is gone from the server (deleted
 * elsewhere — drop it)" apart from "this local project was created offline
 * (push it)".
 */
const persistSyncMap = (syncState: SyncedPersistenceState): void => {
  assertOwner(syncState);

  try {
    const revisions: Record<string, number> = {};

    for (const [projectId, entry] of syncState.syncEntries) {
      revisions[projectId] = entry.revision;
    }

    window.localStorage.setItem(getSyncMapStorageKey(syncState), JSON.stringify({ revisions }));
  } catch {
    // Cache only; sync still works for this session.
  }
};

const loadPersistedRevisions = (syncState: SyncedPersistenceState): Record<string, number> => {
  assertOwner(syncState);

  try {
    const raw = window.localStorage.getItem(getSyncMapStorageKey(syncState));
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
  assertOwner(syncState);
  const document = serializeProjectDocument(project);

  try {
    const created = await apiCreateProject(
      { data: document, name: project.name, project_id: project.id },
      syncState.owner.signal
    );

    assertOwner(syncState);
    syncState.syncEntries.set(project.id, { pushedDoc: JSON.stringify(document), revision: created.revision });
    syncState.pendingBoardAssignments.push({ boardId: created.board_id, projectId: project.id });

    return true;
  } catch (error) {
    assertOwner(syncState);

    if (isProjectConflictError(error)) {
      // The id already exists server-side (e.g. a previous import raced a
      // reload). Adopt the server revision; the regular save path will PUT.
      try {
        const existing = await apiGetProject(project.id, syncState.owner.signal);

        assertOwner(syncState);
        syncState.syncEntries.set(project.id, {
          pushedDoc: JSON.stringify(existing.data),
          revision: existing.revision,
        });
        syncState.pendingBoardAssignments.push({ boardId: existing.board_id, projectId: project.id });

        return true;
      } catch {
        assertOwner(syncState);

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
  assertOwner(syncState);

  try {
    const server = await apiGetProject(project.id, syncState.owner.signal);

    assertOwner(syncState);
    const serverDocJson = JSON.stringify(server.data);

    syncState.syncEntries.set(project.id, { pushedDoc: serverDocJson, revision: server.revision });

    if (serverDocJson === documentJson) {
      return { kind: 'adopted' };
    }

    if (basePushedDoc !== null && serverDocJson === basePushedDoc) {
      return { kind: 'retry' };
    }

    const serverProject = deserializeProjectRecord(server);

    if (!serverProject) {
      return { kind: 'failed' };
    }

    const { recoveredDocument, recoveredId, recoveredName } = createRecoveredDocument(project, document);
    const recoveredProject = deserializeProjectDocument(recoveredDocument);

    if (!recoveredProject) {
      return { kind: 'failed' };
    }

    const created = await apiCreateProject(
      { data: recoveredDocument, name: recoveredName, project_id: recoveredId },
      syncState.owner.signal
    );

    assertOwner(syncState);
    syncState.syncEntries.set(recoveredId, {
      pushedDoc: JSON.stringify(recoveredDocument),
      revision: created.revision,
    });

    return { kind: 'forked', resolution: { projectId: project.id, recoveredProject, serverProject } };
  } catch {
    assertOwner(syncState);

    return { kind: 'failed' };
  }
};

/**
 * Rescue the local edits of a project the server no longer has.
 *
 * Deliberately a *fork*, not a re-create. Pushing the original id back would resurrect a project
 * the user deleted — on every device, and here as a project whose board went with the deletion.
 * The fork gets a fresh id, and with it a fresh board from the create response.
 */
const forkDeletedProject = async (
  syncState: SyncedPersistenceState,
  project: Project,
  document: Record<string, unknown>
): Promise<ProjectDeletionFork | null> => {
  assertOwner(syncState);

  const { recoveredDocument, recoveredId, recoveredName } = createRecoveredDocument(project, document);
  const recoveredProject = deserializeProjectDocument(recoveredDocument);

  if (!recoveredProject) {
    return null;
  }

  try {
    const created = await apiCreateProject(
      { data: recoveredDocument, name: recoveredName, project_id: recoveredId },
      syncState.owner.signal
    );

    assertOwner(syncState);
    syncState.syncEntries.set(recoveredId, {
      pushedDoc: JSON.stringify(recoveredDocument),
      revision: created.revision,
    });
    syncState.pendingBoardAssignments.push({ boardId: created.board_id, projectId: recoveredId });

    return { projectId: project.id, recoveredProject };
  } catch {
    assertOwner(syncState);

    return null;
  }
};

const pushProject = async (syncState: SyncedPersistenceState, project: Project): Promise<string> => {
  assertOwner(syncState);
  const { document, json: documentJson } = getSerializedProjectDocument(syncState, project);
  const entry = syncState.syncEntries.get(project.id);

  if (
    syncState.deletedProjectIds.has(project.id) ||
    syncState.forkedProjectIds.has(project.id) ||
    entry?.pushedDoc === documentJson
  ) {
    return documentJson;
  }

  if (!entry) {
    if (!(await pushNewProject(syncState, project))) {
      assertOwner(syncState);
      syncState.hasPending = true;
    }

    assertOwner(syncState);
    return documentJson;
  }

  try {
    const updated = await apiUpdateProject(
      project.id,
      {
        data: document,
        expected_revision: entry.revision,
        name: project.name,
      },
      syncState.owner.signal
    );

    assertOwner(syncState);
    syncState.syncEntries.set(project.id, { pushedDoc: documentJson, revision: updated.revision });
  } catch (error) {
    assertOwner(syncState);

    if (isProjectConflictError(error)) {
      const outcome = await recoverConflictingProject(syncState, project, document, documentJson, entry.pushedDoc);

      assertOwner(syncState);
      if (outcome.kind === 'retry') {
        try {
          const baseRevision = syncState.syncEntries.get(project.id)?.revision ?? entry.revision;
          const retried = await apiUpdateProject(
            project.id,
            {
              data: document,
              expected_revision: baseRevision,
              name: project.name,
            },
            syncState.owner.signal
          );

          assertOwner(syncState);
          syncState.syncEntries.set(project.id, { pushedDoc: documentJson, revision: retried.revision });
        } catch {
          assertOwner(syncState);
          // A genuinely concurrent writer; the next save re-evaluates.
          syncState.hasPending = true;
        }
      } else if (outcome.kind === 'forked') {
        syncState.pendingConflicts.push(outcome.resolution);
      } else if (outcome.kind === 'failed') {
        syncState.hasPending = true;
      }
    } else if (isProjectNotFoundError(error)) {
      // Deleted on another device while we held local edits. Re-creating under the same id would
      // undo that deletion — the project would reappear on every device, and on this one it would
      // be a project whose board the deletion already took. Fork instead: the deletion stands, and
      // the local work survives as a recovered project with its own id and its own board.
      syncState.syncEntries.delete(project.id);

      const fork = await forkDeletedProject(syncState, project, document);

      assertOwner(syncState);
      if (fork === null) {
        syncState.hasPending = true;
      } else {
        // Recorded before the aggregate hears about it, so nothing re-creates the original id in
        // the meantime — see `forkedProjectIds`.
        syncState.forkedProjectIds.add(project.id);
        syncState.pendingDeletedForks.push(fork);
      }
    } else {
      syncState.hasPending = true;
    }
  }

  assertOwner(syncState);
  return documentJson;
};

const pushSessionState = async (syncState: SyncedPersistenceState, state: WorkbenchState): Promise<void> => {
  assertOwner(syncState);
  const blob = serializeSessionBlob(state);

  if (blob === syncState.lastPushedAccount) {
    return;
  }

  try {
    await setClientStateValue(SESSION_STATE_KEY, blob, syncState.owner.signal);

    assertOwner(syncState);
    syncState.lastPushedAccount = blob;
  } catch {
    assertOwner(syncState);
    syncState.hasPending = true;
  }
};

const loadFromBackend = async (
  syncState: SyncedPersistenceState,
  local: HydratedWorkbenchSnapshot | null,
  options?: WorkbenchLoadOptions
): Promise<HydratedWorkbenchSnapshot> => {
  assertOwner(syncState);
  const [summaries, sessionBlob] = await Promise.all([
    listProjects(syncState.owner.signal),
    fetchSessionBlob(syncState.owner.signal),
  ]);

  assertOwner(syncState);
  const persistedRevisions = loadPersistedRevisions(syncState);

  seedProjectLibrary(summaries, syncState.owner);

  // First contact: a backend with no projects adopts the browser's existing
  // workbench (one-time import of the pre-backend localStorage data).
  if (summaries.length === 0 && local && local.state.projects.length > 0) {
    for (const project of local.state.projects) {
      if (!(await pushNewProject(syncState, project))) {
        assertOwner(syncState);
        syncState.hasPending = true;
      }

      assertOwner(syncState);
      const entry = syncState.syncEntries.get(project.id);

      upsertProjectSummary({ id: project.id, name: project.name, revision: entry?.revision ?? null }, syncState.owner);
    }

    await pushSessionState(syncState, local.state);
    assertOwner(syncState);
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
  const records = await Promise.all(
    openIds.map((id) =>
      apiGetProject(id, syncState.owner.signal).catch(() => {
        assertOwner(syncState);

        return null;
      })
    )
  );

  assertOwner(syncState);
  const serverProjects: Project[] = [];

  for (const record of records) {
    if (!record) {
      continue;
    }

    const project = deserializeProjectRecord(record);

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
  await syncState.localPersistence.saveWorkbench(state);
  assertOwner(syncState);

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
export const clearAllWorkbenchData = async (owner: AccountScope = captureAccountScope()): Promise<void> => {
  const syncState = createSyncedPersistenceState(owner);

  assertOwner(syncState);

  try {
    const summaries = await listProjects(syncState.owner.signal);

    assertOwner(syncState);
    await Promise.all(summaries.map((summary) => apiDeleteProject(summary.project_id, syncState.owner.signal)));
    assertOwner(syncState);
    await deleteClientStateValue(SESSION_STATE_KEY, syncState.owner.signal);
    assertOwner(syncState);
  } catch {
    assertOwner(syncState);
    // Backend unreachable; at least reset this browser.
  }

  seedProjectLibrary([], owner);

  try {
    window.localStorage.removeItem(getSyncMapStorageKey(syncState));
  } catch {
    // Nothing to clear if storage is unavailable.
  }

  await syncState.localPersistence.clearWorkbench();
  assertOwner(syncState);
};

/** Construct one synchronization lifetime per mounted Workbench. */
export const createSyncedWorkbenchPersistence = (
  owner: AccountScope = captureAccountScope()
): SyncedWorkbenchPersistence => {
  const syncState = createSyncedPersistenceState(owner);
  let loadPromise: Promise<HydratedWorkbenchSnapshot | null> | null = null;
  // Every mutation below shares syncEntries and its optimistic revisions.
  // Serialize them so this browser cannot race itself and manufacture a 409.
  let mutationTail: Promise<void> = Promise.resolve();

  const enqueueMutation = <Result>(operation: () => Promise<Result>): Promise<Result> => {
    const run = (): Promise<Result> => {
      assertOwner(syncState);

      return operation();
    };
    const result = mutationTail.then(
      () => run(),
      () => run()
    );

    mutationTail = result.then(
      () => undefined,
      () => undefined
    );

    return result;
  };

  const adoptProjectRecord = (record: ProjectRecordDTO): Project | null => {
    assertOwner(syncState);
    const project = deserializeProjectRecord(record);

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
    clearWorkbench(): Promise<void> {
      return enqueueMutation(async () => {
        await clearAllWorkbenchData(syncState.owner);

        assertOwner(syncState);
        syncState.syncEntries.clear();
        syncState.deletedProjectIds.clear();
        syncState.lastPushedAccount = null;
        syncState.hasPending = false;
      });
    },
    flushProjectToServer(project): Promise<void> {
      return enqueueMutation(async () => {
        // A flush is a targeted push, not a save. Any conflict or fork it produces waits on
        // `syncState` for the next save to drain — the flush has no caller to hand them to, and
        // they are already true of the server by the time it returns.
        await pushProject(syncState, project);
        assertOwner(syncState);
        persistSyncMap(syncState);
      });
    },
    hasPendingChanges(): boolean {
      assertOwner(syncState);
      return syncState.hasPending;
    },
    async hydrateProjectFromServer(projectId): Promise<Project | null> {
      assertOwner(syncState);

      try {
        const record = await apiGetProject(projectId, syncState.owner.signal);

        assertOwner(syncState);
        return adoptProjectRecord(record);
      } catch {
        assertOwner(syncState);

        return null;
      }
    },
    /**
     * Load from the backend, falling back to the localStorage cache when it is
     * unreachable. Returns null when there is nothing anywhere (first run with
     * no backend); the caller then keeps its default boot state.
     */
    loadWorkbench(options?: WorkbenchLoadOptions): Promise<HydratedWorkbenchSnapshot | null> {
      // React StrictMode replays the Workbench mount effect. The synchronization
      // lifetime is stable across that replay, so both calls must observe the
      // same import instead of racing duplicate project POSTs.
      if (loadPromise) {
        return loadPromise;
      }

      loadPromise = (async () => {
        assertOwner(syncState);
        let local: HydratedWorkbenchSnapshot | null = null;

        try {
          local = await syncState.localPersistence.loadWorkbench();
          assertOwner(syncState);
        } catch {
          assertOwner(syncState);
          local = null;
        }

        try {
          return await loadFromBackend(syncState, local, options);
        } catch {
          assertOwner(syncState);
          // Backend unreachable: run from the cache; saves queue up locally and
          // replay on reconnect.
          syncState.hasPending = true;

          const persistedRevisions = loadPersistedRevisions(syncState);

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
          if (!local || local.state.projects.length === 0) {
            return null;
          }

          // `?new=true` means a fresh draft whether or not the backend answered.
          // Returning the cache verbatim here used to hand the caller whichever
          // project was last active, so an offline "New project" silently
          // reopened — and then let the Launchpad's intent rearrange — existing
          // work.
          if (options?.createNew) {
            const draft = createDraftProject(local.state.projects);

            return {
              ...local,
              state: {
                ...local.state,
                activeProjectId: draft.id,
                projects: [...local.state.projects, draft],
              },
            };
          }

          return local;
        }
      })();

      return loadPromise;
    },

    /**
     * Write-through save: localStorage cache always, then every dirty open
     * project and the session blob to the backend. Revision conflicts come
     * back as resolutions for the caller to apply to workbench state. Saving
     * never deletes anything: a project absent from state is merely closed,
     * and removal happens only through the library's explicit delete.
     */
    markProjectDeleted(projectId): void {
      assertOwner(syncState);
      syncState.deletedProjectIds.add(projectId);
      syncState.syncEntries.delete(projectId);
      persistSyncMap(syncState);
    },
    persistEmptySession(state): Promise<void> {
      return enqueueMutation(async () => {
        const emptied: WorkbenchState = { ...state, activeProjectId: '', projects: [] };

        await syncState.localPersistence.saveWorkbench(emptied);
        assertOwner(syncState);

        try {
          const blob = serializeSessionBlob(emptied);

          await setClientStateValue(SESSION_STATE_KEY, blob, syncState.owner.signal);
          assertOwner(syncState);
          syncState.lastPushedAccount = blob;
        } catch {
          assertOwner(syncState);
          syncState.hasPending = true;
        }
      });
    },
    releaseProjectSync(projectId): void {
      assertOwner(syncState);
      syncState.syncEntries.delete(projectId);
      persistSyncMap(syncState);
    },
    saveWorkbench(state: WorkbenchState): Promise<WorkbenchSaveResult> {
      return enqueueMutation(async () => {
        const snapshot = createSnapshot(state);

        await syncState.localPersistence.saveWorkbench(state);

        assertOwner(syncState);
        syncState.hasPending = false;

        const projectSyncInfos: Record<string, ProjectSyncInfo> = {};

        await pushSessionState(syncState, state);
        assertOwner(syncState);

        for (const project of state.projects) {
          assertOwner(syncState);
          const lastAckedDoc = syncState.syncEntries.get(project.id)?.pushedDoc ?? null;
          const documentJson = await pushProject(syncState, project);

          assertOwner(syncState);
          const entry = syncState.syncEntries.get(project.id);

          projectSyncInfos[project.id] = {
            isPendingPush: entry?.pushedDoc !== documentJson,
            revision: entry?.revision ?? null,
          };

          // The server acknowledged new content for this project — keep the
          // library summary current without a refetch.
          if (entry && entry.pushedDoc === documentJson && lastAckedDoc !== documentJson) {
            upsertProjectSummary({ id: project.id, name: project.name, revision: entry.revision }, syncState.owner);
          }
        }

        persistSyncMap(syncState);
        reportProjectSync({ hasPendingChanges: syncState.hasPending, projects: projectSyncInfos });

        // Drained rather than read: each outcome is applied to the store exactly once. The runtime
        // applies what it is handed even when the save it came from went stale, so draining here is
        // safe — nothing is dropped between this call and the reducer.
        return {
          conflicts: syncState.pendingConflicts.splice(0),
          deletedProjectForks: syncState.pendingDeletedForks.splice(0),
          hasPendingChanges: syncState.hasPending,
          projectBoardAssignments: syncState.pendingBoardAssignments.splice(0),
          snapshot,
        };
      });
    },
    unmarkProjectDeleted(projectId): void {
      assertOwner(syncState);
      syncState.deletedProjectIds.delete(projectId);
    },
  };
};
