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

import type { ProjectPushOutcome, ProjectRecoveredIdentity } from './projectFlush';

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
 * The server is the source of truth: one revision-versioned document per project, plus a session
 * blob in the per-user client-state KV. The localStorage snapshot is a write-through cache, so the
 * workbench still loads and autosaves offline and replays on reconnect.
 *
 * Workbench state holds only the open projects; the rest live in the library as summaries. Saving
 * never deletes — projects leave the server only through the library's explicit delete.
 *
 * Conflicts never lose work: the server version keeps the id, and the local version forks into a
 * "(recovered)" project beside it.
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
  recoveredIdentity: ProjectRecoveredIdentity;
}

/** Local edits rescued from a project that was deleted on another device. */
export interface ProjectDeletionFork {
  projectId: string;
  /** The fork carrying the local edits, under a fresh id and its own board. */
  recoveredProject: Project;
  recoveredIdentity: ProjectRecoveredIdentity;
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
   * Ids already forked because the server no longer had them. The original stays in the aggregate
   * until reconciliation reaches it, and in that window `pushProject` would find no sync entry and
   * re-create it under its old id — undoing the deletion the fork exists to respect.
   */
  forkedProjectIds: Set<string>;
  hasPending: boolean;
  localPersistence: WorkbenchPersistenceService;
  lastPushedAccount: string | null;
  /** Immutable owner captured when this synchronization lifetime was constructed. */
  owner: AccountScope;
  /**
   * What the server said that the aggregate has not been told yet, drained by the next save. A push
   * happens from a save *and* from a targeted flush, and only the save has somewhere to return an
   * outcome to — so a flush that forks a project cannot silently drop the answer.
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
 * Rehydrate a *server record*, which knows the project's real board. The document's own
 * `projectBoardId` is a stale-able cache, so overwriting it here means every path that reads from
 * the server agrees on one answer. The saved destination is left alone — it is a deliberate choice.
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
 * The revision map survives reloads so an offline runtime can tell "synced before, now gone from
 * the server — drop it" apart from "created offline — push it".
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

/** Lineage points at the root original, so a recovery of a recovery still keys to the first. */
export const createRecoveredDocument = (
  project: Project,
  document: Record<string, unknown>
): { recoveredIdentity: ProjectRecoveredIdentity; recoveredDocument: Record<string, unknown> } => {
  const recoveryOf = project.recoveryOf ?? project.id;
  const recoveredIdentity: ProjectRecoveredIdentity = {
    id: `${recoveryOf}-recovered-${Date.now().toString(36)}`,
    name: `${getRecoveryBaseName(project.name)} (recovered)`,
    recoveredAt: new Date().toISOString(),
    recoveryOf,
  };

  return {
    recoveredDocument: {
      ...document,
      id: recoveredIdentity.id,
      name: recoveredIdentity.name,
      recoveredAt: recoveredIdentity.recoveredAt,
      recoveryOf: recoveredIdentity.recoveryOf,
    },
    recoveredIdentity,
  };
};

type ConflictOutcome =
  | { kind: 'adopted' }
  | { kind: 'retry' }
  | { kind: 'forked'; resolution: ProjectConflictResolution }
  | { kind: 'failed' };

/**
 * A save lost the revision race. Forking is the last resort — only when content actually diverged:
 *
 * - server content == what we pushed → adopt the revision, done
 * - server content == this edit's base → revisions drifted without divergence; adopt and retry
 * - anything else → the server version keeps the id, the local edits fork into "(recovered)"
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

    const { recoveredDocument, recoveredIdentity } = createRecoveredDocument(project, document);
    const recoveredProject = deserializeProjectDocument(recoveredDocument);

    if (!recoveredProject) {
      return { kind: 'failed' };
    }

    const created = await apiCreateProject(
      { data: recoveredDocument, name: recoveredIdentity.name, project_id: recoveredIdentity.id },
      syncState.owner.signal
    );

    assertOwner(syncState);
    syncState.syncEntries.set(recoveredIdentity.id, {
      pushedDoc: JSON.stringify(recoveredDocument),
      revision: created.revision,
    });
    // Recorded for the same reason the deletion fork records it: the fork is a project the server
    // created, so its board id exists nowhere else. Without this the fork opens with a gallery bound
    // to the original's board — which the conflict left in the server version's hands.
    syncState.pendingBoardAssignments.push({ boardId: created.board_id, projectId: recoveredIdentity.id });

    return {
      kind: 'forked',
      resolution: { projectId: project.id, recoveredIdentity, recoveredProject, serverProject },
    };
  } catch {
    assertOwner(syncState);

    return { kind: 'failed' };
  }
};

type DeletionForkOutcome =
  | { kind: 'forked'; fork: ProjectDeletionFork }
  /** The 404 was this browser's own deletion. There is nothing to rescue and nothing went wrong. */
  | { kind: 'abandoned' }
  | { kind: 'failed' };

/**
 * Rescue the local edits of a project the server no longer has. A *fork*, not a re-create: pushing
 * the original id back would resurrect a project the user deleted, on every device. The fork gets a
 * fresh id, and with it a fresh board from the create response.
 */
const forkDeletedProject = async (
  syncState: SyncedPersistenceState,
  project: Project,
  document: Record<string, unknown>
): Promise<DeletionForkOutcome> => {
  assertOwner(syncState);

  const { recoveredDocument, recoveredIdentity } = createRecoveredDocument(project, document);
  const recoveredProject = deserializeProjectDocument(recoveredDocument);

  if (!recoveredProject) {
    return { kind: 'failed' };
  }

  try {
    const created = await apiCreateProject(
      { data: recoveredDocument, name: recoveredIdentity.name, project_id: recoveredIdentity.id },
      syncState.owner.signal
    );

    assertOwner(syncState);

    // Re-read after the create, not just before it. `deleteLibraryProject` marks the id before
    // issuing its DELETE, but a PUT already on the wire is past that check — the 404 that brought us
    // here can be this browser's own deletion arriving first. Forking on it would resurrect the
    // project the person just deleted, as a copy pointing at media the deletion already removed.
    if (syncState.deletedProjectIds.has(project.id)) {
      try {
        await apiDeleteProject(recoveredIdentity.id, syncState.owner.signal);
      } catch {
        // The fork is an empty private project either way; failing to remove it is clutter, not a
        // broken state, and it must not replace the deletion's own outcome.
      }

      assertOwner(syncState);
      syncState.syncEntries.delete(recoveredIdentity.id);

      return { kind: 'abandoned' };
    }

    syncState.syncEntries.set(recoveredIdentity.id, {
      pushedDoc: JSON.stringify(recoveredDocument),
      revision: created.revision,
    });
    syncState.pendingBoardAssignments.push({ boardId: created.board_id, projectId: recoveredIdentity.id });

    return { fork: { projectId: project.id, recoveredIdentity, recoveredProject }, kind: 'forked' };
  } catch {
    assertOwner(syncState);

    return { kind: 'failed' };
  }
};

const pushProject = async (syncState: SyncedPersistenceState, project: Project): Promise<ProjectPushOutcome> => {
  assertOwner(syncState);
  const { document, json: documentJson } = getSerializedProjectDocument(syncState, project);
  const entry = syncState.syncEntries.get(project.id);

  // A deleted or already-forked id holds someone else's answer — the deletion, or the server
  // version that won the race. Nothing is pushed, and nothing read back under this id would be ours.
  if (syncState.deletedProjectIds.has(project.id) || syncState.forkedProjectIds.has(project.id)) {
    return { documentJson, kind: 'superseded' };
  }

  if (entry?.pushedDoc === documentJson) {
    return { documentJson, kind: 'acknowledged' };
  }

  if (!entry) {
    if (!(await pushNewProject(syncState, project))) {
      assertOwner(syncState);
      syncState.hasPending = true;

      return { documentJson, kind: 'unsynced' };
    }

    assertOwner(syncState);
    return { documentJson, kind: 'acknowledged' };
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

          return { documentJson, kind: 'unsynced' };
        }
      } else if (outcome.kind === 'forked') {
        syncState.pendingConflicts.push(outcome.resolution);

        // This id now holds the server's version, not ours.
        return { documentJson, kind: 'superseded' };
      } else if (outcome.kind === 'failed') {
        syncState.hasPending = true;

        return { documentJson, kind: 'unsynced' };
      }
      // 'adopted': the server already held these exact bytes, so the push had nothing to do.
    } else if (isProjectNotFoundError(error)) {
      // Deleted on another device while we held local edits. Re-creating under the same id would
      // undo that deletion — the project would reappear on every device, and on this one it would
      // be a project whose board the deletion already took. Fork instead: the deletion stands, and
      // the local work survives as a recovered project with its own id and its own board.
      syncState.syncEntries.delete(project.id);

      // Unless the deletion was ours. `markDeleted` runs before the DELETE, but a PUT already on the
      // wire is past the check at the top of this function, so a fast DELETE can turn our own push
      // into a 404 — and forking on that resurrects, server-side, exactly what the person deleted.
      if (syncState.deletedProjectIds.has(project.id)) {
        return { documentJson, kind: 'superseded' };
      }

      const outcome = await forkDeletedProject(syncState, project, document);

      assertOwner(syncState);
      if (outcome.kind === 'failed') {
        syncState.hasPending = true;

        return { documentJson, kind: 'unsynced' };
      }

      if (outcome.kind === 'forked') {
        // Recorded before the aggregate hears about it, so nothing re-creates the original id in
        // the meantime — see `forkedProjectIds`.
        syncState.forkedProjectIds.add(project.id);
        syncState.pendingDeletedForks.push(outcome.fork);
      }

      return { documentJson, kind: 'superseded' };
    } else {
      syncState.hasPending = true;

      return { documentJson, kind: 'unsynced' };
    }
  }

  assertOwner(syncState);
  return { documentJson, kind: 'acknowledged' };
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
  const account = sessionBlob?.account ?? base.account;
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
    const draft = createDraftProject(projects, account);

    projects = [...projects, draft];
    activeProjectId = draft.id;
  }

  const state: WorkbenchState = {
    ...base,
    account,
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
  deleteProjectOnServer(projectId: string): Promise<void>;
  flushProjectToServer(project: Project): Promise<ProjectPushOutcome>;
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

  /**
   * Sync entries dropped by {@link markDeleted}, kept so a deletion that fails can be undone whole.
   *
   * Without this, unmarking restores the project's right to save but not its place in the revision
   * chain: the next push finds no entry, takes the create path, and has to recover through a 409.
   */
  const entriesHeldForDeletion = new Map<string, SyncEntry>();

  const markDeleted = (projectId: string): void => {
    assertOwner(syncState);
    syncState.deletedProjectIds.add(projectId);

    const entry = syncState.syncEntries.get(projectId);

    if (entry) {
      entriesHeldForDeletion.set(projectId, entry);
    }

    syncState.syncEntries.delete(projectId);
    persistSyncMap(syncState);
  };

  const unmarkDeleted = (projectId: string): void => {
    assertOwner(syncState);
    syncState.deletedProjectIds.delete(projectId);

    const entry = entriesHeldForDeletion.get(projectId);

    if (entry) {
      syncState.syncEntries.set(projectId, entry);
      entriesHeldForDeletion.delete(projectId);
      persistSyncMap(syncState);
    }
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
        entriesHeldForDeletion.clear();
        syncState.lastPushedAccount = null;
        syncState.hasPending = false;
      });
    },
    /** Queued, not issued directly — see {@link OpenProjectHandle.deleteOnServer}. */
    deleteProjectOnServer(projectId): Promise<void> {
      return enqueueMutation(async () => {
        markDeleted(projectId);

        try {
          await apiDeleteProject(projectId, syncState.owner.signal);
        } catch (error) {
          assertOwner(syncState);
          unmarkDeleted(projectId);
          throw error;
        }

        assertOwner(syncState);
      });
    },
    flushProjectToServer(project): Promise<ProjectPushOutcome> {
      return enqueueMutation(async () => {
        // A flush is a targeted push, not a save. Any conflict or fork it produces waits on
        // `syncState` for the next save to drain — the flush has no caller to hand them to, and
        // they are already true of the server by the time it returns.
        //
        // What it *does* hand back is whether the push landed. Every caller here has a recoverable
        // failure on its hands, so this still does not reject; but "recoverable" and "done" are
        // different answers, and a caller about to read the project back from the server needs the
        // second one. `assertProjectFlushed` in `./projectFlush` is where that is spent.
        const outcome = await pushProject(syncState, project);

        assertOwner(syncState);
        persistSyncMap(syncState);

        return outcome;
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

          if (!local) {
            return null;
          }

          // A cache holding an empty session (last tab closed offline) still
          // owns the account's preset defaults, so build the replacement draft
          // here instead of falling back to the store's shipped defaults.
          if (local.state.projects.length === 0) {
            const draft = createDraftProject([], local.state.account);

            return {
              ...local,
              state: { ...local.state, activeProjectId: draft.id, projects: [draft] },
            };
          }

          // `?new=true` means a fresh draft whether or not the backend answered.
          // Returning the cache verbatim here used to hand the caller whichever
          // project was last active, so an offline "New project" silently
          // reopened — and then let the Launchpad's intent rearrange — existing
          // work.
          if (options?.createNew) {
            const draft = createDraftProject(local.state.projects, local.state.account);

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
      markDeleted(projectId);
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
          const { documentJson } = await pushProject(syncState, project);

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
      unmarkDeleted(projectId);
    },
  };
};
