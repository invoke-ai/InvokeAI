import type { HydratedWorkbenchSnapshot } from '@workbench/persistenceContracts';
import type { Project, ProjectLoadResult, RefusedWorkbenchProject, WorkbenchState } from '@workbench/projectContracts';

import { createUuid } from '@platform/browser/randomUuid';
import { createLogger } from '@platform/logging/logger';
import { assertAccountScopeCurrent, isAccountScopeCurrent, type AccountScope } from '@platform/state/accountLifecycle';
import {
  getProjectCanvasSchemaRequirement,
  isCanvasSchemaVersionSupported,
  MAX_SUPPORTED_CANVAS_SCHEMA_VERSION,
} from '@workbench/canvasSchemaVersion';
import { hasActiveQueueRuns } from '@workbench/queue-integration/activeQueueRuns';
import { createAccountOwnedQueueRunJournal } from '@workbench/queue-integration/queueRunJournal';
import {
  createDraftProject,
  createInitialWorkbenchState,
  normalizeWorkbenchAccount,
  withAuthoritativeProjectBoard,
} from '@workbench/workbenchState';

import type { ProjectCreateRequest, ProjectRecordDTO, ProjectSummaryDTO, ProjectUpdateRequest } from './api';
import type {
  ProjectDraft,
  ProjectDraftCopyReservation,
  ProjectDraftRetargetCursor,
  ProjectDraftRetargetHandoff,
  ProjectDraftStore,
  ProjectDraftSummary,
  ProjectUnloadJournalEntry,
  ProjectUnloadJournalKey,
  ProjectUnloadJournalReconciliation,
} from './draftStore';
import type { EditorSession } from './editorSession';
import type { ProjectPushOutcome, ProjectSchemaRefusal } from './projectFlush';
import type { WorkbenchSessionBlob } from './session';
import type { DeleteWorkbenchDatabaseFinalResult, DeleteWorkbenchDatabaseResult } from './workbenchDatabase';

import {
  createProjectSettled,
  deleteClientStateValue,
  deleteProject,
  getProject,
  getProjectCanvasSchemaCompatibilityRefusal,
  getProjectWriteSizeRefusal,
  isProjectConflictError,
  isProjectNotFoundError,
  listProjects,
  ProjectCreateAbsentError,
  setClientStateValue,
  updateProject,
} from './api';
import { recordProjectCover } from './covers';
import {
  getUtf8ByteSize,
  getCopySourceProjectName,
  PROJECT_DRAFT_PROJECT_LIMIT,
  PROJECT_UNLOAD_JOURNAL_MAX_BYTES,
  PROJECT_UNLOAD_JOURNAL_SCHEMA_VERSION,
  toConflictProjectDraft,
  toDirtyProjectDraft,
  toSchemaRefusedProjectDraft,
} from './draftStore';
import { getEditorSession, isEditorSessionLive } from './editorSession';
import { createDeterministicProjectId } from './ids';
import { createAccountOwnedProjectDraftStore } from './indexedDbDraftStore';
import { seedProjectLibrary, upsertProjectSummary } from './library';
import { selectCoverImageName } from './projectAssets';
import {
  migrateProjectDocument,
  PROJECT_DOCUMENT_MAX_BYTES,
  PROJECT_DOCUMENT_SCHEMA_VERSION,
  serializeProjectDocumentV3Json,
} from './projectDocument';
import { assertProjectFlushed, ProjectFlushError } from './projectFlush';
import { deserializeProjectDocument, deserializeProjectRecord } from './projectHydration';
import { acquireProjectMutationLock } from './projectLifecycleLocks';
import { fetchSessionBlobStrict, serializeSessionBlob, SESSION_STATE_KEY } from './session';
import { getOpenProject, reportProjectSync, reportProjectSyncEntry, resolveProjectSyncConflict } from './syncStore';
import { deleteWorkbenchDatabase } from './workbenchDatabase';

export interface ProjectConflictInfo {
  detectedAt: string;
  kind: 'deleted' | 'revision';
  projectId: string;
  serverRevision?: number;
}

export type LocalDraftStatus = 'ok' | 'unavailable';

export interface ProjectBoardAssignment {
  boardId: string;
  projectId: string;
}

export interface RecoverableProjectDraft {
  documentByteSize: number;
  editorSessionId: string;
  generation: number;
  projectId: string;
  state: ProjectDraft['state'];
  updatedAt: number;
}

export type RecoverableProjectDraftPage =
  | { items: RecoverableProjectDraft[]; kind: 'available'; nextCursor: [string, string] | null }
  | { kind: 'unavailable' };

export interface SaveDraftAsNewInput {
  copyProjectId: string;
  document: Record<string, unknown>;
  minimumCanvasSchemaVersion: number;
  name: string;
  owner: AccountScope;
  sourceProjectId: string;
}

export type SaveDraftAsNew = (input: SaveDraftAsNewInput) => Promise<ProjectRecordDTO>;

/** How long a failed upload-provenance creation answers later uploads before another attempt is made. */
const PROJECT_ENSURE_FAILURE_REUSE_MS = 5_000;

const saveDraftAsNew: SaveDraftAsNew = (input) => {
  return createProjectSettled(
    {
      data: input.document,
      minimum_canvas_schema_version: input.minimumCanvasSchemaVersion,
      name: input.name,
      project_id: input.copyProjectId,
    },
    input.owner
  );
};

const mapConcurrent = async <T, R>(
  items: readonly T[],
  limit: number,
  visit: (item: T, index: number) => Promise<R>
): Promise<R[]> => {
  let nextIndex = 0;
  let firstError: unknown = null;
  const results: R[] = [];
  results.length = items.length;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    for (;;) {
      if (firstError) {
        return;
      }
      const index = nextIndex;
      nextIndex += 1;
      if (index >= items.length) {
        return;
      }
      try {
        results[index] = await visit(items[index]!, index);
      } catch (error) {
        firstError ??= error;
        return;
      }
    }
  });
  await Promise.all(workers);
  if (firstError) {
    throw firstError instanceof Error ? firstError : new Error('A project could not be loaded.');
  }
  return results;
};

const forEachConcurrent = async <T>(items: readonly T[], limit: number, visit: (item: T) => Promise<void>) => {
  await mapConcurrent(items, limit, visit);
};

const awaitDatabaseDeletion = async (
  deletion: DeleteWorkbenchDatabaseResult,
  timeoutMs: number
): Promise<DeleteWorkbenchDatabaseFinalResult> => {
  if (deletion.kind !== 'blocked') {
    return deletion;
  }
  let timeout: ReturnType<typeof globalThis.setTimeout> | undefined;
  try {
    return await Promise.race([
      deletion.completion,
      new Promise<{ kind: 'unavailable' }>((resolve) => {
        timeout = globalThis.setTimeout(() => resolve({ kind: 'unavailable' }), timeoutMs);
      }),
    ]);
  } finally {
    if (timeout !== undefined) {
      globalThis.clearTimeout(timeout);
    }
  }
};

export interface DurableWorkbenchSaveResult {
  conflicts: ProjectConflictInfo[];
  error: string | null;
  hasPendingChanges: boolean;
  localDraftStatus: LocalDraftStatus;
  projectBoardAssignments: ProjectBoardAssignment[];
  shouldRetry: boolean;
  snapshot: HydratedWorkbenchSnapshot;
}

export type UnloadJournalOutcome =
  | {
      kind: 'journaled';
      projectIds: string[];
      skippedOversizedProjectIds: string[];
      /** Settles true once the write commits, false if it aborted. */
      written: Promise<boolean>;
    }
  | { kind: 'nothing-to-journal' | 'unavailable' };

export interface QueueRecoveryProject {
  projectId: string;
  reason: 'open-limit' | 'project-unavailable';
}

export interface QueueRecoveryState {
  projects: QueueRecoveryProject[];
  status: 'available' | 'unavailable';
}

export interface DurableHydratedWorkbenchSnapshot extends HydratedWorkbenchSnapshot {
  conflicts: ProjectConflictInfo[];
  localDraftStatus: LocalDraftStatus;
  queueRecovery: QueueRecoveryState;
}

export interface DurableProjectPersistenceApi {
  createProject(request: ProjectCreateRequest, owner: AccountScope): Promise<ProjectRecordDTO>;
  deleteProject(projectId: string, signal?: AbortSignal): Promise<void>;
  deleteSession(signal?: AbortSignal): Promise<void>;
  getProject(projectId: string, signal?: AbortSignal): Promise<ProjectRecordDTO>;
  listProjects(signal?: AbortSignal): Promise<ProjectSummaryDTO[]>;
  loadSession(signal?: AbortSignal): Promise<WorkbenchSessionBlob | null>;
  saveSession(
    state: WorkbenchState,
    editorSessionId: string,
    draftEditorSessionIds: Record<string, string>,
    signal?: AbortSignal
  ): Promise<void>;
  updateProject(projectId: string, request: ProjectUpdateRequest, signal?: AbortSignal): Promise<ProjectRecordDTO>;
}

const productionApi: DurableProjectPersistenceApi = {
  createProject: createProjectSettled,
  deleteProject,
  deleteSession: (signal) => deleteClientStateValue(SESSION_STATE_KEY, signal),
  getProject,
  listProjects,
  loadSession: fetchSessionBlobStrict,
  saveSession: (state, editorSessionId, draftEditorSessionIds, signal) =>
    setClientStateValue(SESSION_STATE_KEY, serializeSessionBlob(state, editorSessionId, draftEditorSessionIds), signal),
  updateProject,
};

interface DurableSyncedPersistenceDependencies {
  api?: DurableProjectPersistenceApi;
  autoOpenDraftByteLimit?: number;
  autoOpenDraftLimit?: number;
  autoOpenQueueRunProjectLimit?: number;
  clearLegacyStorage?: () => void;
  databaseDeleteTimeoutMs?: number;
  deleteDatabase?: typeof deleteWorkbenchDatabase;
  draftStore?: Promise<ProjectDraftStore>;
  editorSession?: Promise<EditorSession>;
  /** Whether another page still holds the editor session, so its unload journal is left for it to settle. */
  isEditorSessionLive?: (editorSessionId: string) => Promise<boolean>;
  now?: () => string;
  projectMutationLock?: (projectId: string) => ReturnType<typeof acquireProjectMutationLock>;
  queueRunJournal?: typeof createAccountOwnedQueueRunJournal;
  saveDraftAsNew?: SaveDraftAsNew;
  writerToken?: string;
}

interface SyncEntry {
  minimumCanvasSchemaVersion: number;
  pushedDoc: string;
  revision: number;
}

interface LoadedDraft {
  draft: ProjectDraft;
  project: Project;
}

const isStatus = (error: unknown, status: number): boolean =>
  (error instanceof Error && 'status' in error && error.status === status) ||
  (status === 409 ? isProjectConflictError(error) : status === 404 ? isProjectNotFoundError(error) : false);

const isDeterministicClientError = (error: unknown): boolean => {
  if (!error || typeof error !== 'object' || !('status' in error) || typeof error.status !== 'number') {
    return false;
  }
  return error.status >= 400 && error.status < 500 && ![408, 409, 429].includes(error.status);
};

const toSnapshot = (state: WorkbenchState, savedAt: string): HydratedWorkbenchSnapshot => ({
  refusedProjects: [],
  savedAt,
  state,
  version: 1,
});

const recordDocumentJson = (record: ProjectRecordDTO): string | null => {
  const loaded = deserializeProjectRecord(record);

  return loaded.status === 'loaded' ? serializeProjectDocumentV3Json(loaded.project).documentJson : null;
};

const toProjectSummary = (record: ProjectRecordDTO): ProjectSummaryDTO => ({
  board_id: record.board_id,
  created_at: record.created_at,
  minimum_canvas_schema_version: record.minimum_canvas_schema_version,
  name: record.name,
  project_id: record.project_id,
  revision: record.revision,
  updated_at: record.updated_at,
});

const recordRequiresDocumentUpgrade = (record: ProjectRecordDTO): boolean =>
  record.data.documentSchemaVersion !== PROJECT_DOCUMENT_SCHEMA_VERSION;

const toCanvasSchemaRefusal = (refusal: {
  maxCanvasSchemaVersion: number;
  minimumCanvasSchemaVersion: number;
}): ProjectSchemaRefusal => ({ ...refusal, kind: 'canvas' });

const toDocumentSchemaRefusal = (documentSchemaVersion: number): ProjectSchemaRefusal => ({
  documentSchemaVersion,
  kind: 'document',
  maxDocumentSchemaVersion: PROJECT_DOCUMENT_SCHEMA_VERSION,
});

const loadDraftProject = (draft: ProjectDraft): Project | null => {
  try {
    const raw = JSON.parse(draft.documentJson) as unknown;
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      return null;
    }
    const loaded = deserializeProjectDocument(raw as Record<string, unknown>);
    return loaded.status === 'loaded' ? loaded.project : null;
  } catch {
    return null;
  }
};

const getRaisedCanvasSchemaFloor = (
  document: Record<string, unknown>,
  retainedMinimumCanvasSchemaVersion: number
): number | undefined => {
  const required = getProjectCanvasSchemaRequirement(document);
  return required > retainedMinimumCanvasSchemaVersion ? required : undefined;
};

export class WorkbenchBackendUnavailableError extends Error {
  constructor(cause?: unknown) {
    super('The project backend is unavailable.', { cause });
    this.name = 'WorkbenchBackendUnavailableError';
  }
}

export class ProjectDocumentTooLargeError extends Error {
  readonly actualBytes: number;
  readonly maxBytes: number;

  constructor(actualBytes: number, maxBytes = PROJECT_DOCUMENT_MAX_BYTES) {
    super(`The project document is ${actualBytes} bytes; the maximum is ${maxBytes} bytes.`);
    this.name = 'ProjectDocumentTooLargeError';
    this.actualBytes = actualBytes;
    this.maxBytes = maxBytes;
  }
}

export class ProjectDraftWriteRejectedError extends Error {
  constructor(readonly reason: 'corrupt' | 'fenced' | 'generation-conflict' | 'stale') {
    super(`The local project draft write was rejected (${reason}).`);
    this.name = 'ProjectDraftWriteRejectedError';
  }
}

export interface DurableSyncedWorkbenchPersistence {
  acknowledgeProjectResolution(projectId: string): void;
  abortProjectResolution(projectId: string): void;
  adoptProjectRecord(record: ProjectRecordDTO): ProjectLoadResult;
  clearWorkbench(): Promise<void>;
  close(): void;
  deleteRecoverableDraft(
    projectId: string,
    editorSessionId: string,
    generation: number,
    updatedAt: number
  ): Promise<void>;
  deleteProjectOnServer(projectId: string): Promise<void>;
  ensureProjectOnServer(project: Project): Promise<void>;
  flushProjectToServer(project: Project): Promise<ProjectPushOutcome>;
  getProjectDraftDocument(projectId: string): Promise<string | null>;
  getRecoverableDraftDocument(projectId: string, editorSessionId: string): Promise<string | null>;
  /** True from `persistEmptySession` until `reopenSession`: the editor is leaving and its session must stay empty. */
  hasClosedSession(): boolean;
  /**
   * Synchronously journals every open project whose document is not yet in local recovery, for a page that may unload
   * before an asynchronous save can stage it. The next load reconciles the journal under the draft store's rules.
   */
  journalBeforeUnload(state: WorkbenchState): UnloadJournalOutcome;
  hasPendingChanges(): boolean;
  hydrateProjectFromServer(projectId: string, projectName?: string): Promise<ProjectLoadResult>;
  loadWorkbench(options?: { createNew?: boolean; openProjectId?: string }): Promise<DurableHydratedWorkbenchSnapshot>;
  listRecoverableDrafts(options?: { after?: [string, string]; limit?: number }): Promise<RecoverableProjectDraftPage>;
  markProjectDeleted(projectId: string): void;
  persistEmptySession(state: WorkbenchState): Promise<void>;
  /** Releases cross-tab project locks held for unfinished resolutions once a newer editor in this tab takes over. */
  releaseMutationLocks(): void;
  /** Undoes `persistEmptySession` when the editor stays open after all, writing its session again. */
  reopenSession(state: WorkbenchState): Promise<DurableWorkbenchSaveResult>;
  /** Called after `reopenSession` has reopened the session, so saves held back while it was closed can resume. */
  subscribeSessionReopened(listener: () => void): () => void;
  resolveConflictDiscard(projectId: string): Promise<void>;
  resolveConflictSaveAsNew(project: Project): Promise<{
    boardId: string;
    name: string;
    project: Project;
    sourceName: string;
    sourceProjectId: string;
    targetProjectId: string;
  }>;
  resolveConflictUseServer(projectId: string): Promise<ProjectLoadResult>;
  releaseProjectSync(projectId: string): void;
  /** The release resolves once this service has either been retained again or released all of its resources. */
  retain(): () => Promise<void>;
  saveWorkbench(state: WorkbenchState): Promise<DurableWorkbenchSaveResult>;
  unmarkProjectDeleted(projectId: string): void;
}

export const createDurableSyncedWorkbenchPersistence = (
  owner: AccountScope,
  dependencies: DurableSyncedPersistenceDependencies = {}
): DurableSyncedWorkbenchPersistence => {
  const api = dependencies.api ?? productionApi;
  const logger = createLogger({ area: 'autosave', namespace: 'persistence' }, { owner });
  const autoOpenDraftByteLimit = dependencies.autoOpenDraftByteLimit ?? 64 * 1024 * 1024;
  const autoOpenDraftLimit = dependencies.autoOpenDraftLimit ?? 8;
  const requestedQueueRunProjectLimit = dependencies.autoOpenQueueRunProjectLimit ?? 8;
  const autoOpenQueueRunProjectLimit =
    Number.isSafeInteger(requestedQueueRunProjectLimit) && requestedQueueRunProjectLimit >= 0
      ? requestedQueueRunProjectLimit
      : 8;
  const clearLegacyStorage =
    dependencies.clearLegacyStorage ??
    (() => {
      try {
        const base = `invokeai:v7:webv2:workbench${owner.storageSuffix}`;
        window.localStorage.removeItem(base);
        window.localStorage.removeItem(`invokeai:v7:webv2:workbench-sync${owner.storageSuffix}`);
        window.localStorage.removeItem(`${base}:refused-projects`);
      } catch {
        return;
      }
    });
  let draftStorePromise: Promise<ProjectDraftStore> | null = null;
  let editorSessionPromise: Promise<EditorSession> | null = null;
  /** The resolved store and editor session, for the synchronous unload journal; set once loading has them. */
  let unloadJournalTarget: { editorSessionId: string; store: ProjectDraftStore } | null = null;
  const getDraftStore = () =>
    (draftStorePromise ??= dependencies.draftStore ?? createAccountOwnedProjectDraftStore(owner));
  const getEditorSessionForService = () => (editorSessionPromise ??= dependencies.editorSession ?? getEditorSession());
  const isOtherEditorSessionLive = dependencies.isEditorSessionLive ?? isEditorSessionLive;
  const now = dependencies.now ?? (() => new Date().toISOString());
  const getProjectMutationLock =
    dependencies.projectMutationLock ??
    ((projectId: string) => acquireProjectMutationLock(owner.storageSuffix, projectId));
  const openQueueRunJournal = dependencies.queueRunJournal ?? createAccountOwnedQueueRunJournal;
  const listQueueRunProjectIds = async (): Promise<
    { kind: 'available'; projectIds: string[] } | { kind: 'unavailable' }
  > => {
    let journal: Awaited<ReturnType<typeof openQueueRunJournal>>;
    try {
      journal = await openQueueRunJournal(owner);
    } catch {
      return { kind: 'unavailable' };
    }
    try {
      return await journal.listProjectIds();
    } catch {
      return { kind: 'unavailable' };
    } finally {
      journal.close();
    }
  };
  const assertNoDurableQueueRuns = async (projectId: string): Promise<void> => {
    const journal = await openQueueRunJournal(owner);
    try {
      const result = await journal.listForProject(projectId);
      if (result.kind === 'unavailable') {
        throw new Error('Active queue runs could not be verified.');
      }
      if (result.entries.length > 0) {
        throw new Error('Wait for active queue runs to finish before resolving this project.');
      }
    } finally {
      journal.close();
    }
  };
  const writerToken = dependencies.writerToken ?? createUuid();
  const saveAsNew = dependencies.saveDraftAsNew ?? saveDraftAsNew;
  const deleteDatabase = dependencies.deleteDatabase ?? deleteWorkbenchDatabase;
  const databaseDeleteTimeoutMs = dependencies.databaseDeleteTimeoutMs ?? 5_000;
  const syncEntries = new Map<string, SyncEntry>();
  const projectEnsures = new Map<string, Promise<void>>();
  const projectEnsureFailures = new Map<string, { at: number; error: unknown }>();
  const conflicts = new Map<string, ProjectConflictInfo>();
  const schemaRefusals = new Map<string, ProjectSchemaRefusal>();
  const generations = new Map<string, number>();
  /**
   * The highest generation this writer has used per project, never reset: the claim's unload journal mark is compared
   * with this writer's generations, so they must not restart when a project is closed and reopened.
   */
  const issuedGenerations = new Map<string, number>();
  const noteGeneration = (projectId: string, generation: number): void => {
    generations.set(projectId, generation);
    issuedGenerations.set(projectId, Math.max(issuedGenerations.get(projectId) ?? 0, generation));
  };
  const nextGeneration = (projectId: string): number =>
    Math.max(generations.get(projectId) ?? 0, issuedGenerations.get(projectId) ?? 0) + 1;
  const deletedProjectIds = new Set<string>();
  const serverDeletedProjectIds = new Set<string>();
  const pendingProjectIds = new Set<string>();
  const projectsRequiringDocumentUpgrade = new Set<string>();
  const pendingBoardAssignments = new Map<string, ProjectBoardAssignment>();
  const retargetedProjects = new Map<
    string,
    { boardId: string; name: string; projectId: string; sourceName: string }
  >();
  const volatileDrafts = new Map<string, ProjectDraft>();
  const unopenableDrafts = new Map<string, RecoverableProjectDraft>();
  const pendingRetargetHandoffs = new Map<string, ProjectDraftRetargetHandoff>();
  const explicitRetargetSourcesAwaitingAck = new Set<string>();
  const closedRetargetTargetsAwaitingAck = new Set<string>();
  const draftEditorSessionIds = new Map<string, string>();
  const copyCaptureProjectIds = new Set<string>();
  const projectMutationLocks = new Map<string, { release(): Promise<void> }>();
  const reservedCopyTargetIds = new Set<string>();
  const projectResolutionFences = new Map<
    string,
    { kind: 'pending' | 'remove' } | { kind: 'replace'; project: Project }
  >();
  let isClosed = false;
  let isTerminallyCleared = false;
  let retainCount = 0;
  let releaseGeneration = 0;
  let lastSessionJson: string | null = null;
  let lastKnownState: WorkbenchState | null = null;
  let hasPending = false;
  let sessionSavePending = false;
  let isSessionClosed = false;
  const sessionReopenedListeners = new Set<() => void>();
  let localDraftStatus: LocalDraftStatus = 'ok';
  const localDraftFailures = new Set<string>();
  /** Documents this writer journaled and has not yet seen acknowledged, by resolved project id. */
  const journaledDocuments = new Map<string, ProjectUnloadJournalEntry>();
  /**
   * Projects whose journal record no longer stands for a stored entry, so its document is journaled again: retired at
   * a hidden event (the record stays, so the next save writes the settled mark), or being settled right now.
   */
  const retiredJournalProjectIds = new Set<string>();
  const settlingJournalProjectIds = new Set<string>();
  /** Stages started but not finished, by project id: the page may unload before they commit. */
  const stagingDocuments = new Map<string, { documentJson: string; generation: number }>();
  let mutationTail: Promise<void> = Promise.resolve();
  let loadPromise: Promise<DurableHydratedWorkbenchSnapshot> | null = null;

  const recomputeHasPending = (): boolean =>
    (hasPending = sessionSavePending || pendingProjectIds.size > 0 || conflicts.size > 0 || schemaRefusals.size > 0);
  const markLocalDraftFailure = (key: string): void => {
    localDraftFailures.add(key);
    localDraftStatus = 'unavailable';
  };
  const clearLocalDraftFailure = (key: string): void => {
    localDraftFailures.delete(key);
    localDraftStatus = localDraftFailures.size === 0 ? 'ok' : 'unavailable';
  };
  const projectDraftFailureKey = (projectId: string): string => `project:${projectId}`;
  const releaseProjectMutation = (projectId: string): void => {
    const lock = projectMutationLocks.get(projectId);
    if (!lock) {
      return;
    }
    projectMutationLocks.delete(projectId);
    void lock.release().catch(() => undefined);
  };

  const assertOwner = (): void => assertAccountScopeCurrent(owner);
  const assertNotCleared = (): void => {
    if (isTerminallyCleared) {
      throw new Error('Workbench persistence was cleared and must be reloaded.');
    }
  };
  const resolveProjectId = (projectId: string): string => retargetedProjects.get(projectId)?.projectId ?? projectId;
  const isCopyTargetOpen = (projectId: string): boolean =>
    syncEntries.has(projectId) ||
    lastKnownState?.projects.some((project) => project.id === projectId) === true ||
    getOpenProject(projectId) !== null;
  const applyProjectResolutionFences = (state: WorkbenchState): WorkbenchState => {
    let changed = false;
    const projects = state.projects.flatMap((project) => {
      const fence = projectResolutionFences.get(project.id);
      if (!fence || fence.kind === 'pending') {
        return [project];
      }
      changed = true;
      return fence.kind === 'replace' ? [fence.project] : [];
    });
    if (!changed) {
      return state;
    }
    const activeProjectId = projects.some((project) => project.id === state.activeProjectId)
      ? state.activeProjectId
      : (projects[0]?.id ?? '');
    return { ...state, activeProjectId, projects };
  };
  const enqueue = <T>(operation: () => Promise<T>): Promise<T> => {
    const result = mutationTail.then(operation, operation);
    mutationTail = result.then(
      () => undefined,
      () => undefined
    );
    return result;
  };
  const releaseAllMutationLocks = (): Promise<unknown>[] =>
    [...projectMutationLocks].map(([projectId, lock]) => {
      projectMutationLocks.delete(projectId);
      return lock.release();
    });
  let closed: Promise<void> | null = null;
  const close = (): Promise<void> => {
    if (closed) {
      return closed;
    }
    isClosed = true;
    const releases = releaseAllMutationLocks();
    if (draftStorePromise) {
      releases.push(draftStorePromise.then((store) => store.close()));
    }
    if (editorSessionPromise) {
      releases.push(editorSessionPromise.then((session) => session.release()));
    }
    closed = Promise.allSettled(releases).then(() => undefined);
    return closed;
  };

  const getOwnedDraft = async (store: ProjectDraftStore, projectId: string, editorSessionId: string) => {
    let result = await store.get(projectId, editorSessionId);
    if (result.kind === 'found' && result.draft.writerToken !== writerToken) {
      const claim = await store.claimWriter(projectId, editorSessionId, result.draft.writerToken, writerToken);
      if (claim.kind !== 'claimed') {
        return null;
      }
      result = await store.get(projectId, editorSessionId);
    } else if (result.kind === 'empty' && result.writerToken !== writerToken) {
      const started = await store.startFreshWriter(projectId, editorSessionId, result.writerToken, writerToken);
      if (started.kind !== 'started') {
        return null;
      }
      result = await store.get(projectId, editorSessionId);
    }
    return result.kind === 'found' ? result.draft : null;
  };

  const rememberUnopenableDraft = (draft: ProjectDraft | ProjectDraftSummary): void => {
    if (
      draft.state === 'corrupt' ||
      draft.documentByteSize === null ||
      draft.generation === null ||
      draft.updatedAt === null
    ) {
      return;
    }
    unopenableDrafts.set(`${draft.projectId}\u0000${draft.editorSessionId}`, {
      documentByteSize: draft.documentByteSize,
      editorSessionId: draft.editorSessionId,
      generation: draft.generation,
      projectId: draft.projectId,
      state: draft.state,
      updatedAt: draft.updatedAt,
    });
  };

  const isCompatibleDraftSummary = (draft: ProjectDraftSummary): boolean =>
    draft.state !== 'corrupt' &&
    draft.documentSchemaVersion !== null &&
    draft.documentSchemaVersion <= PROJECT_DOCUMENT_SCHEMA_VERSION &&
    draft.documentByteSize !== null &&
    draft.generation !== null &&
    draft.updatedAt !== null;

  const rememberFutureDraft = (draft: ProjectDraftSummary): void => {
    if (
      draft.state !== 'corrupt' &&
      draft.documentSchemaVersion !== null &&
      draft.documentSchemaVersion > PROJECT_DOCUMENT_SCHEMA_VERSION &&
      draft.documentByteSize !== null &&
      draft.generation !== null &&
      draft.updatedAt !== null
    ) {
      rememberUnopenableDraft(draft);
    }
  };

  const getDraftEditorSessionId = (projectId: string, editorSessionId: string): string =>
    draftEditorSessionIds.get(projectId) ?? editorSessionId;

  const serializeDraftEditorSessionIds = (): Record<string, string> => Object.fromEntries(draftEditorSessionIds);

  const retargetHandoffKey = (handoff: Pick<ProjectDraftRetargetHandoff, 'editorSessionId' | 'projectId'>): string =>
    `${handoff.projectId}\u0000${handoff.editorSessionId}`;

  /** The editor session whose page writes a lineage; an isolated lineage is named after its session and writer. */
  const getLineageEditorSessionId = (lineageEditorSessionId: string): string =>
    lineageEditorSessionId.split(':writer:', 1)[0]!;

  const isolateUnopenableDraft = (projectId: string, editorSessionId: string): string => {
    const preferred = `${editorSessionId}:writer:${writerToken}`;
    const isolated = draftEditorSessionIds.get(projectId) === preferred ? `${preferred}:${createUuid()}` : preferred;
    draftEditorSessionIds.set(projectId, isolated);
    return isolated;
  };

  const adoptNewestDraft = async (
    store: ProjectDraftStore,
    projectId: string,
    editorSessionId: string,
    draftSummaries: readonly ProjectDraftSummary[]
  ): Promise<ProjectDraft | null> => {
    const projectDrafts = draftSummaries
      .filter((item) => item.projectId === projectId)
      .sort(
        (a, b) =>
          (b.updatedAt ?? Number.NEGATIVE_INFINITY) - (a.updatedAt ?? Number.NEGATIVE_INFINITY) ||
          a.editorSessionId.localeCompare(b.editorSessionId)
      );
    for (const item of projectDrafts) {
      rememberFutureDraft(item);
    }
    const currentSummary = projectDrafts.find((item) => item.editorSessionId === editorSessionId);
    const activeEditorSessionId =
      currentSummary?.documentSchemaVersion !== null &&
      currentSummary?.documentSchemaVersion !== undefined &&
      currentSummary.documentSchemaVersion > PROJECT_DOCUMENT_SCHEMA_VERSION
        ? isolateUnopenableDraft(projectId, editorSessionId)
        : getDraftEditorSessionId(projectId, editorSessionId);
    const own = await getOwnedDraft(store, projectId, activeEditorSessionId);
    if (own) {
      for (const item of projectDrafts) {
        if (item.editorSessionId !== activeEditorSessionId && isCompatibleDraftSummary(item)) {
          rememberUnopenableDraft(item);
        }
      }
      return own;
    }
    const compatibleForeign = projectDrafts.filter(
      (item) => item.editorSessionId !== activeEditorSessionId && isCompatibleDraftSummary(item)
    );
    const foreign = compatibleForeign[0];
    if (!foreign) {
      return null;
    }
    for (const item of compatibleForeign.slice(1)) {
      rememberUnopenableDraft(item);
    }
    const adopted = await store.adopt(projectId, foreign.editorSessionId, activeEditorSessionId, writerToken);
    if (adopted.kind !== 'adopted') {
      if (adopted.kind === 'quota' || adopted.kind === 'unavailable') {
        markLocalDraftFailure(`adopt:${projectId}`);
        rememberUnopenableDraft(foreign);
      } else if (adopted.kind === 'corrupt') {
        rememberUnopenableDraft(foreign);
      }
      return null;
    }
    clearLocalDraftFailure(`adopt:${projectId}`);
    return getOwnedDraft(store, projectId, activeEditorSessionId);
  };

  const listDraftProjectCandidates = async (store: ProjectDraftStore): Promise<ProjectDraftSummary[]> => {
    const candidates: ProjectDraftSummary[] = [];
    let after: [string, string] | undefined;
    for (;;) {
      const page = await store.list({ ...(after ? { after } : {}), limit: 100 });
      if (page.kind !== 'available') {
        markLocalDraftFailure('list:drafts');
        return candidates;
      }
      for (const item of page.items) {
        rememberFutureDraft(item);
        candidates.push(item);
      }
      if (!page.nextCursor) {
        clearLocalDraftFailure('list:drafts');
        return candidates;
      }
      after = page.nextCursor;
    }
  };

  const listProjectDraftCandidates = async (
    store: ProjectDraftStore,
    projectId: string
  ): Promise<ProjectDraftSummary[]> => {
    const candidates: ProjectDraftSummary[] = [];
    let after: string | undefined;
    for (;;) {
      const page = await store.listForProject(projectId, {
        ...(after ? { after } : {}),
        limit: PROJECT_DRAFT_PROJECT_LIMIT,
      });
      if (page.kind !== 'available') {
        markLocalDraftFailure(`list:${projectId}`);
        return candidates;
      }
      for (const item of page.items) {
        rememberFutureDraft(item);
        candidates.push(item);
      }
      if (!page.nextCursor) {
        clearLocalDraftFailure(`list:${projectId}`);
        return candidates;
      }
      after = page.nextCursor;
    }
  };

  const groupDraftCandidatesByProject = (
    candidates: readonly ProjectDraftSummary[]
  ): Map<string, ProjectDraftSummary[]> => {
    const grouped = new Map<string, ProjectDraftSummary[]>();
    for (const candidate of candidates) {
      const projectCandidates = grouped.get(candidate.projectId) ?? [];
      projectCandidates.push(candidate);
      grouped.set(candidate.projectId, projectCandidates);
    }
    return grouped;
  };

  const selectAutoOpenDraftProjectIds = (
    candidates: ProjectDraftSummary[],
    alwaysOpenProjectIds: ReadonlySet<string>
  ): string[] => {
    const selected: string[] = [];
    let selectedBytes = 0;
    const candidatesByProject = new Map<string, ProjectDraftSummary[]>();
    for (const candidate of candidates) {
      if (!isCompatibleDraftSummary(candidate)) {
        continue;
      }
      const projectCandidates = candidatesByProject.get(candidate.projectId) ?? [];
      projectCandidates.push(candidate);
      candidatesByProject.set(candidate.projectId, projectCandidates);
    }
    const newestCandidates = [...candidatesByProject.values()].map(
      (items) => items.sort((a, b) => (b.updatedAt ?? -1) - (a.updatedAt ?? -1))[0]!
    );
    for (const candidate of newestCandidates.sort(
      (a, b) => (b.updatedAt ?? -1) - (a.updatedAt ?? -1) || a.projectId.localeCompare(b.projectId)
    )) {
      if (alwaysOpenProjectIds.has(candidate.projectId)) {
        continue;
      }
      const byteSize = Math.max(
        ...(candidatesByProject.get(candidate.projectId) ?? [candidate]).map((item) => item.documentByteSize ?? 0)
      );
      if (selected.length < autoOpenDraftLimit && selectedBytes + byteSize <= autoOpenDraftByteLimit) {
        selected.push(candidate.projectId);
        selectedBytes += byteSize;
      } else {
        for (const item of candidatesByProject.get(candidate.projectId) ?? []) {
          rememberUnopenableDraft(item);
        }
      }
    }
    return selected;
  };

  const listRetargetHandoffs = async (store: ProjectDraftStore): Promise<ProjectDraftRetargetHandoff[]> => {
    const handoffs: ProjectDraftRetargetHandoff[] = [];
    let after: ProjectDraftRetargetCursor | undefined;
    for (;;) {
      const page = await store.listRetargets({ ...(after ? { after } : {}), limit: 100 });
      if (page.kind !== 'available') {
        markLocalDraftFailure('list:retargets');
        return handoffs;
      }
      handoffs.push(...page.items);
      if (!page.nextCursor) {
        clearLocalDraftFailure('list:retargets');
        return handoffs;
      }
      after = page.nextCursor;
    }
  };

  const createProjectSavePlan = (inputProject: Project) => {
    const project = resolveInputProject(inputProject);
    const serialized = serializeProjectDocumentV3Json(project);
    const entry = syncEntries.get(project.id);
    const existingDraft = volatileDrafts.get(project.id);
    const needsDocumentUpgrade = projectsRequiringDocumentUpgrade.has(project.id);
    const needsStage =
      needsDocumentUpgrade ||
      (!entry && existingDraft?.documentJson !== serialized.documentJson) ||
      (entry !== undefined &&
        entry.pushedDoc !== serialized.documentJson &&
        existingDraft?.documentJson !== serialized.documentJson);
    const needsPush =
      needsStage || pendingProjectIds.has(project.id) || conflicts.has(project.id) || schemaRefusals.has(project.id);
    return { existingDraft, needsPush, needsStage, project, serialized };
  };
  type ProjectSavePlan = ReturnType<typeof createProjectSavePlan>;

  const stageProject = async (
    project: Project,
    serialized: ReturnType<typeof serializeProjectDocumentV3Json> = serializeProjectDocumentV3Json(project)
  ) => {
    const store = await getDraftStore();
    const session = await getEditorSessionForService();
    if (serialized.byteSize > PROJECT_DOCUMENT_MAX_BYTES) {
      pendingProjectIds.add(project.id);
      hasPending = true;
      throw new ProjectDocumentTooLargeError(serialized.byteSize);
    }
    const generation = nextGeneration(project.id);
    noteGeneration(project.id, generation);
    const entry = syncEntries.get(project.id);
    if (!entry || entry.pushedDoc !== serialized.documentJson) {
      pendingProjectIds.add(project.id);
    }
    let draftInput = {
      baseMinimumCanvasSchemaVersion:
        entry?.minimumCanvasSchemaVersion ?? getProjectCanvasSchemaRequirement(serialized.document),
      baseRevision: entry?.revision ?? null,
      documentJson: serialized.documentJson,
      documentSchemaVersion: PROJECT_DOCUMENT_SCHEMA_VERSION,
      editorSessionId: getDraftEditorSessionId(project.id, session.id),
      generation,
      projectId: project.id,
      updatedAt: Date.parse(now()),
      writerToken,
    };
    stagingDocuments.set(project.id, { documentJson: serialized.documentJson, generation });
    let result: Awaited<ReturnType<ProjectDraftStore['stage']>>;
    try {
      result = await store.stage(draftInput);
      if (result.kind === 'corrupt') {
        markLocalDraftFailure(projectDraftFailureKey(project.id));
        const corruptKey = `corrupt:${project.id}:${draftInput.editorSessionId}`;
        const cleanup = await store.deleteCorrupt(project.id, draftInput.editorSessionId);
        if (cleanup.kind === 'deleted') {
          clearLocalDraftFailure(corruptKey);
        } else {
          markLocalDraftFailure(corruptKey);
        }
        draftInput = {
          ...draftInput,
          editorSessionId: isolateUnopenableDraft(project.id, session.id),
        };
        result = await store.stage(draftInput);
      }
    } finally {
      if (stagingDocuments.get(project.id)?.generation === generation) {
        stagingDocuments.delete(project.id);
      }
    }
    if (['corrupt', 'fenced', 'generation-conflict', 'stale'].includes(result.kind)) {
      markLocalDraftFailure(projectDraftFailureKey(project.id));
      throw new ProjectDraftWriteRejectedError(result.kind as 'corrupt' | 'fenced' | 'generation-conflict' | 'stale');
    }
    const volatileInput = { ...draftInput, documentByteSize: getUtf8ByteSize(serialized.documentJson) };
    const previousVolatile = volatileDrafts.get(project.id);
    const dirty = previousVolatile
      ? toDirtyProjectDraft(previousVolatile, volatileInput)
      : ({ ...volatileInput, state: 'dirty' } as ProjectDraft);
    const conflict = conflicts.get(project.id);
    const refusal = schemaRefusals.get(project.id);
    volatileDrafts.set(
      project.id,
      conflict
        ? toConflictProjectDraft(
            dirty,
            conflict.kind === 'revision'
              ? { kind: 'revision', serverRevision: conflict.serverRevision! }
              : { kind: 'deleted' }
          )
        : refusal
          ? toSchemaRefusedProjectDraft(dirty, refusal)
          : dirty
    );
    if (['quota', 'too-large', 'unavailable'].includes(result.kind)) {
      markLocalDraftFailure(projectDraftFailureKey(project.id));
    } else {
      clearLocalDraftFailure(projectDraftFailureKey(project.id));
      // The lineage now holds this generation, which supersedes the writer's journal through it.
      const journaled = journaledDocuments.get(project.id);
      if (journaled && journaled.generation <= generation && journaled.editorSessionId === draftInput.editorSessionId) {
        journaledDocuments.delete(project.id);
        retiredJournalProjectIds.delete(project.id);
        await store.discardUnloadJournal(project.id, journaled.editorSessionId, writerToken, journaled.generation);
      }
    }
    return { editorSessionId: draftInput.editorSessionId, generation, serialized, store };
  };

  const settleAck = async (
    project: Project,
    sentGeneration: number,
    record: ProjectRecordDTO,
    documentJson: string,
    store: ProjectDraftStore,
    editorSessionId: string
  ): Promise<void> => {
    syncEntries.set(project.id, {
      minimumCanvasSchemaVersion: record.minimum_canvas_schema_version,
      pushedDoc: documentJson,
      revision: record.revision,
    });
    upsertProjectSummary(
      {
        id: project.id,
        minimumCanvasSchemaVersion: record.minimum_canvas_schema_version,
        name: project.name,
        revision: record.revision,
      },
      owner
    );
    recordProjectCover(project.id, selectCoverImageName(record.data), owner);
    conflicts.delete(project.id);
    schemaRefusals.delete(project.id);
    projectsRequiringDocumentUpgrade.delete(project.id);
    pendingProjectIds.delete(project.id);
    // Also settles the writer's unload journal through `sentGeneration`, in the same transaction.
    const settled = await store.settleAcknowledgement(
      project.id,
      editorSessionId,
      writerToken,
      sentGeneration,
      record.revision,
      record.minimum_canvas_schema_version
    );
    const journaled = journaledDocuments.get(project.id);
    if (journaled && journaled.generation <= sentGeneration) {
      await settleJournaledDocument(project.id, store);
    } else if (
      journaled &&
      journaled.editorSessionId === editorSessionId &&
      journaled.baseRevision !== record.revision
    ) {
      // A newer journaled document builds on the acknowledged one: rebase it, as a newer staged draft is rebased.
      const rebased = {
        ...journaled,
        baseMinimumCanvasSchemaVersion: record.minimum_canvas_schema_version,
        baseRevision: record.revision,
      };
      const write = store.journalBeforeUnload({ entries: [rebased], retired: [] });
      if (write.kind === 'started' && (await write.written) && journaledDocuments.get(project.id) === journaled) {
        journaledDocuments.set(project.id, rebased);
      }
    }
    if (settled.kind === 'rebased') {
      pendingProjectIds.add(project.id);
      hasPending = true;
      clearLocalDraftFailure(projectDraftFailureKey(project.id));
    } else if (settled.kind === 'deleted') {
      clearLocalDraftFailure(projectDraftFailureKey(project.id));
    } else {
      markLocalDraftFailure(projectDraftFailureKey(project.id));
    }
    const volatile = volatileDrafts.get(project.id);
    if (volatile) {
      if (volatile.generation > sentGeneration) {
        volatileDrafts.set(project.id, toDirtyProjectDraft(volatile, { baseRevision: record.revision }));
        pendingProjectIds.add(project.id);
        hasPending = true;
      } else {
        volatileDrafts.delete(project.id);
      }
    }
  };

  const markConflict = async (
    projectId: string,
    generation: number,
    conflict: { kind: 'deleted' } | { kind: 'revision'; serverRevision: number },
    store: ProjectDraftStore,
    editorSessionId: string
  ): Promise<void> => {
    const info: ProjectConflictInfo = {
      detectedAt: now(),
      kind: conflict.kind,
      projectId,
      ...(conflict.kind === 'revision' ? { serverRevision: conflict.serverRevision } : {}),
    };
    conflicts.set(projectId, info);
    pendingProjectIds.add(projectId);
    hasPending = true;
    const settled = await store.settleConflict(projectId, editorSessionId, writerToken, generation, conflict);
    const volatile = volatileDrafts.get(projectId);
    if (volatile) {
      volatileDrafts.set(projectId, toConflictProjectDraft(volatile, conflict));
    }
    if (settled.kind === 'marked' || settled.kind === 'rebased') {
      clearLocalDraftFailure(projectDraftFailureKey(projectId));
    } else {
      markLocalDraftFailure(projectDraftFailureKey(projectId));
    }
  };

  const resolveInputProject = (inputProject: Project): Project => {
    const retarget = retargetedProjects.get(inputProject.id);
    return retarget
      ? withAuthoritativeProjectBoard(
          {
            ...inputProject,
            id: retarget.projectId,
            name: inputProject.name === retarget.sourceName ? retarget.name : inputProject.name,
          },
          retarget.boardId
        )
      : inputProject;
  };

  const retargetCopyDocument = (documentJson: string, reservation: ProjectDraftCopyReservation): string => {
    const current = JSON.parse(documentJson) as Record<string, unknown>;
    return JSON.stringify({
      ...current,
      id: reservation.copyProjectId,
      name: current.name === reservation.copySourceProjectName ? reservation.copyProjectName : (current.name as string),
    });
  };

  const pushStagedProject = async (
    project: Project,
    staged: Awaited<ReturnType<typeof stageProject>>
  ): Promise<ProjectPushOutcome> => {
    assertNotCleared();
    assertOwner();
    const { editorSessionId, generation, serialized, store } = staged;
    const { document, documentJson } = serialized;
    const entry = syncEntries.get(project.id);
    const refusal = schemaRefusals.get(project.id);

    if (refusal) {
      hasPending = true;
      return { documentJson, kind: 'schema-refused', refusal };
    }
    if (deletedProjectIds.has(project.id)) {
      pendingProjectIds.delete(project.id);
      return { documentJson, kind: 'superseded' };
    }
    if (conflicts.has(project.id)) {
      hasPending = true;
      return { documentJson, kind: 'conflicted' };
    }
    if (entry?.pushedDoc === documentJson && !projectsRequiringDocumentUpgrade.has(project.id)) {
      await settleAck(
        project,
        generation,
        {
          board_id: '',
          created_at: '',
          data: document,
          minimum_canvas_schema_version: entry.minimumCanvasSchemaVersion,
          name: project.name,
          project_id: project.id,
          revision: entry.revision,
          updated_at: '',
        },
        documentJson,
        store,
        editorSessionId
      );
      return { documentJson, kind: 'acknowledged' };
    }

    try {
      if (!entry) {
        const created = await api.createProject(
          {
            data: document,
            minimum_canvas_schema_version: getProjectCanvasSchemaRequirement(document),
            name: project.name,
            project_id: project.id,
          },
          owner
        );
        assertOwner();
        const serverJson = JSON.stringify(created.data);
        if (serverJson !== documentJson) {
          await markConflict(
            project.id,
            generation,
            { kind: 'revision', serverRevision: created.revision },
            store,
            editorSessionId
          );
          return { documentJson, kind: 'conflicted' };
        }
        await settleAck(project, generation, created, documentJson, store, editorSessionId);
        pendingBoardAssignments.set(project.id, { boardId: created.board_id, projectId: project.id });
        return { documentJson, kind: 'acknowledged' };
      }

      const raisedFloor = getRaisedCanvasSchemaFloor(document, entry.minimumCanvasSchemaVersion);
      const updated = await api.updateProject(
        project.id,
        {
          data: document,
          expected_revision: entry.revision,
          ...(raisedFloor === undefined ? {} : { minimum_canvas_schema_version: raisedFloor }),
          name: project.name,
        },
        owner.signal
      );
      assertOwner();
      await settleAck(project, generation, updated, documentJson, store, editorSessionId);
      return { documentJson, kind: 'acknowledged' };
    } catch (error) {
      assertOwner();
      const refusalError = error instanceof ProjectCreateAbsentError ? error.cause : error;
      const sizeRefusal = getProjectWriteSizeRefusal(refusalError);
      if (sizeRefusal) {
        hasPending = true;
        throw new ProjectDocumentTooLargeError(sizeRefusal.actualBytes, sizeRefusal.maxBytes);
      }
      const schemaRefusal = getProjectCanvasSchemaCompatibilityRefusal(refusalError);
      if (schemaRefusal) {
        const refusal = toCanvasSchemaRefusal(schemaRefusal);
        schemaRefusals.set(project.id, refusal);
        hasPending = true;
        await store.settleSchemaRefusal(project.id, editorSessionId, writerToken, generation, refusal);
        const volatile = volatileDrafts.get(project.id);
        if (volatile) {
          volatileDrafts.set(project.id, toSchemaRefusedProjectDraft(volatile, refusal));
        }
        return { documentJson, kind: 'schema-refused', refusal };
      }
      if (error instanceof ProjectCreateAbsentError) {
        throw error.cause;
      }
      if (isStatus(error, 404)) {
        if (deletedProjectIds.has(project.id)) {
          return { documentJson, kind: 'superseded' };
        }
        await markConflict(project.id, generation, { kind: 'deleted' }, store, editorSessionId);
        return { documentJson, kind: 'conflicted' };
      }
      if (isStatus(error, 409)) {
        try {
          const current = await api.getProject(project.id, owner.signal);
          assertOwner();
          const currentJson = recordDocumentJson(current);
          if (currentJson === documentJson) {
            await settleAck(project, generation, current, documentJson, store, editorSessionId);
            return { documentJson, kind: 'acknowledged' };
          }
          if (entry && currentJson === entry.pushedDoc) {
            syncEntries.set(project.id, {
              minimumCanvasSchemaVersion: current.minimum_canvas_schema_version,
              pushedDoc: entry.pushedDoc,
              revision: current.revision,
            });
            const retryRaisedFloor = getRaisedCanvasSchemaFloor(document, current.minimum_canvas_schema_version);
            const retried = await api.updateProject(
              project.id,
              {
                data: document,
                expected_revision: current.revision,
                ...(retryRaisedFloor === undefined ? {} : { minimum_canvas_schema_version: retryRaisedFloor }),
                name: project.name,
              },
              owner.signal
            );
            assertOwner();
            await settleAck(project, generation, retried, documentJson, store, editorSessionId);
            return { documentJson, kind: 'acknowledged' };
          }
          await markConflict(
            project.id,
            generation,
            { kind: 'revision', serverRevision: current.revision },
            store,
            editorSessionId
          );
          return { documentJson, kind: 'conflicted' };
        } catch (readError) {
          assertOwner();
          const retrySizeRefusal = getProjectWriteSizeRefusal(readError);
          if (retrySizeRefusal) {
            throw new ProjectDocumentTooLargeError(retrySizeRefusal.actualBytes, retrySizeRefusal.maxBytes);
          }
          const retrySchemaRefusal = getProjectCanvasSchemaCompatibilityRefusal(readError);
          if (retrySchemaRefusal) {
            const refusal = toCanvasSchemaRefusal(retrySchemaRefusal);
            schemaRefusals.set(project.id, refusal);
            await store.settleSchemaRefusal(project.id, editorSessionId, writerToken, generation, refusal);
            const volatile = volatileDrafts.get(project.id);
            if (volatile) {
              volatileDrafts.set(project.id, toSchemaRefusedProjectDraft(volatile, refusal));
            }
            return { documentJson, kind: 'schema-refused', refusal };
          }
          if (isStatus(readError, 404)) {
            await markConflict(project.id, generation, { kind: 'deleted' }, store, editorSessionId);
            return { documentJson, kind: 'conflicted' };
          }
          if (isStatus(readError, 409)) {
            let latest: ProjectRecordDTO;
            try {
              latest = await api.getProject(project.id, owner.signal);
            } catch (latestError) {
              assertOwner();
              if (isStatus(latestError, 404)) {
                await markConflict(project.id, generation, { kind: 'deleted' }, store, editorSessionId);
                return { documentJson, kind: 'conflicted' };
              }
              if (isDeterministicClientError(latestError)) {
                throw latestError;
              }
              hasPending = true;
              return { documentJson, kind: 'unsynced' };
            }
            assertOwner();
            if (recordDocumentJson(latest) === documentJson) {
              await settleAck(project, generation, latest, documentJson, store, editorSessionId);
              return { documentJson, kind: 'acknowledged' };
            }
            await markConflict(
              project.id,
              generation,
              { kind: 'revision', serverRevision: latest.revision },
              store,
              editorSessionId
            );
            return { documentJson, kind: 'conflicted' };
          }
          if (isDeterministicClientError(readError)) {
            throw readError;
          }
          hasPending = true;
          return { documentJson, kind: 'unsynced' };
        }
      }
      if (isDeterministicClientError(error)) {
        throw error;
      }
      hasPending = true;
      return { documentJson, kind: 'unsynced' };
    }
  };

  const pushProjectPlan = async (
    plan: ProjectSavePlan,
    staged?: Awaited<ReturnType<typeof stageProject>>
  ): Promise<ProjectPushOutcome> => {
    // The document went back to what recovery holds (an undo, say): the journal of a later state must not return.
    if (journaledDocuments.has(plan.project.id) && isInRecovery(plan.project.id, plan.serialized.documentJson)) {
      await settleJournaledDocument(plan.project.id, await getDraftStore());
    }
    if (deletedProjectIds.has(plan.project.id)) {
      return { documentJson: plan.serialized.documentJson, kind: 'superseded' };
    }
    if (!plan.needsPush) {
      return { documentJson: plan.serialized.documentJson, kind: 'acknowledged' };
    }
    const refusal = schemaRefusals.get(plan.project.id);
    if (!plan.needsStage && refusal) {
      return { documentJson: plan.serialized.documentJson, kind: 'schema-refused', refusal };
    }
    if (!plan.needsStage && conflicts.has(plan.project.id)) {
      return { documentJson: plan.serialized.documentJson, kind: 'conflicted' };
    }
    if (plan.needsStage) {
      return pushStagedProject(plan.project, staged ?? (await stageProject(plan.project, plan.serialized)));
    }
    if (!plan.existingDraft) {
      return { documentJson: plan.serialized.documentJson, kind: 'unsynced' };
    }
    const store = await getDraftStore();
    return pushStagedProject(plan.project, {
      editorSessionId: plan.existingDraft.editorSessionId,
      generation: plan.existingDraft.generation,
      serialized: plan.serialized,
      store,
    });
  };

  const pushProject = (inputProject: Project): Promise<ProjectPushOutcome> =>
    pushProjectPlan(createProjectSavePlan(inputProject));

  const getRecoverableDraftSummaries = () =>
    [...unopenableDrafts.values()]
      .sort(
        (a, b) =>
          b.updatedAt - a.updatedAt ||
          a.projectId.localeCompare(b.projectId) ||
          a.editorSessionId.localeCompare(b.editorSessionId)
      )
      .map(({ editorSessionId, generation, projectId, updatedAt }) => ({
        editorSessionId,
        generation,
        projectId,
        updatedAt,
      }));

  const getProjectSyncInfo = (projectId: string) => {
    const entry = syncEntries.get(projectId);
    const schemaRefusal = schemaRefusals.get(projectId);
    const conflict = conflicts.get(projectId);
    return {
      isPendingPush:
        !entry || pendingProjectIds.has(projectId) || conflict !== undefined || schemaRefusal !== undefined,
      revision: entry?.revision ?? null,
      ...(conflict
        ? {
            conflict: {
              detectedAt: conflict.detectedAt,
              kind: conflict.kind,
              ...(conflict.serverRevision === undefined ? {} : { serverRevision: conflict.serverRevision }),
            },
          }
        : {}),
      ...(schemaRefusal ? { schemaRefusal } : {}),
    };
  };

  const reportSync = (projects: Project[]): void => {
    reportProjectSync({
      hasPendingChanges: hasPending,
      localDraftStatus,
      projects: Object.fromEntries(projects.map((project) => [project.id, getProjectSyncInfo(project.id)])),
      recoverableDrafts: getRecoverableDraftSummaries(),
    });
  };

  const reportSingleProjectSync = (projectId: string): void => {
    reportProjectSyncEntry(projectId, getProjectSyncInfo(projectId), {
      hasPendingChanges: hasPending,
      localDraftStatus,
      recoverableDrafts: getRecoverableDraftSummaries(),
    });
  };

  const persistCurrentSession = async (state: WorkbenchState, editorSessionId: string): Promise<boolean> => {
    const draftLineages = serializeDraftEditorSessionIds();
    const sessionJson = serializeSessionBlob(state, editorSessionId, draftLineages);
    if (sessionJson !== lastSessionJson) {
      try {
        await api.saveSession(state, editorSessionId, draftLineages, owner.signal);
        assertOwner();
        lastSessionJson = sessionJson;
      } catch {
        assertOwner();
        sessionSavePending = true;
        hasPending = true;
        return false;
      }
    }
    sessionSavePending = false;
    recomputeHasPending();
    return true;
  };

  const acknowledgeExplicitRetarget = async (projectId: string | undefined): Promise<void> => {
    if (!projectId) {
      return;
    }
    const store = await getDraftStore();
    for (const [key, handoff] of pendingRetargetHandoffs) {
      if (handoff.projectId !== projectId) {
        continue;
      }
      const result = await store.acknowledgeRetarget(
        handoff.projectId,
        handoff.editorSessionId,
        handoff.targetProjectId
      );
      if (result.kind === 'unavailable') {
        markLocalDraftFailure(`retarget:${key}`);
      } else if (result.kind === 'deleted') {
        clearLocalDraftFailure(`retarget:${key}`);
        pendingRetargetHandoffs.delete(key);
      }
    }
    const stillPending = [...pendingRetargetHandoffs.values()].some((handoff) => handoff.projectId === projectId);
    if (!stillPending) {
      explicitRetargetSourcesAwaitingAck.delete(projectId);
    }
  };

  const acknowledgeAwaitingExplicitRetargets = async (): Promise<void> => {
    for (const projectId of explicitRetargetSourcesAwaitingAck) {
      await acknowledgeExplicitRetarget(projectId);
    }
  };

  const acknowledgeCommittedRetargets = async (state: WorkbenchState): Promise<void> => {
    const openProjectIds = new Set(state.projects.map((project) => project.id));
    const store = await getDraftStore();
    for (const [key, handoff] of pendingRetargetHandoffs) {
      const committedOpenRetarget =
        openProjectIds.has(handoff.targetProjectId) && !openProjectIds.has(handoff.projectId);
      const committedClosedRetarget =
        closedRetargetTargetsAwaitingAck.has(handoff.targetProjectId) && !openProjectIds.has(handoff.targetProjectId);
      if (!committedOpenRetarget && !committedClosedRetarget) {
        continue;
      }
      const result = await store.acknowledgeRetarget(
        handoff.projectId,
        handoff.editorSessionId,
        handoff.targetProjectId
      );
      if (result.kind === 'unavailable') {
        markLocalDraftFailure(`retarget:${key}`);
      } else if (result.kind === 'deleted') {
        clearLocalDraftFailure(`retarget:${key}`);
        pendingRetargetHandoffs.delete(key);
        if (![...pendingRetargetHandoffs.values()].some((item) => item.targetProjectId === handoff.targetProjectId)) {
          closedRetargetTargetsAwaitingAck.delete(handoff.targetProjectId);
        }
      }
    }
  };

  const requireDraft = async (
    projectId: string
  ): Promise<{ draft: ProjectDraft; editorSessionId: string; store: ProjectDraftStore }> => {
    const store = await getDraftStore();
    const session = await getEditorSessionForService();
    const editorSessionId = getDraftEditorSessionId(projectId, session.id);
    const draft = await getOwnedDraft(store, projectId, editorSessionId);
    const volatileDraft = volatileDrafts.get(projectId);
    const availableDraft =
      draft && volatileDraft
        ? draft.generation > volatileDraft.generation
          ? draft
          : volatileDraft
        : (draft ?? volatileDraft);
    if (!availableDraft) {
      throw new Error('The local project draft is unavailable.');
    }
    return { draft: availableDraft, editorSessionId, store };
  };

  /**
   * What the next load would restore for a project is exactly `documentJson`: its newest staged draft when it has one
   * (and that stage was durable), otherwise the acknowledged server document.
   */
  const isInRecovery = (projectId: string, documentJson: string): boolean => {
    const volatile = volatileDrafts.get(projectId);
    return volatile
      ? volatile.documentJson === documentJson && !localDraftFailures.has(projectDraftFailureKey(projectId))
      : syncEntries.get(projectId)?.pushedDoc === documentJson;
  };

  /**
   * Settles this writer's journal of a project in its lineage's claim (the mark makes a deletion that fails moot), then
   * deletes it. The record is kept, for a later retry, until the mark commits. A fenced lineage needs no mark: its
   * entries are discarded anyway. False while the journal could still come back.
   */
  const settleJournaledDocument = async (projectId: string, store: ProjectDraftStore): Promise<boolean> => {
    const journaled = journaledDocuments.get(projectId);
    if (!journaled) {
      return true;
    }
    settlingJournalProjectIds.add(projectId);
    let settled: Awaited<ReturnType<ProjectDraftStore['settleUnloadJournal']>>;
    try {
      settled = await store.settleUnloadJournal(
        projectId,
        journaled.editorSessionId,
        writerToken,
        journaled.generation
      );
    } finally {
      settlingJournalProjectIds.delete(projectId);
    }
    if (settled.kind !== 'settled' && settled.kind !== 'fenced') {
      return false;
    }
    // A hidden event meanwhile journaled a newer generation, which this settlement does not cover.
    if (journaledDocuments.get(projectId) === journaled) {
      journaledDocuments.delete(projectId);
      retiredJournalProjectIds.delete(projectId);
    }
    return true;
  };

  const deleteDraft = async (projectId: string): Promise<void> => {
    const { draft, editorSessionId, store } = await requireDraft(projectId);
    // Settling the journal in the deleting transaction keeps a discarded edit from coming back.
    const result = await store.delete(projectId, editorSessionId, draft.writerToken, issuedGenerations.get(projectId));
    if (result.kind !== 'deleted') {
      throw new Error('The local project draft could not be deleted.');
    }
    volatileDrafts.delete(projectId);
    await settleJournaledDocument(projectId, store);
  };

  const alignLoadedDraftWithProject = async (
    draft: ProjectDraft,
    project: Project,
    store: ProjectDraftStore
  ): Promise<ProjectDraft> => {
    const serialized = serializeProjectDocumentV3Json(project);
    if (draft.documentJson === serialized.documentJson) {
      clearLocalDraftFailure(projectDraftFailureKey(project.id));
      volatileDrafts.set(project.id, draft);
      return draft;
    }
    const generation = Math.max(draft.generation, issuedGenerations.get(project.id) ?? 0) + 1;
    let input = {
      baseMinimumCanvasSchemaVersion: draft.baseMinimumCanvasSchemaVersion,
      baseRevision: draft.baseRevision,
      documentJson: serialized.documentJson,
      documentSchemaVersion: PROJECT_DOCUMENT_SCHEMA_VERSION,
      editorSessionId: draft.editorSessionId,
      generation,
      projectId: project.id,
      updatedAt: Date.parse(now()),
      writerToken,
    };
    let result = await store.stage(input);
    if (result.kind === 'corrupt') {
      markLocalDraftFailure(projectDraftFailureKey(project.id));
      const corruptKey = `corrupt:${project.id}:${input.editorSessionId}`;
      const cleanup = await store.deleteCorrupt(project.id, input.editorSessionId);
      if (cleanup.kind === 'deleted') {
        clearLocalDraftFailure(corruptKey);
      } else {
        markLocalDraftFailure(corruptKey);
      }
      const session = await getEditorSessionForService();
      input = { ...input, editorSessionId: isolateUnopenableDraft(project.id, session.id) };
      result = await store.stage(input);
    }
    if (['corrupt', 'fenced', 'generation-conflict', 'stale'].includes(result.kind)) {
      markLocalDraftFailure(projectDraftFailureKey(project.id));
      throw new ProjectDraftWriteRejectedError(result.kind as ProjectDraftWriteRejectedError['reason']);
    }
    if (['quota', 'too-large', 'unavailable'].includes(result.kind)) {
      markLocalDraftFailure(projectDraftFailureKey(project.id));
    } else {
      clearLocalDraftFailure(projectDraftFailureKey(project.id));
    }
    const aligned = {
      ...draft,
      ...input,
      documentByteSize: serialized.byteSize,
    } as ProjectDraft;
    noteGeneration(project.id, generation);
    volatileDrafts.set(project.id, aligned);
    return aligned;
  };

  const hydrateProjectWithDraft = async (projectId: string, projectName: string): Promise<ProjectLoadResult> => {
    assertOwner();
    const [store, session] = await Promise.all([getDraftStore(), getEditorSessionForService()]);
    const draftSummaries = await listProjectDraftCandidates(store, projectId);
    const draft = await adoptNewestDraft(store, projectId, session.id, draftSummaries);
    assertOwner();
    const reportOwnedProjectSync = (): void => {
      assertOwner();
      reportSingleProjectSync(projectId);
    };
    let loadedDraft: LoadedDraft | null = null;
    if (draft) {
      noteGeneration(projectId, draft.generation);
      volatileDrafts.set(projectId, draft);
      const project = loadDraftProject(draft);
      if (project) {
        loadedDraft = { draft, project };
      } else {
        rememberUnopenableDraft(draft);
        isolateUnopenableDraft(projectId, session.id);
        generations.delete(projectId);
        volatileDrafts.delete(projectId);
      }
    }

    let record: ProjectRecordDTO;
    try {
      record = await api.getProject(projectId, owner.signal);
      assertOwner();
    } catch (error) {
      assertOwner();
      const compatibility = getProjectCanvasSchemaCompatibilityRefusal(error);
      if (compatibility && loadedDraft) {
        const refusal = toCanvasSchemaRefusal(compatibility);
        loadedDraft.draft = await alignLoadedDraftWithProject(loadedDraft.draft, loadedDraft.project, store);
        schemaRefusals.set(projectId, refusal);
        await store.settleSchemaRefusal(
          projectId,
          loadedDraft.draft.editorSessionId,
          writerToken,
          loadedDraft.draft.generation,
          refusal
        );
        volatileDrafts.set(projectId, toSchemaRefusedProjectDraft(loadedDraft.draft, refusal));
        pendingProjectIds.add(projectId);
        hasPending = true;
        reportOwnedProjectSync();
        return { project: loadedDraft.project, status: 'loaded' };
      }
      if (isStatus(error, 404) && loadedDraft) {
        loadedDraft.draft = await alignLoadedDraftWithProject(loadedDraft.draft, loadedDraft.project, store);
        if (loadedDraft.draft.baseRevision === null) {
          pendingProjectIds.add(projectId);
          hasPending = true;
        } else {
          await markConflict(
            projectId,
            loadedDraft.draft.generation,
            { kind: 'deleted' },
            store,
            loadedDraft.draft.editorSessionId
          );
        }
        reportOwnedProjectSync();
        return { project: loadedDraft.project, status: 'loaded' };
      }
      if (compatibility) {
        return {
          refused: {
            projectId,
            projectName,
            raw: null,
            refusal: {
              raw: null,
              scope: 'document',
              status: 'unsupported-version',
              version: compatibility.minimumCanvasSchemaVersion,
            },
            source: 'canvas',
          },
          status: 'refused',
        };
      }
      return { status: 'unavailable' };
    }

    const result = deserializeProjectRecord(record);
    if (result.status !== 'loaded') {
      if (!loadedDraft) {
        return result;
      }
      const localProject = withAuthoritativeProjectBoard(loadedDraft.project, record.board_id);
      loadedDraft.draft = await alignLoadedDraftWithProject(loadedDraft.draft, localProject, store);
      const refusal: ProjectSchemaRefusal =
        result.status === 'refused' &&
        result.refused.source === 'project-document' &&
        result.refused.refusal.status === 'unsupported-version'
          ? toDocumentSchemaRefusal(result.refused.refusal.version)
          : result.status === 'refused' &&
              result.refused.source === 'canvas' &&
              result.refused.refusal.status === 'unsupported-version'
            ? toCanvasSchemaRefusal({
                maxCanvasSchemaVersion: MAX_SUPPORTED_CANVAS_SCHEMA_VERSION,
                minimumCanvasSchemaVersion: result.refused.refusal.version,
              })
            : { kind: 'invalid-server-document' };
      schemaRefusals.set(projectId, refusal);
      await store.settleSchemaRefusal(
        projectId,
        loadedDraft.draft.editorSessionId,
        writerToken,
        loadedDraft.draft.generation,
        refusal
      );
      volatileDrafts.set(projectId, toSchemaRefusedProjectDraft(loadedDraft.draft, refusal));
      pendingProjectIds.add(projectId);
      hasPending = true;
      reportOwnedProjectSync();
      return { project: localProject, status: 'loaded' };
    }

    const serverJson = serializeProjectDocumentV3Json(result.project).documentJson;
    syncEntries.set(projectId, {
      minimumCanvasSchemaVersion: record.minimum_canvas_schema_version,
      pushedDoc: serverJson,
      revision: record.revision,
    });
    if (recordRequiresDocumentUpgrade(record)) {
      projectsRequiringDocumentUpgrade.add(projectId);
      pendingProjectIds.add(projectId);
      hasPending = true;
    }
    if (!loadedDraft) {
      reportOwnedProjectSync();
      return result;
    }

    const localProject = withAuthoritativeProjectBoard(loadedDraft.project, record.board_id);
    loadedDraft.draft = await alignLoadedDraftWithProject(loadedDraft.draft, localProject, store);
    if (loadedDraft.draft.documentJson === serverJson) {
      await store.settleAcknowledgement(
        projectId,
        loadedDraft.draft.editorSessionId,
        writerToken,
        loadedDraft.draft.generation,
        record.revision,
        record.minimum_canvas_schema_version
      );
      volatileDrafts.delete(projectId);
      pendingProjectIds.delete(projectId);
      reportOwnedProjectSync();
      return result;
    }
    if (
      loadedDraft.draft.state === 'schema-refused' &&
      loadedDraft.draft.refusal.kind === 'canvas' &&
      !isCanvasSchemaVersionSupported(loadedDraft.draft.refusal.minimumCanvasSchemaVersion)
    ) {
      schemaRefusals.set(projectId, loadedDraft.draft.refusal);
    } else if (loadedDraft.draft.state === 'conflict' || loadedDraft.draft.baseRevision !== record.revision) {
      await markConflict(
        projectId,
        loadedDraft.draft.generation,
        loadedDraft.draft.state === 'conflict'
          ? loadedDraft.draft.conflict
          : { kind: 'revision', serverRevision: record.revision },
        store,
        loadedDraft.draft.editorSessionId
      );
    } else {
      if (loadedDraft.draft.state === 'schema-refused') {
        await store.resumeSchemaRefused(
          projectId,
          loadedDraft.draft.editorSessionId,
          writerToken,
          loadedDraft.draft.generation
        );
        volatileDrafts.set(projectId, toDirtyProjectDraft(loadedDraft.draft, {}));
      }
      pendingProjectIds.add(projectId);
      hasPending = true;
    }
    reportOwnedProjectSync();
    return { project: localProject, status: 'loaded' };
  };

  const service: DurableSyncedWorkbenchPersistence = {
    acknowledgeProjectResolution: (projectId) => {
      projectResolutionFences.delete(projectId);
      releaseProjectMutation(projectId);
    },
    abortProjectResolution: releaseProjectMutation,
    adoptProjectRecord: (record) => {
      assertOwner();
      if (reservedCopyTargetIds.has(record.project_id)) {
        throw new Error('This project copy is still being finalized. Retry opening it when recovery finishes.');
      }
      const result = deserializeProjectRecord(record);
      if (result.status === 'loaded') {
        syncEntries.set(record.project_id, {
          minimumCanvasSchemaVersion: record.minimum_canvas_schema_version,
          pushedDoc: serializeProjectDocumentV3Json(result.project).documentJson,
          revision: record.revision,
        });
        conflicts.delete(record.project_id);
        schemaRefusals.delete(record.project_id);
        if (recordRequiresDocumentUpgrade(record)) {
          projectsRequiringDocumentUpgrade.add(record.project_id);
          pendingProjectIds.add(record.project_id);
        }
      }
      return result;
    },
    clearWorkbench: () => {
      isTerminallyCleared = true;
      return enqueue(async () => {
        const failures: unknown[] = [];
        try {
          let summaries: ProjectSummaryDTO[] = [];
          try {
            summaries = await api.listProjects(owner.signal);
          } catch (error) {
            failures.push(error);
          }
          await mapConcurrent(summaries, 2, async (summary) => {
            try {
              await api.deleteProject(summary.project_id, owner.signal);
            } catch (error) {
              failures.push(error);
            }
          });
          try {
            await api.deleteSession(owner.signal);
          } catch (error) {
            failures.push(error);
          }
        } finally {
          assertOwner();
          const store = await getDraftStore();
          store.close();
          const session = await getEditorSessionForService();
          await session.release();
          const deletion = await deleteDatabase(owner.storageSuffix);
          const deletionResult = await awaitDatabaseDeletion(deletion, databaseDeleteTimeoutMs);
          if (deletionResult.kind !== 'deleted') {
            failures.push(new Error('Browser recovery data could not be deleted.'));
          }
          if (failures.length === 0) {
            seedProjectLibrary([], owner);
          }
          syncEntries.clear();
          conflicts.clear();
          schemaRefusals.clear();
          pendingProjectIds.clear();
          projectsRequiringDocumentUpgrade.clear();
          pendingBoardAssignments.clear();
          pendingRetargetHandoffs.clear();
          explicitRetargetSourcesAwaitingAck.clear();
          closedRetargetTargetsAwaitingAck.clear();
          retargetedProjects.clear();
          projectResolutionFences.clear();
          copyCaptureProjectIds.clear();
          reservedCopyTargetIds.clear();
          deletedProjectIds.clear();
          serverDeletedProjectIds.clear();
          volatileDrafts.clear();
          generations.clear();
          draftEditorSessionIds.clear();
          unopenableDrafts.clear();
          localDraftFailures.clear();
          localDraftStatus = 'ok';
          hasPending = false;
          sessionSavePending = false;
          lastKnownState = null;
          lastSessionJson = null;
        }
        if (failures.length > 0) {
          throw new AggregateError(failures, 'Workbench data could not be fully cleared.');
        }
      });
    },
    close,
    deleteRecoverableDraft: (projectId, editorSessionId, generation, updatedAt) =>
      enqueue(async () => {
        const store = await getDraftStore();
        const current = await store.get(projectId, editorSessionId);
        if (current.kind === 'missing' || current.kind === 'empty' || current.kind === 'retargeted') {
          unopenableDrafts.delete(`${projectId}\u0000${editorSessionId}`);
          reportSync(lastKnownState?.projects ?? []);
          return;
        }
        if (
          current.kind !== 'found' ||
          current.draft.generation !== generation ||
          current.draft.updatedAt !== updatedAt
        ) {
          throw new Error('The recoverable draft changed before it could be deleted. Refresh and retry.');
        }
        const owned = await getOwnedDraft(store, projectId, editorSessionId);
        if (!owned || owned.generation !== generation || owned.updatedAt !== updatedAt) {
          throw new Error('The recoverable draft changed before it could be deleted. Refresh and retry.');
        }
        const deleted = await store.delete(projectId, editorSessionId, writerToken);
        if (deleted.kind !== 'deleted') {
          throw new Error('The recoverable draft could not be deleted.');
        }
        unopenableDrafts.delete(`${projectId}\u0000${editorSessionId}`);
        reportSync(lastKnownState?.projects ?? []);
      }),
    deleteProjectOnServer: (projectId) =>
      enqueue(async () => {
        const resolvedProjectId = resolveProjectId(projectId);
        const hadDeleteFence = deletedProjectIds.has(resolvedProjectId);
        deletedProjectIds.add(resolvedProjectId);
        let serverDeleteCommitted = serverDeletedProjectIds.has(resolvedProjectId);
        try {
          const [store, session] = await Promise.all([getDraftStore(), getEditorSessionForService()]);
          const editorSessionId = getDraftEditorSessionId(resolvedProjectId, session.id);
          const preflight = await store.get(resolvedProjectId, editorSessionId);
          if (preflight.kind === 'unavailable' || preflight.kind === 'corrupt' || preflight.kind === 'retargeted') {
            throw new Error('Local project recovery must be available before deleting the server project.');
          }
          const ownedDraft =
            preflight.kind === 'found' ? await getOwnedDraft(store, resolvedProjectId, editorSessionId) : null;
          if (preflight.kind === 'found' && !ownedDraft) {
            throw new Error('This tab no longer owns the local project draft.');
          }
          try {
            await api.deleteProject(resolvedProjectId, owner.signal);
          } catch (error) {
            if (!isStatus(error, 404)) {
              throw error;
            }
          }
          serverDeleteCommitted = true;
          serverDeletedProjectIds.add(resolvedProjectId);
          if (ownedDraft) {
            const deleted = await store.delete(
              resolvedProjectId,
              editorSessionId,
              writerToken,
              issuedGenerations.get(resolvedProjectId)
            );
            if (deleted.kind !== 'deleted') {
              throw new Error(
                'The server project was deleted, but its local draft could not be removed. Retry deletion.'
              );
            }
          }
          // A journal from an earlier hidden page would otherwise bring the deleted project back on the next load.
          if (!(await settleJournaledDocument(resolvedProjectId, store))) {
            throw new Error(
              'The server project was deleted, but its local recovery could not be removed. Retry deletion.'
            );
          }
          volatileDrafts.delete(resolvedProjectId);
          draftEditorSessionIds.delete(resolvedProjectId);
          syncEntries.delete(resolvedProjectId);
          conflicts.delete(resolvedProjectId);
          projectsRequiringDocumentUpgrade.delete(resolvedProjectId);
          pendingProjectIds.delete(resolvedProjectId);
          for (const [key, handoff] of pendingRetargetHandoffs) {
            if (handoff.targetProjectId !== resolvedProjectId) {
              continue;
            }
            const acknowledged = await store.acknowledgeRetarget(
              handoff.projectId,
              handoff.editorSessionId,
              handoff.targetProjectId
            );
            if (acknowledged.kind === 'unavailable') {
              markLocalDraftFailure(`retarget:${key}`);
            } else if (acknowledged.kind === 'deleted') {
              clearLocalDraftFailure(`retarget:${key}`);
              pendingRetargetHandoffs.delete(key);
            }
          }
        } catch (error) {
          if (!serverDeleteCommitted && !hadDeleteFence) {
            deletedProjectIds.delete(resolvedProjectId);
          }
          throw error;
        }
      }),
    ensureProjectOnServer: (project) => {
      const assertProjectIdentityCurrent = () => {
        assertNotCleared();
        assertOwner();
        if (deletedProjectIds.has(project.id) || retargetedProjects.has(project.id)) {
          throw new ProjectFlushError('superseded');
        }
        if (conflicts.get(project.id)?.kind === 'deleted') {
          throw new ProjectFlushError('conflicted');
        }
      };
      // Upload provenance only needs the project's acknowledged identity; later edits stay with autosave. An
      // acknowledged project answers at once instead of waiting behind every queued save.
      if (syncEntries.has(project.id)) {
        return Promise.resolve().then(assertProjectIdentityCurrent);
      }
      // Uploads share one creation attempt, and fall back at once while a recent attempt's failure is fresh.
      const failure = projectEnsureFailures.get(project.id);
      if (failure && Date.now() - failure.at < PROJECT_ENSURE_FAILURE_REUSE_MS) {
        return Promise.resolve().then(() => {
          assertProjectIdentityCurrent();
          throw failure.error;
        });
      }
      projectEnsureFailures.delete(project.id);
      let ensure = projectEnsures.get(project.id);
      if (!ensure) {
        ensure = enqueue(async () => {
          assertProjectIdentityCurrent();
          if (!syncEntries.has(project.id)) {
            assertProjectFlushed(await pushProject(project));
          }
        }).then(
          () => {
            projectEnsures.delete(project.id);
            projectEnsureFailures.delete(project.id);
          },
          (error: unknown) => {
            projectEnsures.delete(project.id);
            projectEnsureFailures.set(project.id, { at: Date.now(), error });
            throw error;
          }
        );
        projectEnsures.set(project.id, ensure);
      }
      return ensure.then(assertProjectIdentityCurrent);
    },
    flushProjectToServer: (project) => {
      if (isTerminallyCleared) {
        return Promise.reject(new Error('Workbench persistence was cleared and must be reloaded.'));
      }
      return enqueue(() => pushProject(project));
    },
    getProjectDraftDocument: async (projectId) => {
      const { draft } = await requireDraft(projectId);
      return draft.documentJson;
    },
    getRecoverableDraftDocument: async (projectId, editorSessionId) => {
      const store = await getDraftStore();
      const result = await store.get(projectId, editorSessionId);
      return result.kind === 'found' ? result.draft.documentJson : null;
    },
    hasClosedSession: () => isSessionClosed,
    hasPendingChanges: () => hasPending,
    journalBeforeUnload: (state) => {
      const target = unloadJournalTarget;
      // Nothing of a closed session, a cleared workbench, or an ended account lifetime may be written.
      if (
        !target ||
        isClosed ||
        isTerminallyCleared ||
        isSessionClosed ||
        owner.accountId === null ||
        !isAccountScopeCurrent(owner)
      ) {
        return { kind: 'nothing-to-journal' };
      }
      const entries: ProjectUnloadJournalEntry[] = [];
      const retired: ProjectUnloadJournalKey[] = [];
      const skippedOversizedProjectIds: string[] = [];
      const journaledAt = Date.parse(now());
      let remainingBytes = PROJECT_UNLOAD_JOURNAL_MAX_BYTES;
      for (const inputProject of applyProjectResolutionFences(state).projects) {
        const project = resolveInputProject(inputProject);
        // Deletion owns this project's recovery: it must not come back from a journal.
        if (deletedProjectIds.has(project.id)) {
          continue;
        }
        const serialized = serializeProjectDocumentV3Json(project);
        const journaled = journaledDocuments.get(project.id);
        const isJournalStored = !retiredJournalProjectIds.has(project.id) && !settlingJournalProjectIds.has(project.id);
        if (serialized.documentJson === journaled?.documentJson && isJournalStored) {
          continue;
        }
        if (isInRecovery(project.id, serialized.documentJson)) {
          // Back to what recovery holds: the journal of a later state must not come back as a newer draft. The record
          // stays, so the next save also settles it, should this blind deletion be lost.
          if (journaled && isJournalStored) {
            retired.push([project.id, journaled.editorSessionId, writerToken, journaled.generation]);
          }
          continue;
        }
        if (serialized.byteSize > remainingBytes) {
          skippedOversizedProjectIds.push(project.id);
          continue;
        }
        remainingBytes -= serialized.byteSize;
        // A stage of this very document that is still in flight shares its generation: whichever lands first, the
        // lineage ends with one copy, and the acknowledgement of that stage also settles the journal.
        const staging = stagingDocuments.get(project.id);
        const entry = syncEntries.get(project.id);
        entries.push({
          accountId: owner.accountId,
          baseMinimumCanvasSchemaVersion:
            entry?.minimumCanvasSchemaVersion ?? getProjectCanvasSchemaRequirement(serialized.document),
          baseRevision: entry?.revision ?? null,
          documentByteSize: serialized.byteSize,
          documentJson: serialized.documentJson,
          documentSchemaVersion: PROJECT_DOCUMENT_SCHEMA_VERSION,
          editorSessionId: getDraftEditorSessionId(project.id, target.editorSessionId),
          generation:
            staging?.documentJson === serialized.documentJson ? staging.generation : nextGeneration(project.id),
          journaledAt,
          projectId: project.id,
          recordType: 'unload-journal',
          schemaVersion: PROJECT_UNLOAD_JOURNAL_SCHEMA_VERSION,
          writerToken,
        });
      }
      if (entries.length === 0 && retired.length === 0) {
        return skippedOversizedProjectIds.length > 0
          ? { kind: 'journaled', projectIds: [], skippedOversizedProjectIds, written: Promise.resolve(true) }
          : { kind: 'nothing-to-journal' };
      }
      const write = target.store.journalBeforeUnload({ entries, retired });
      if (write.kind !== 'started') {
        return { kind: 'unavailable' };
      }
      // Remembered at once, so becoming hidden right after `pagehide` does not write it again; an aborted write puts
      // back what was there, so the next hidden event writes again.
      const replaced = new Map<
        string,
        {
          after: ProjectUnloadJournalEntry | undefined;
          before: ProjectUnloadJournalEntry | undefined;
          wasRetired: boolean;
        }
      >();
      for (const [projectId] of retired) {
        const record = journaledDocuments.get(projectId);
        replaced.set(projectId, { after: record, before: record, wasRetired: false });
        retiredJournalProjectIds.add(projectId);
      }
      for (const entry of entries) {
        replaced.set(entry.projectId, {
          after: entry,
          before: journaledDocuments.get(entry.projectId),
          wasRetired: retiredJournalProjectIds.has(entry.projectId),
        });
        journaledDocuments.set(entry.projectId, entry);
        retiredJournalProjectIds.delete(entry.projectId);
        // Never reused: a journal settled at this generation must not swallow a later edit journaled next.
        issuedGenerations.set(entry.projectId, Math.max(issuedGenerations.get(entry.projectId) ?? 0, entry.generation));
      }
      const written = write.written.then((isWritten) => {
        if (!isWritten) {
          for (const [projectId, { after, before, wasRetired }] of replaced) {
            if (journaledDocuments.get(projectId) === after) {
              if (before) {
                journaledDocuments.set(projectId, before);
              } else {
                journaledDocuments.delete(projectId);
              }
              if (wasRetired) {
                retiredJournalProjectIds.add(projectId);
              } else {
                retiredJournalProjectIds.delete(projectId);
              }
            }
          }
        }
        return isWritten;
      });
      return {
        kind: 'journaled',
        projectIds: entries.map((entry) => entry.projectId),
        skippedOversizedProjectIds,
        written,
      };
    },
    hydrateProjectFromServer: (projectId, projectName = projectId) =>
      enqueue(() => {
        if (
          retargetedProjects.has(projectId) ||
          [...pendingRetargetHandoffs.values()].some((handoff) => handoff.projectId === projectId)
        ) {
          explicitRetargetSourcesAwaitingAck.add(projectId);
          retargetedProjects.delete(projectId);
          volatileDrafts.delete(projectId);
          generations.delete(projectId);
          pendingProjectIds.delete(projectId);
          draftEditorSessionIds.delete(projectId);
        }
        return hydrateProjectWithDraft(projectId, projectName);
      }),
    loadWorkbench(options) {
      assertNotCleared();
      if (loadPromise) {
        return loadPromise;
      }
      const loading = (async () => {
        assertOwner();
        clearLegacyStorage();
        const store = await getDraftStore();
        const session = await getEditorSessionForService();
        assertOwner();
        // Before this writer claims any lineage, so a journal is judged against the lineage its writer left behind;
        // before the backend is asked, so a journaled document can be exported even when it is unreachable. A journal
        // of a page that still runs (hidden, not unloaded) is left to that page, which settles it with its own saves.
        const reconciledJournal =
          owner.accountId === null
            ? ({ kind: 'unavailable' } as const)
            : await store.reconcileUnloadJournal(owner.accountId, Date.parse(now()), {
                isEditorSessionLive: (lineageEditorSessionId) => {
                  const editorSessionId = getLineageEditorSessionId(lineageEditorSessionId);
                  return editorSessionId === session.id
                    ? Promise.resolve(false)
                    : isOtherEditorSessionLive(editorSessionId);
                },
              });
        assertOwner();
        unloadJournalTarget = { editorSessionId: session.id, store };
        let summaries: ProjectSummaryDTO[];
        let sessionBlob: WorkbenchSessionBlob | null;
        let queueRunProjects: Awaited<ReturnType<typeof listQueueRunProjectIds>>;
        try {
          [summaries, sessionBlob, queueRunProjects] = await Promise.all([
            api.listProjects(owner.signal),
            api.loadSession(owner.signal),
            listQueueRunProjectIds(),
          ]);
        } catch (error) {
          throw new WorkbenchBackendUnavailableError(error);
        }
        assertOwner();
        syncEntries.clear();
        conflicts.clear();
        schemaRefusals.clear();
        projectsRequiringDocumentUpgrade.clear();
        generations.clear();
        pendingProjectIds.clear();
        pendingBoardAssignments.clear();
        retargetedProjects.clear();
        volatileDrafts.clear();
        unopenableDrafts.clear();
        pendingRetargetHandoffs.clear();
        explicitRetargetSourcesAwaitingAck.clear();
        closedRetargetTargetsAwaitingAck.clear();
        projectResolutionFences.clear();
        draftEditorSessionIds.clear();
        journaledDocuments.clear();
        retiredJournalProjectIds.clear();
        settlingJournalProjectIds.clear();
        localDraftFailures.clear();
        localDraftStatus = 'ok';
        // Without its journal, recovery works as it did before the journal existed: that is logged, not reported. An
        // entry the draft store could not take is a draft-store failure for its project, which its next stage clears.
        const journalReplacedDrafts = new Map<string, ProjectUnloadJournalReconciliation>();
        if (reconciledJournal.kind === 'unavailable') {
          logger.info({
            message: 'The unload journal is unavailable; recovery relies on staged drafts alone.',
            name: 'persistence.unload-journal-unavailable',
          });
        } else {
          for (const reconciled of reconciledJournal.outcomes) {
            if (reconciled.outcome === 'corrupt' || reconciled.outcome === 'quota') {
              markLocalDraftFailure(projectDraftFailureKey(reconciled.projectId));
            } else if (reconciled.replacedDraft) {
              journalReplacedDrafts.set(reconciled.projectId, reconciled);
            }
          }
          if (reconciledJournal.outcomes.length > 0) {
            logger.info({
              context: {
                outcomes: reconciledJournal.outcomes.map(({ outcome, projectId }) => ({ outcome, projectId })),
              },
              message: 'Reconciled the unload journal.',
              name: 'persistence.unload-journal-reconciled',
            });
          }
        }
        for (const [projectId, editorSessionId] of Object.entries(sessionBlob?.draftEditorSessionIds ?? {})) {
          draftEditorSessionIds.set(projectId, editorSessionId);
        }
        hasPending = false;
        sessionSavePending = false;
        seedProjectLibrary(summaries, owner);
        const [draftCandidates, retargetHandoffs] = await Promise.all([
          listDraftProjectCandidates(store),
          listRetargetHandoffs(store),
        ]);
        const draftCandidatesByProject = groupDraftCandidatesByProject(draftCandidates);
        const summaryById = new Map(summaries.map((summary) => [summary.project_id, summary]));
        const handoffRecordsById = new Map<string, ProjectRecordDTO>();
        const handoffTargetIds = new Set<string>();
        for (const handoff of retargetHandoffs) {
          if (!summaryById.has(handoff.targetProjectId)) {
            try {
              const record = await api.getProject(handoff.targetProjectId, owner.signal);
              assertOwner();
              const summary = toProjectSummary(record);
              summaryById.set(record.project_id, summary);
              handoffRecordsById.set(record.project_id, record);
              upsertProjectSummary(
                {
                  id: summary.project_id,
                  minimumCanvasSchemaVersion: summary.minimum_canvas_schema_version,
                  name: summary.name,
                  revision: summary.revision,
                },
                owner
              );
            } catch (error) {
              if (!isStatus(error, 404)) {
                throw new WorkbenchBackendUnavailableError(error);
              }
              const result = await store.acknowledgeRetarget(
                handoff.projectId,
                handoff.editorSessionId,
                handoff.targetProjectId
              );
              if (result.kind === 'unavailable') {
                markLocalDraftFailure(`retarget:${retargetHandoffKey(handoff)}`);
              }
              continue;
            }
          }
          pendingRetargetHandoffs.set(retargetHandoffKey(handoff), handoff);
          handoffTargetIds.add(handoff.targetProjectId);
        }
        if (options?.openProjectId && retargetHandoffs.some((handoff) => handoff.projectId === options.openProjectId)) {
          explicitRetargetSourcesAwaitingAck.add(options.openProjectId);
          draftEditorSessionIds.delete(options.openProjectId);
          isolateUnopenableDraft(options.openProjectId, session.id);
        }
        const sessionProjectIds = sessionBlob
          ? (sessionBlob.openProjectIds ?? summaries.map((summary) => summary.project_id))
          : summaries.slice(0, 1).map((summary) => summary.project_id);
        const alreadyRequestedProjectIds = new Set([
          ...sessionProjectIds,
          ...handoffTargetIds,
          ...(options?.openProjectId ? [options.openProjectId] : []),
        ]);
        const queueRunProjectCandidates =
          queueRunProjects.kind === 'available'
            ? [...new Set(queueRunProjects.projectIds)]
                .filter((projectId) => !alreadyRequestedProjectIds.has(projectId))
                .sort((left, right) => left.localeCompare(right))
            : [];
        const queueRunProjectIds = queueRunProjectCandidates.slice(0, autoOpenQueueRunProjectLimit);
        const recoverableQueueProjects: QueueRecoveryProject[] = queueRunProjectCandidates
          .slice(autoOpenQueueRunProjectLimit)
          .map((projectId) => ({ projectId, reason: 'open-limit' }));
        const alwaysOpenProjectIds = new Set([
          ...sessionProjectIds,
          ...queueRunProjectIds,
          ...handoffTargetIds,
          ...(options?.openProjectId ? [options.openProjectId] : []),
        ]);
        const draftIds = selectAutoOpenDraftProjectIds(draftCandidates, alwaysOpenProjectIds);
        const loadRetargets = new Map<string, string>();
        const previousEditorSessionId = sessionBlob?.editorSessionId;
        for (const projectId of sessionProjectIds) {
          if (!previousEditorSessionId || projectId === options?.openProjectId) {
            continue;
          }
          const result = await store.get(
            projectId,
            sessionBlob?.draftEditorSessionIds?.[projectId] ?? previousEditorSessionId
          );
          if (result.kind !== 'retargeted') {
            continue;
          }
          const target = summaryById.get(result.projectId);
          if (target) {
            loadRetargets.set(projectId, result.projectId);
            retargetedProjects.set(projectId, {
              boardId: target.board_id,
              name: target.name,
              projectId: target.project_id,
              sourceName: getCopySourceProjectName(target.name),
            });
          }
        }
        const requestedIds = new Set([
          ...sessionProjectIds.map((projectId) => loadRetargets.get(projectId) ?? projectId),
          ...queueRunProjectIds,
          ...draftIds,
          ...handoffTargetIds,
          ...(options?.openProjectId ? [options.openProjectId] : []),
        ]);
        const loadedProjects = new Map<string, Project>();
        const refusedProjects: RefusedWorkbenchProject[] = [];

        await forEachConcurrent([...requestedIds], 1, async (projectId) => {
          const draft = await adoptNewestDraft(
            store,
            projectId,
            session.id,
            draftCandidatesByProject.get(projectId) ?? []
          );
          if (draft) {
            noteGeneration(projectId, draft.generation);
            volatileDrafts.set(projectId, draft);
          }
          const summary = summaryById.get(projectId);
          const summaryIsIncompatible = Boolean(
            summary && !isCanvasSchemaVersionSupported(summary.minimum_canvas_schema_version)
          );
          let record: ProjectRecordDTO | null = null;
          if (summary && isCanvasSchemaVersionSupported(summary.minimum_canvas_schema_version)) {
            try {
              record = handoffRecordsById.get(projectId) ?? (await api.getProject(projectId, owner.signal));
            } catch (error) {
              if (!isStatus(error, 404)) {
                throw new WorkbenchBackendUnavailableError(error);
              }
            }
          } else if (summary) {
            refusedProjects.push({
              projectId,
              projectName: summary.name,
              raw: null,
              refusal: {
                raw: null,
                scope: 'document',
                status: 'unsupported-version',
                version: summary.minimum_canvas_schema_version,
              },
              source: 'canvas',
            });
          }

          let loadedDraft: LoadedDraft | null = null;
          if (draft) {
            const project = loadDraftProject(draft);
            if (project) {
              loadedDraft = { draft, project };
            } else {
              try {
                const raw = JSON.parse(draft.documentJson) as unknown;
                if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
                  const draftLoad = deserializeProjectDocument(raw as Record<string, unknown>);
                  if (draftLoad.status === 'refused') {
                    refusedProjects.push(draftLoad.refused);
                    rememberUnopenableDraft(draft);
                    isolateUnopenableDraft(projectId, session.id);
                    generations.delete(projectId);
                    volatileDrafts.delete(projectId);
                  }
                }
              } catch {
                // The draft store retains damaged rows for explicit recovery cleanup.
              }
            }
          }
          if (!record) {
            if (!loadedDraft) {
              return;
            }
            loadedDraft.draft = await alignLoadedDraftWithProject(loadedDraft.draft, loadedDraft.project, store);
            if (summaryIsIncompatible) {
              const minimumCanvasSchemaVersion = summary!.minimum_canvas_schema_version;
              const refusal: ProjectSchemaRefusal = {
                kind: 'canvas',
                maxCanvasSchemaVersion: MAX_SUPPORTED_CANVAS_SCHEMA_VERSION,
                minimumCanvasSchemaVersion,
              };
              schemaRefusals.set(projectId, refusal);
              await store.settleSchemaRefusal(
                projectId,
                loadedDraft.draft.editorSessionId,
                writerToken,
                loadedDraft.draft.generation,
                refusal
              );
              pendingProjectIds.add(projectId);
              hasPending = true;
            } else if (loadedDraft.draft.baseRevision !== null) {
              await markConflict(
                projectId,
                loadedDraft.draft.generation,
                { kind: 'deleted' },
                store,
                loadedDraft.draft.editorSessionId
              );
            } else {
              pendingProjectIds.add(projectId);
              hasPending = true;
            }
            loadedProjects.set(projectId, loadedDraft.project);
            return;
          }

          const serverLoad = deserializeProjectRecord(record);
          if (serverLoad.status === 'refused') {
            refusedProjects.push(serverLoad.refused);
            if (loadedDraft) {
              const localProject = withAuthoritativeProjectBoard(loadedDraft.project, record.board_id);
              loadedDraft.draft = await alignLoadedDraftWithProject(loadedDraft.draft, localProject, store);
              const refusal: ProjectSchemaRefusal =
                serverLoad.refused.source === 'project-document' &&
                serverLoad.refused.refusal.status === 'unsupported-version'
                  ? toDocumentSchemaRefusal(serverLoad.refused.refusal.version)
                  : serverLoad.refused.source === 'canvas' &&
                      serverLoad.refused.refusal.status === 'unsupported-version'
                    ? toCanvasSchemaRefusal({
                        maxCanvasSchemaVersion: MAX_SUPPORTED_CANVAS_SCHEMA_VERSION,
                        minimumCanvasSchemaVersion: serverLoad.refused.refusal.version,
                      })
                    : { kind: 'invalid-server-document' };
              schemaRefusals.set(projectId, refusal);
              await store.settleSchemaRefusal(
                projectId,
                loadedDraft.draft.editorSessionId,
                writerToken,
                loadedDraft.draft.generation,
                refusal
              );
              volatileDrafts.set(projectId, toSchemaRefusedProjectDraft(loadedDraft.draft, refusal));
              pendingProjectIds.add(projectId);
              hasPending = true;
              loadedProjects.set(projectId, localProject);
            }
            return;
          }
          if (serverLoad.status !== 'loaded') {
            return;
          }
          const serverJson = serializeProjectDocumentV3Json(serverLoad.project).documentJson;
          syncEntries.set(projectId, {
            minimumCanvasSchemaVersion: record.minimum_canvas_schema_version,
            pushedDoc: serverJson,
            revision: record.revision,
          });
          if (recordRequiresDocumentUpgrade(record)) {
            projectsRequiringDocumentUpgrade.add(projectId);
            pendingProjectIds.add(projectId);
            hasPending = true;
          }
          if (!loadedDraft) {
            loadedProjects.set(projectId, serverLoad.project);
            return;
          }
          const localProject = withAuthoritativeProjectBoard(loadedDraft.project, record.board_id);
          loadedDraft.draft = await alignLoadedDraftWithProject(loadedDraft.draft, localProject, store);
          if (loadedDraft.draft.documentJson === serverJson) {
            await store.settleAcknowledgement(
              projectId,
              loadedDraft.draft.editorSessionId,
              writerToken,
              loadedDraft.draft.generation,
              record.revision,
              record.minimum_canvas_schema_version
            );
            pendingProjectIds.delete(projectId);
            loadedProjects.set(projectId, serverLoad.project);
            return;
          }
          // The journal replaced a staged draft whose save reached the server without its answer reaching the page:
          // the server holds exactly that document, so the journal is its successor, as if the answer had arrived.
          const replaced = journalReplacedDrafts.get(projectId)?.replacedDraft;
          if (
            replaced?.documentJson === serverJson &&
            journalReplacedDrafts.get(projectId)?.editorSessionId === loadedDraft.draft.editorSessionId &&
            loadedDraft.draft.state === 'dirty' &&
            loadedDraft.draft.baseRevision !== record.revision
          ) {
            const rebased = await store.settleAcknowledgement(
              projectId,
              loadedDraft.draft.editorSessionId,
              writerToken,
              replaced.generation,
              record.revision,
              record.minimum_canvas_schema_version
            );
            if (rebased.kind === 'rebased') {
              loadedDraft.draft = rebased.draft;
              volatileDrafts.set(projectId, rebased.draft);
            }
          }
          if (
            loadedDraft.draft.state === 'schema-refused' &&
            loadedDraft.draft.refusal.kind === 'canvas' &&
            !isCanvasSchemaVersionSupported(loadedDraft.draft.refusal.minimumCanvasSchemaVersion)
          ) {
            schemaRefusals.set(projectId, loadedDraft.draft.refusal);
            pendingProjectIds.add(projectId);
            hasPending = true;
          } else if (loadedDraft.draft.state === 'conflict' || loadedDraft.draft.baseRevision !== record.revision) {
            const conflict =
              loadedDraft.draft.state === 'conflict'
                ? loadedDraft.draft.conflict
                : ({ kind: 'revision', serverRevision: record.revision } as const);
            await markConflict(
              projectId,
              loadedDraft.draft.generation,
              conflict,
              store,
              loadedDraft.draft.editorSessionId
            );
          } else {
            if (loadedDraft.draft.state === 'schema-refused') {
              await store.resumeSchemaRefused(
                projectId,
                loadedDraft.draft.editorSessionId,
                writerToken,
                loadedDraft.draft.generation
              );
              volatileDrafts.set(projectId, toDirtyProjectDraft(loadedDraft.draft, {}));
            }
            pendingProjectIds.add(projectId);
            hasPending = true;
          }
          loadedProjects.set(projectId, localProject);
        });
        assertOwner();

        for (const projectId of queueRunProjectIds) {
          if (!loadedProjects.has(projectId)) {
            recoverableQueueProjects.push({ projectId, reason: 'project-unavailable' });
          }
        }

        const projects = [...requestedIds].flatMap((projectId) => {
          const project = loadedProjects.get(projectId);
          return project ? [project] : [];
        });

        const account = normalizeWorkbenchAccount(sessionBlob?.account ?? createInitialWorkbenchState().account);
        if (options?.createNew || projects.length === 0) {
          projects.push(createDraftProject(projects, account));
          pendingProjectIds.add(projects.at(-1)!.id);
          hasPending = true;
        }
        const requestedActive = options?.createNew
          ? projects.at(-1)!.id
          : options?.openProjectId
            ? options.openProjectId
            : sessionBlob?.activeProjectId
              ? (loadRetargets.get(sessionBlob.activeProjectId) ?? sessionBlob.activeProjectId)
              : undefined;
        const activeProjectId = projects.some((project) => project.id === requestedActive)
          ? requestedActive!
          : projects[0]!.id;
        const state: WorkbenchState = {
          ...createInitialWorkbenchState(),
          account,
          activeProjectId,
          projects,
        };
        lastKnownState = state;
        lastSessionJson = sessionBlob ? JSON.stringify(sessionBlob) : null;
        if (await persistCurrentSession(state, session.id)) {
          await acknowledgeAwaitingExplicitRetargets();
          await acknowledgeCommittedRetargets(state);
        }
        reportSync(projects);
        const snapshot = toSnapshot(state, now());
        snapshot.refusedProjects = refusedProjects;
        return {
          ...snapshot,
          conflicts: [...conflicts.values()],
          localDraftStatus,
          queueRecovery: {
            projects: recoverableQueueProjects.filter(({ projectId }) => !loadedProjects.has(projectId)),
            status: queueRunProjects.kind,
          },
        };
      })();
      loadPromise = loading;
      void loading.catch(() => {
        if (loadPromise === loading) {
          loadPromise = null;
        }
      });
      return loading;
    },
    listRecoverableDrafts: async (options = {}) => {
      const store = await getDraftStore();
      const page = await store.list({ ...(options.after ? { after: options.after } : {}), limit: options.limit ?? 50 });
      if (page.kind !== 'available') {
        return { kind: 'unavailable' };
      }
      return {
        items: page.items.flatMap((item) =>
          item.state === 'corrupt' ||
          item.documentByteSize === null ||
          item.generation === null ||
          item.updatedAt === null
            ? []
            : [
                {
                  documentByteSize: item.documentByteSize,
                  editorSessionId: item.editorSessionId,
                  generation: item.generation,
                  projectId: item.projectId,
                  state: item.state,
                  updatedAt: item.updatedAt,
                },
              ]
        ),
        kind: 'available',
        nextCursor: page.nextCursor,
      };
    },
    markProjectDeleted: (projectId) => {
      deletedProjectIds.add(resolveProjectId(projectId));
    },
    persistEmptySession: (state) => {
      if (isTerminallyCleared) {
        return Promise.reject(new Error('Workbench persistence was cleared and must be reloaded.'));
      }
      // Set before the write is queued, so no save requested meanwhile can put the project back.
      isSessionClosed = true;
      return enqueue(async () => {
        const emptyState = { ...state, activeProjectId: '', projects: [] };
        const session = await getEditorSessionForService();
        const draftLineages = serializeDraftEditorSessionIds();
        await api.saveSession(emptyState, session.id, draftLineages, owner.signal);
        lastSessionJson = serializeSessionBlob(emptyState, session.id, draftLineages);
        lastKnownState = emptyState;
        sessionSavePending = false;
        for (const handoff of pendingRetargetHandoffs.values()) {
          closedRetargetTargetsAwaitingAck.add(handoff.targetProjectId);
        }
        await acknowledgeCommittedRetargets(emptyState);
        recomputeHasPending();
      });
    },
    resolveConflictDiscard: async (projectId) => {
      const mutationLock = await getProjectMutationLock(projectId);
      if (mutationLock.kind === 'contended') {
        throw new Error('Wait for active queue runs to finish before discarding this project.');
      }
      if (mutationLock.kind === 'unavailable') {
        throw new Error('Discard is unavailable because cross-tab coordination could not be established.');
      }
      try {
        await assertNoDurableQueueRuns(projectId);
      } catch (error) {
        await mutationLock.release();
        throw error;
      }
      projectResolutionFences.set(projectId, { kind: 'pending' });
      return enqueue(async () => {
        await deleteDraft(projectId);
        conflicts.delete(projectId);
        schemaRefusals.delete(projectId);
        pendingProjectIds.delete(projectId);
        draftEditorSessionIds.delete(projectId);
        recomputeHasPending();
        resolveProjectSyncConflict(projectId, undefined, hasPending);
        projectResolutionFences.set(projectId, { kind: 'remove' });
        if (lastKnownState) {
          lastKnownState = applyProjectResolutionFences(lastKnownState);
        }
        projectMutationLocks.set(projectId, mutationLock);
      }).catch((error) => {
        projectResolutionFences.delete(projectId);
        void mutationLock.release().catch(() => undefined);
        throw error;
      });
    },
    resolveConflictSaveAsNew: async (inputProject) => {
      if (hasActiveQueueRuns(inputProject)) {
        throw new Error('Wait for active queue runs to finish before saving this project as new.');
      }
      if (isTerminallyCleared) {
        throw new Error('Workbench persistence was cleared and must be reloaded.');
      }
      if (copyCaptureProjectIds.has(inputProject.id)) {
        throw new Error('A project copy is already being saved.');
      }
      copyCaptureProjectIds.add(inputProject.id);
      const invocationGeneration = generations.get(inputProject.id) ?? 0;
      const mutationLock = await getProjectMutationLock(inputProject.id).catch((error) => {
        copyCaptureProjectIds.delete(inputProject.id);
        throw error;
      });
      if (mutationLock.kind === 'contended') {
        copyCaptureProjectIds.delete(inputProject.id);
        throw new Error('Wait for active queue runs to finish before saving this project as new.');
      }
      if (mutationLock.kind === 'unavailable') {
        copyCaptureProjectIds.delete(inputProject.id);
        throw new Error('Saving as new is unavailable because cross-tab coordination could not be established.');
      }
      try {
        await assertNoDurableQueueRuns(inputProject.id);
      } catch (error) {
        copyCaptureProjectIds.delete(inputProject.id);
        await mutationLock.release();
        throw error;
      }
      let reservedTargetId: string | null = null;
      let handedOffMutationLock = false;
      return enqueue(async () => {
        const projectId = inputProject.id;
        const currentDocument = serializeProjectDocumentV3Json(inputProject);
        let owned = await requireDraft(projectId);
        if (
          owned.draft.documentJson !== currentDocument.documentJson &&
          owned.draft.generation <= invocationGeneration
        ) {
          await stageProject(inputProject, currentDocument);
          owned = await requireDraft(projectId);
        }
        const { editorSessionId, store } = owned;
        const durable = await store.get(projectId, editorSessionId);
        if (
          durable.kind !== 'found' ||
          durable.draft.writerToken !== writerToken ||
          durable.draft.generation !== owned.draft.generation ||
          durable.draft.documentJson !== owned.draft.documentJson
        ) {
          markLocalDraftFailure(projectDraftFailureKey(projectId));
          throw new Error('Save as new requires an up-to-date local recovery draft. Free browser storage and retry.');
        }
        const draft = durable.draft;
        const copyProjectGeneration = draft.generation;
        const sourceDocument = JSON.parse(draft.documentJson) as Record<string, unknown>;
        const sourceProjectName = typeof sourceDocument.name === 'string' ? sourceDocument.name : inputProject.name;
        const conflict = conflicts.get(projectId);
        if (!conflict) {
          throw new Error('The project no longer has a conflict to resolve.');
        }
        const session = await getEditorSessionForService();
        if (!lastKnownState || !(await persistCurrentSession(lastKnownState, session.id))) {
          throw new Error('The project session could not be secured for retry-safe recovery.');
        }
        const volatile = volatileDrafts.get(projectId);
        const volatileReservation: ProjectDraftCopyReservation | null =
          volatile?.copyProjectId &&
          volatile.copyDocumentJson &&
          volatile.copyDocumentByteSize !== undefined &&
          volatile.copyProjectGeneration !== undefined &&
          volatile.copyProjectMinimumCanvasSchemaVersion !== undefined &&
          volatile.copyProjectName
            ? {
                copyDocumentByteSize: volatile.copyDocumentByteSize,
                copyDocumentJson: volatile.copyDocumentJson,
                copyProjectGeneration: volatile.copyProjectGeneration,
                copyProjectId: volatile.copyProjectId,
                copyProjectMinimumCanvasSchemaVersion: volatile.copyProjectMinimumCanvasSchemaVersion,
                copyProjectName: volatile.copyProjectName,
                copySourceProjectName:
                  volatile.copySourceProjectName ?? getCopySourceProjectName(volatile.copyProjectName),
              }
            : null;
        const sourceEntry = syncEntries.get(projectId);
        const proposedReservation: ProjectDraftCopyReservation =
          volatileReservation ??
          (await (async () => {
            const copyProjectId = await createDeterministicProjectId(
              [
                'workbench-conflict-copy-v1',
                projectId,
                String(draft.baseRevision),
                conflict.kind,
                conflict.kind === 'revision' ? String(conflict.serverRevision) : '',
                draft.documentJson,
              ].join('\u0000')
            );
            const name = `${sourceProjectName} (copy)`;
            // A draft staged by an older client copies forward in the current schema.
            const migratedSource = migrateProjectDocument(sourceDocument);
            if (migratedSource.status === 'malformed') {
              throw new Error('The project draft cannot be copied: its document is malformed.');
            }
            const document = { ...migratedSource.document, id: copyProjectId, name };
            const documentJson = JSON.stringify(document);
            const documentByteSize = getUtf8ByteSize(documentJson);
            if (documentByteSize > PROJECT_DOCUMENT_MAX_BYTES) {
              throw new ProjectDocumentTooLargeError(documentByteSize);
            }
            return {
              copyDocumentByteSize: documentByteSize,
              copyDocumentJson: documentJson,
              copyProjectGeneration,
              copyProjectId,
              copyProjectMinimumCanvasSchemaVersion: Math.max(
                sourceEntry?.minimumCanvasSchemaVersion ?? draft.baseMinimumCanvasSchemaVersion ?? 1,
                getProjectCanvasSchemaRequirement(document)
              ),
              copyProjectName: name,
              copySourceProjectName: sourceProjectName,
            };
          })());
        if (isCopyTargetOpen(proposedReservation.copyProjectId)) {
          throw new Error('The project copy is already open. Close it before retrying this resolution.');
        }
        const storedReservation = await store.reserveCopyIdentity(
          projectId,
          editorSessionId,
          writerToken,
          proposedReservation
        );
        if (storedReservation.kind !== 'reserved') {
          if (storedReservation.kind === 'quota' || storedReservation.kind === 'unavailable') {
            markLocalDraftFailure(projectDraftFailureKey(projectId));
          }
          throw new Error('A retry-safe project identity could not be reserved.');
        }
        const reservation = storedReservation;
        if (volatile) {
          volatileDrafts.set(projectId, { ...volatile, ...reservation });
        }
        const copyProjectId = reservation.copyProjectId;
        reservedTargetId = copyProjectId;
        reservedCopyTargetIds.add(copyProjectId);
        const name = reservation.copyProjectName;
        const document = JSON.parse(reservation.copyDocumentJson) as Record<string, unknown>;
        if (
          getUtf8ByteSize(reservation.copyDocumentJson) !== reservation.copyDocumentByteSize ||
          document.id !== copyProjectId ||
          document.name !== name ||
          migrateProjectDocument(document).status === 'malformed'
        ) {
          throw new Error('The reserved project copy is damaged.');
        }
        const record = await saveAsNew({
          copyProjectId,
          document,
          minimumCanvasSchemaVersion: reservation.copyProjectMinimumCanvasSchemaVersion,
          name,
          owner,
          sourceProjectId: projectId,
        });
        assertOwner();
        if (isCopyTargetOpen(copyProjectId)) {
          throw new Error(
            'The project copy opened before recovery finished. Close it before retrying this resolution.'
          );
        }
        const settlement = await store.retargetAcknowledgedCopy({
          acknowledgedRevision: record.revision,
          copyProjectId,
          editorSessionId,
          projectId,
          retargetDocument: (documentJson) => retargetCopyDocument(documentJson, reservation),
          sentGeneration: reservation.copyProjectGeneration,
          writerToken,
        });
        if (settlement.kind !== 'retargeted') {
          if (settlement.kind === 'quota' || settlement.kind === 'unavailable') {
            markLocalDraftFailure(projectDraftFailureKey(projectId));
          }
          throw new Error('The acknowledged copy could not replace its local draft. Retry to finish recovery.');
        }
        clearLocalDraftFailure(projectDraftFailureKey(projectId));
        const latestVolatile = volatileDrafts.get(projectId);
        const volatileRetargeted =
          latestVolatile && latestVolatile.generation > reservation.copyProjectGeneration
            ? toDirtyProjectDraft(latestVolatile, {
                baseRevision: record.revision,
                copyDocumentByteSize: undefined,
                copyDocumentJson: undefined,
                copyProjectGeneration: undefined,
                copyProjectId: undefined,
                copyProjectMinimumCanvasSchemaVersion: undefined,
                copyProjectName: undefined,
                copySourceProjectName: undefined,
                documentJson: retargetCopyDocument(latestVolatile.documentJson, reservation),
                projectId: copyProjectId,
              })
            : null;
        volatileDrafts.delete(projectId);
        if (volatileRetargeted) {
          volatileDrafts.set(copyProjectId, volatileRetargeted);
        }
        const acknowledgedJson = JSON.stringify(record.data);
        syncEntries.delete(projectId);
        conflicts.delete(projectId);
        projectsRequiringDocumentUpgrade.delete(projectId);
        schemaRefusals.delete(projectId);
        pendingProjectIds.delete(projectId);
        retargetedProjects.delete(projectId);
        syncEntries.set(copyProjectId, {
          minimumCanvasSchemaVersion: record.minimum_canvas_schema_version,
          pushedDoc: acknowledgedJson,
          revision: record.revision,
        });
        upsertProjectSummary(
          {
            id: copyProjectId,
            minimumCanvasSchemaVersion: record.minimum_canvas_schema_version,
            name,
            revision: record.revision,
          },
          owner
        );
        recordProjectCover(copyProjectId, selectCoverImageName(record.data), owner);
        const durableRetargeted = settlement.kind === 'retargeted' ? settlement.draft : null;
        const survivingDraft =
          durableRetargeted && volatileRetargeted
            ? durableRetargeted.generation >= volatileRetargeted.generation
              ? durableRetargeted
              : volatileRetargeted
            : (durableRetargeted ?? volatileRetargeted);
        if (survivingDraft) {
          noteGeneration(copyProjectId, survivingDraft.generation);
          pendingProjectIds.add(copyProjectId);
        } else {
          generations.delete(projectId);
        }
        pendingBoardAssignments.set(copyProjectId, { boardId: record.board_id, projectId: copyProjectId });
        draftEditorSessionIds.delete(projectId);
        if (editorSessionId !== session.id) {
          draftEditorSessionIds.set(copyProjectId, editorSessionId);
        }
        if (settlement.kind === 'retargeted') {
          const handoff: ProjectDraftRetargetHandoff = {
            editorSessionId,
            projectId,
            revision: record.revision,
            targetProjectId: copyProjectId,
            updatedAt: Date.parse(now()),
          };
          pendingRetargetHandoffs.set(retargetHandoffKey(handoff), handoff);
        }
        retargetedProjects.set(projectId, {
          boardId: record.board_id,
          name,
          projectId: copyProjectId,
          sourceName: reservation.copySourceProjectName,
        });
        recomputeHasPending();
        resolveProjectSyncConflict(projectId, { projectId: copyProjectId, revision: record.revision }, hasPending);
        const loadedCopy = deserializeProjectRecord(record);
        const visible = survivingDraft
          ? loadDraftProject(survivingDraft)
          : loadedCopy.status === 'loaded'
            ? loadedCopy.project
            : null;
        if (!visible) {
          throw new Error('The copied project could not be loaded.');
        }
        return {
          boardId: record.board_id,
          name: visible.name,
          project: withAuthoritativeProjectBoard(visible, record.board_id),
          sourceName: reservation.copySourceProjectName,
          sourceProjectId: projectId,
          targetProjectId: copyProjectId,
        };
      })
        .then((result) => {
          projectMutationLocks.set(inputProject.id, mutationLock);
          handedOffMutationLock = true;
          return result;
        })
        .finally(async () => {
          copyCaptureProjectIds.delete(inputProject.id);
          if (reservedTargetId) {
            reservedCopyTargetIds.delete(reservedTargetId);
          }
          if (!handedOffMutationLock) {
            await mutationLock.release();
          }
        });
    },
    resolveConflictUseServer: (projectId) => {
      projectResolutionFences.set(projectId, { kind: 'pending' });
      return enqueue(async () => {
        const record = await api.getProject(projectId, owner.signal);
        assertOwner();
        const result = deserializeProjectRecord(record);
        if (result.status !== 'loaded') {
          projectResolutionFences.delete(projectId);
          return result;
        }
        await deleteDraft(projectId);
        const documentJson = serializeProjectDocumentV3Json(result.project).documentJson;
        syncEntries.set(projectId, {
          minimumCanvasSchemaVersion: record.minimum_canvas_schema_version,
          pushedDoc: documentJson,
          revision: record.revision,
        });
        conflicts.delete(projectId);
        schemaRefusals.delete(projectId);
        pendingProjectIds.delete(projectId);
        retargetedProjects.delete(projectId);
        draftEditorSessionIds.delete(projectId);
        recomputeHasPending();
        resolveProjectSyncConflict(projectId, { projectId, revision: record.revision }, hasPending);
        projectResolutionFences.set(projectId, { kind: 'replace', project: result.project });
        if (lastKnownState) {
          lastKnownState = applyProjectResolutionFences(lastKnownState);
        }
        return result;
      }).catch((error) => {
        projectResolutionFences.delete(projectId);
        throw error;
      });
    },
    releaseMutationLocks: () => {
      for (const release of releaseAllMutationLocks()) {
        void release.catch(() => undefined);
      }
    },
    releaseProjectSync: (projectId) => {
      projectEnsureFailures.delete(projectId);
      if ([...pendingRetargetHandoffs.values()].some((handoff) => handoff.targetProjectId === projectId)) {
        closedRetargetTargetsAwaitingAck.add(projectId);
      }
      if (retargetedProjects.has(projectId)) {
        conflicts.delete(projectId);
        schemaRefusals.delete(projectId);
        generations.delete(projectId);
        pendingProjectIds.delete(projectId);
        return;
      }
      syncEntries.delete(projectId);
      conflicts.delete(projectId);
      schemaRefusals.delete(projectId);
      generations.delete(projectId);
      pendingProjectIds.delete(projectId);
      retargetedProjects.delete(projectId);
      draftEditorSessionIds.delete(projectId);
    },
    retain: () => {
      if (isClosed) {
        throw new Error('The project persistence service is closed.');
      }
      retainCount += 1;
      releaseGeneration += 1;
      let isReleased = false;
      return () => {
        if (isReleased) {
          return Promise.resolve();
        }
        isReleased = true;
        retainCount -= 1;
        const generation = ++releaseGeneration;
        return new Promise<void>((resolve) => {
          queueMicrotask(() => {
            if (retainCount === 0 && releaseGeneration === generation) {
              void close().then(resolve);
            } else {
              resolve();
            }
          });
        });
      };
    },
    reopenSession: (state) => {
      isSessionClosed = false;
      const saved = service.saveWorkbench(state);
      for (const listener of sessionReopenedListeners) {
        listener();
      }
      return saved;
    },
    subscribeSessionReopened: (listener) => {
      sessionReopenedListeners.add(listener);
      return () => {
        sessionReopenedListeners.delete(listener);
      };
    },
    saveWorkbench: (state) => {
      if (isTerminallyCleared) {
        return Promise.reject(new Error('Workbench persistence was cleared and must be reloaded.'));
      }
      assertOwner();
      lastKnownState = applyProjectResolutionFences(state);
      const capture = mapConcurrent(
        state.projects.filter((project) => copyCaptureProjectIds.has(project.id)),
        2,
        async (inputProject) => {
          const plan = createProjectSavePlan(inputProject);
          if (!plan.needsStage) {
            return;
          }
          try {
            await stageProject(plan.project, plan.serialized);
          } catch {
            return;
          }
        }
      );

      return enqueue(async () => {
        await capture;
        assertOwner();
        hasPending = sessionSavePending;
        const currentState = lastKnownState ?? state;
        const plans = currentState.projects.map(createProjectSavePlan);
        const stagedProjects = new Map<string, Awaited<ReturnType<typeof stageProject>>>();
        await mapConcurrent(plans, 2, async (plan) => {
          if (!plan.needsStage || deletedProjectIds.has(plan.project.id)) {
            return;
          }
          try {
            stagedProjects.set(plan.project.id, await stageProject(plan.project, plan.serialized));
          } catch {
            return;
          }
        });
        let shouldRetry = false;
        const session = await getEditorSessionForService();
        if (!(await persistCurrentSession(currentState, session.id))) {
          shouldRetry = true;
        } else {
          await acknowledgeAwaitingExplicitRetargets();
          await acknowledgeCommittedRetargets(currentState);
        }
        let firstPushError: unknown = null;
        for (const plan of plans) {
          try {
            const outcome = await pushProjectPlan(plan, stagedProjects.get(plan.project.id));
            if (outcome.kind !== 'acknowledged' && outcome.kind !== 'superseded') {
              hasPending = true;
            }
            if (outcome.kind === 'unsynced') {
              shouldRetry = true;
            }
          } catch (error) {
            firstPushError ??= error;
            hasPending = true;
          }
        }
        recomputeHasPending();
        for (const [sourceProjectId, retarget] of retargetedProjects) {
          if (
            !currentState.projects.some((project) => project.id === sourceProjectId) &&
            currentState.projects.some((project) => project.id === retarget.projectId)
          ) {
            retargetedProjects.delete(sourceProjectId);
          }
        }
        reportSync(currentState.projects);
        const projectBoardAssignments = [...pendingBoardAssignments.values()];
        pendingBoardAssignments.clear();
        return {
          conflicts: [...conflicts.values()],
          error:
            firstPushError instanceof Error
              ? firstPushError.message
              : firstPushError
                ? 'A project could not be saved.'
                : null,
          hasPendingChanges: hasPending,
          localDraftStatus,
          projectBoardAssignments,
          shouldRetry,
          snapshot: toSnapshot(currentState, now()),
        };
      });
    },
    unmarkProjectDeleted: (projectId) => {
      const resolvedProjectId = resolveProjectId(projectId);
      if (!serverDeletedProjectIds.has(resolvedProjectId)) {
        deletedProjectIds.delete(resolvedProjectId);
      }
    },
  };
  return service;
};
