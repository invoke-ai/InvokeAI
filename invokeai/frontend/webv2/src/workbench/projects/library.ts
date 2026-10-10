import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { createSingleFlight } from '@platform/state/singleFlight';
import { normalizeServerTimestamp } from '@platform/time/serverTimestamp';
import { DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION, isCanvasSchemaVersionSupported } from '@workbench/canvasSchemaVersion';

import type { ProjectTransferIssues } from './invk/transfer';

import {
  type DeleteProjectBoards,
  deleteProject as apiDeleteProject,
  getProject as apiGetProject,
  getProjectBoardSnapshot,
  listProjects,
  updateProject as apiUpdateProject,
  type ProjectRecordDTO,
  type ProjectSummaryDTO,
} from './api';
import {
  forgetProjectCover,
  getProjectCoverImageName,
  getProjectCoverUrl,
  loadProjectCovers,
  recordProjectCover,
  subscribeProjectCovers,
} from './covers';
import { refreshOpenProjects } from './openProjects';
import { assertProjectFlushed } from './projectFlush';
import { pruneSessionProject } from './session';
import { getOpenProject } from './syncStore';

/** User-scoped library summaries are separate from hydrated tabs; closing a tab must not delete its server record. */

export interface ProjectSummary {
  id: string;
  name: string;
  revision: number;
  minimumCanvasSchemaVersion: number;
  createdAt: string;
  updatedAt: string;
  /** The cover index is independent of document-free listings; absent entries yield no cover. */
  coverUrl?: string;
}

export type ProjectLibraryStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface ProjectLibrarySnapshot {
  status: ProjectLibraryStatus;
  summaries: ProjectSummary[];
  error?: string;
}

const EMPTY_PROJECT_LIBRARY: ProjectLibrarySnapshot = { status: 'idle', summaries: [] };
const store = createExternalStore<ProjectLibrarySnapshot>(EMPTY_PROJECT_LIBRARY);

registerAccountOwnedResource({
  clear: () => {
    store.setSnapshot(EMPTY_PROJECT_LIBRARY);
  },
  name: 'project-library',
});

const withCover = <T extends { id: string }>(summary: T): T & { coverUrl?: string } => {
  const coverImageName = getProjectCoverImageName(summary.id);

  return coverImageName === undefined ? summary : { ...summary, coverUrl: getProjectCoverUrl(coverImageName) };
};

const toSummary = (dto: ProjectSummaryDTO): ProjectSummary =>
  withCover({
    createdAt: normalizeServerTimestamp(dto.created_at),
    id: dto.project_id,
    minimumCanvasSchemaVersion: dto.minimum_canvas_schema_version,
    name: dto.name,
    revision: dto.revision,
    updatedAt: normalizeServerTimestamp(dto.updated_at),
  });

/** Recompute summaries when the independent cover index changes. */
subscribeProjectCovers(() => {
  const { status, summaries } = store.getSnapshot();

  if (status !== 'ready') {
    return;
  }

  const next = summaries.map((summary) => {
    const coverImageName = getProjectCoverImageName(summary.id);
    const coverUrl = coverImageName === undefined ? undefined : getProjectCoverUrl(coverImageName);

    return coverUrl === summary.coverUrl ? summary : { ...summary, coverUrl };
  });

  if (next.some((summary, index) => summary !== summaries[index])) {
    store.patchSnapshot({ summaries: next });
  }
});

/** Most recently edited first — the order Home and the Open dialog present. */
const sortSummaries = (summaries: ProjectSummary[]): ProjectSummary[] =>
  [...summaries].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));

export const useProjectLibrary = (): ProjectLibrarySnapshot => store.useSnapshot();

export const useProjectLibrarySelector = store.useSelector;

export const getProjectLibrary = (): ProjectLibrarySnapshot => store.getSnapshot();

/** Adopt summaries already fetched elsewhere (the workbench boot pass). */
export const seedProjectLibrary = (dtos: ProjectSummaryDTO[], owner: AccountScope): void => {
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  store.setSnapshot({ status: 'ready', summaries: sortSummaries(dtos.map(toSummary)) });
};

const refreshFlight = createSingleFlight<void>();

/** Re-list from the server; concurrent calls share one request. */
export const refreshProjectLibrary = (): Promise<void> =>
  (() => {
    const owner = captureAccountScope();

    return refreshFlight.run(`project-library:${owner.epoch}`, () =>
      // Fetch covers and listings concurrently to avoid a temporary glyph-only grid.
      Promise.all([listProjects(owner.signal), loadProjectCovers()])
        .then(([dtos]) => {
          seedProjectLibrary(dtos, owner);
        })
        .catch((error: unknown) => {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          store.patchSnapshot({
            error: error instanceof Error ? error.message : 'Failed to load projects.',
            status: 'error',
          });
        })
    );
  })();

/** Refresh replaces optimistic local updatedAt with server time. */
export const upsertProjectSummary = (
  entry: { id: string; minimumCanvasSchemaVersion?: number; name: string; revision: number | null },
  owner: AccountScope
): void => {
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  const { summaries } = store.getSnapshot();
  const existing = summaries.find((summary) => summary.id === entry.id);
  const updatedAt = new Date().toISOString();
  const next: ProjectSummary = withCover({
    createdAt: existing?.createdAt ?? updatedAt,
    id: entry.id,
    minimumCanvasSchemaVersion:
      entry.minimumCanvasSchemaVersion ?? existing?.minimumCanvasSchemaVersion ?? DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
    name: entry.name,
    revision: entry.revision ?? existing?.revision ?? 0,
    updatedAt,
  });

  store.patchSnapshot({
    status: 'ready',
    summaries: sortSummaries([...summaries.filter((summary) => summary.id !== entry.id), next]),
  });
};

export const isProjectSummaryCompatible = (summary: ProjectSummary): boolean =>
  isCanvasSchemaVersionSupported(summary.minimumCanvasSchemaVersion);

/** Open projects mutate through their sync engine; closed projects use the HTTP API directly. */

/**
 * Permanently remove a project from the server, its inbox with it. The only deletion path. Its other boards are
 * released to the Library unless asked to go too; media survives either way.
 */
export const deleteLibraryProject = async (
  projectId: string,
  boards: DeleteProjectBoards = 'release'
): Promise<void> => {
  const owner = captureAccountScope();
  const [{ acquireProjectMutationLock }, { createAccountOwnedQueueRunJournal }] = await Promise.all([
    import('./projectLifecycleLocks'),
    import('@workbench/queue-integration/queueRunJournal'),
  ]);
  assertAccountScopeCurrent(owner);
  const openProject = getOpenProject(projectId);
  const mutationLock = await acquireProjectMutationLock(owner.storageSuffix, projectId);
  if (mutationLock.kind === 'contended') {
    throw new Error('Wait for active queue runs to finish before deleting this project.');
  }
  if (mutationLock.kind === 'unavailable') {
    throw new Error('Project deletion is unavailable because cross-tab coordination could not be established.');
  }
  let journal;

  try {
    journal = await createAccountOwnedQueueRunJournal(owner);
    const queueRuns = await journal.listForProject(projectId);
    if (queueRuns.kind === 'unavailable') {
      throw new Error('Project deletion is unavailable because active queue runs could not be verified.');
    }
    if (queueRuns.entries.length > 0) {
      throw new Error('Wait for active queue runs to finish before deleting this project.');
    }

    if (openProject) {
      // Queueing ensures an in-flight save finishes before DELETE is sent.
      await openProject.deleteOnServer(boards);
    } else {
      await apiDeleteProject(projectId, owner.signal, boards);
    }

    assertAccountScopeCurrent(owner);
    await journal.deleteForProject(projectId);
    openProject?.close();
    forgetProjectCover(projectId, owner);
    store.patchSnapshot({ summaries: store.getSnapshot().summaries.filter((summary) => summary.id !== projectId) });

    // Remove deleted projects from durable session state so missing records do not reopen.
    await pruneSessionProject(projectId, owner.signal);
    await refreshOpenProjects();
  } finally {
    journal?.close();
    await mutationLock.release();
  }
};

/**
 * Rename a project. An open one renames through the reducer so its document, its revision chain and
 * its board stay in step; a closed one is a read-modify-write against the server.
 */
export const renameLibraryProject = async (projectId: string, name: string): Promise<void> => {
  const owner = captureAccountScope();
  const openProject = getOpenProject(projectId);

  if (openProject) {
    await openProject.rename(name);
    assertAccountScopeCurrent(owner);
    // The flush has already told the server; this only keeps the grid from waiting for a refetch.
    upsertProjectSummary({ id: projectId, name, revision: null }, owner);

    return;
  }

  const record = await apiGetProject(projectId, owner.signal);

  assertAccountScopeCurrent(owner);
  const updated = await apiUpdateProject(
    projectId,
    {
      data: { ...record.data, name },
      expected_revision: record.revision,
      name,
    },
    owner.signal
  );

  assertAccountScopeCurrent(owner);
  upsertProjectSummary({ id: updated.project_id, name: updated.name, revision: updated.revision }, owner);
};

/** Export/copy requires the paint barrier and an acknowledged flush before reading server bytes. */
export const readAcknowledgedProject = async (projectId: string, owner: AccountScope): Promise<ProjectRecordDTO> => {
  const openProject = getOpenProject(projectId);

  if (openProject) {
    // Written bytes must include canvas pixels that only the live engine holds.
    await openProject.flushPixels();
    assertProjectFlushed(await openProject.flush());
  }

  assertAccountScopeCurrent(owner);

  return apiGetProject(projectId, owner.signal);
};

/** Adopt a project this account just created elsewhere (an import, a duplication). */
export const adoptCreatedProject = (record: ProjectRecordDTO, owner: AccountScope): ProjectSummary => {
  const summary = toSummary(record);

  upsertProjectSummary(
    {
      id: summary.id,
      minimumCanvasSchemaVersion: summary.minimumCanvasSchemaVersion,
      name: summary.name,
      revision: summary.revision,
    },
    owner
  );

  return summary;
};

export interface DuplicatedProject extends ProjectTransferIssues {
  summary: ProjectSummary;
}

/**
 * Duplicate with a fresh identity and whole-board copy. Enumeration failure is fatal; restoration stays lazy and
 * shared.
 */
export const duplicateLibraryProject = async (
  projectId: string,
  options: { onProgress?: (progress: { completed: number; total: number }) => void; owner?: AccountScope } = {}
): Promise<DuplicatedProject> => {
  const owner = options.owner ?? captureAccountScope();
  const record = await readAcknowledgedProject(projectId, owner);

  assertAccountScopeCurrent(owner);

  const snapshot = await getProjectBoardSnapshot(projectId, owner.signal);

  assertAccountScopeCurrent(owner);

  const { duplicateProjectRecord } = await import('./invk/duplicateProject');
  const duplicated = await duplicateProjectRecord(
    { boards: snapshot.boards, owner, record },
    options.onProgress ? { onProgress: options.onProgress } : {}
  );

  assertAccountScopeCurrent(owner);

  if (duplicated.coverImageName) {
    recordProjectCover(duplicated.record.project_id, duplicated.coverImageName, owner);
  }

  return {
    boardIssues: duplicated.boardIssues,
    boardItemIssues: duplicated.boardItemIssues,
    documentReferenceIssues: duplicated.documentReferenceIssues,
    summary: adoptCreatedProject(duplicated.record, owner),
  };
};
