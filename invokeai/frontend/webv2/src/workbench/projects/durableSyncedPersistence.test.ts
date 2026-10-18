import type { AccountScope } from '@platform/state/accountLifecycle';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { createAccountOwnedQueueRunJournal } from '@workbench/queue-integration/queueRunJournal';
import {
  createDraftProject,
  createInitialWorkbenchState,
  withAuthoritativeProjectBoard,
} from '@workbench/workbenchState';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ProjectSummaryDTO, ProjectWithWorkspaceDTO } from './api';
import type { ProjectDraftStore } from './draftStore';
import type { WorkbenchSessionBlob } from './session';

import { ProjectCreateAbsentError } from './api';
import {
  createMemoryProjectDraftStore,
  createUnavailableProjectDraftStore,
  PROJECT_UNLOAD_JOURNAL_MAX_BYTES,
} from './draftStore';
import {
  createDurableSyncedWorkbenchPersistence,
  ProjectDraftWriteRejectedError,
  WorkbenchBackendUnavailableError,
  type DurableProjectPersistenceApi,
} from './durableSyncedPersistence';
import { createDeterministicProjectId } from './ids';
import { serializeProjectDocumentV3, serializeProjectDocumentV3Json } from './projectDocument';
import { splitProjectDocument } from './projectHalves';
import { deserializeProjectRecord } from './projectHydration';
import { acquireProjectMutationLock } from './projectLifecycleLocks';
import { getProjectSyncSnapshot, registerOpenProject, unregisterOpenProject } from './syncStore';

const listQueueRunProjectIds = vi.hoisted(() =>
  vi.fn<() => Promise<{ kind: 'available'; projectIds: string[] } | { kind: 'unavailable' }>>(() =>
    Promise.resolve({ kind: 'available', projectIds: [] })
  )
);

vi.mock('./projectLifecycleLocks', () => ({
  acquireProjectMutationLock: vi.fn(() =>
    Promise.resolve({ kind: 'acquired' as const, release: () => Promise.resolve() })
  ),
}));
vi.mock('@workbench/queue-integration/queueRunJournal', () => ({
  createAccountOwnedQueueRunJournal: vi.fn(() =>
    Promise.resolve({
      close: vi.fn(),
      listProjectIds: listQueueRunProjectIds,
      listForProject: vi.fn(() => Promise.resolve({ entries: [], kind: 'available', removedCorrupt: 0 })),
    })
  ),
}));

const now = '2026-09-03T12:00:00.000Z';

const stateWith = (projects: Project[]): WorkbenchState => ({
  ...createInitialWorkbenchState(),
  activeProjectId: projects[0]?.id ?? '',
  projects,
});

/** A project as a build before the split saved it: one whole document. */
const toRecord = (project: Project, revision = 1): ProjectWithWorkspaceDTO => ({
  board_id: `board-${project.id}`,
  created_at: now,
  data: serializeProjectDocumentV3(project),
  minimum_canvas_schema_version: 3,
  minimum_document_schema_version: 3,
  name: project.name,
  project_id: project.id,
  revision,
  updated_at: now,
  workspace: null,
  workspace_updated_at: null,
});

/** A project as this build saves it: content, and the caller's workspace beside it. */
const toSplitRecord = (project: Project, revision = 1, workspaceRevision = 1): ProjectWithWorkspaceDTO => {
  const halves = splitProjectDocument(serializeProjectDocumentV3(project));
  return {
    ...toRecord(project, revision),
    data: halves.content as unknown as Record<string, unknown>,
    minimum_document_schema_version: 4,
    workspace: {
      data: halves.workspace as unknown as Record<string, unknown>,
      revision: workspaceRevision,
      updated_at: now,
    },
    workspace_updated_at: now,
  };
};

/** What the server answers a create of `project` with: its content, and no workspace yet. */
const toSummary = (record: ProjectWithWorkspaceDTO): ProjectSummaryDTO => ({
  board_id: record.board_id,
  created_at: record.created_at,
  minimum_canvas_schema_version: record.minimum_canvas_schema_version,
  minimum_document_schema_version: record.minimum_document_schema_version,
  name: record.name,
  project_id: record.project_id,
  revision: record.revision,
  updated_at: record.updated_at,
  workspace_updated_at: record.workspace_updated_at,
});

const createApi = () => {
  const records = new Map<string, ProjectWithWorkspaceDTO>();
  const api: DurableProjectPersistenceApi & { records: Map<string, ProjectWithWorkspaceDTO> } = {
    records,
    createProject: vi.fn((request) => {
      const id = request.project_id!;
      // The caller's first workspace is stored with the content, and the answer carries it.
      const workspace = request.workspace
        ? { data: structuredClone(request.workspace), revision: 1, updated_at: now }
        : null;
      const record: ProjectWithWorkspaceDTO = {
        board_id: `board-${id}`,
        created_at: now,
        data: structuredClone(request.data),
        minimum_canvas_schema_version: request.minimum_canvas_schema_version ?? 3,
        minimum_document_schema_version: request.minimum_document_schema_version ?? 3,
        name: request.name,
        project_id: id,
        revision: 1,
        updated_at: now,
        workspace,
        workspace_updated_at: workspace ? now : null,
      };
      records.set(id, record);
      return Promise.resolve(structuredClone(record));
    }),
    deleteProject: vi.fn((projectId) => {
      records.delete(projectId);
      return Promise.resolve();
    }),
    deleteSession: vi.fn(() => Promise.resolve()),
    getProject: vi.fn((projectId) => {
      const record = records.get(projectId);
      if (!record) {
        return Promise.reject(Object.assign(new Error('not found'), { status: 404 }));
      }
      return Promise.resolve(structuredClone(record));
    }),
    loadSession: vi.fn(() => Promise.resolve(null)),
    listProjects: vi.fn(() => Promise.resolve([...records.values()].map(toSummary))),
    saveProjectWorkspace: vi.fn((projectId, request) => {
      const current = records.get(projectId);
      if (!current) {
        return Promise.reject(Object.assign(new Error('not found'), { status: 404 }));
      }
      if ((current.workspace?.revision ?? null) !== request.expected_revision) {
        return Promise.reject(Object.assign(new Error('conflict'), { status: 409 }));
      }
      const workspace = {
        data: structuredClone(request.data),
        revision: (current.workspace?.revision ?? 0) + 1,
        updated_at: now,
      };
      records.set(projectId, { ...current, workspace, workspace_updated_at: now });
      return Promise.resolve(structuredClone(workspace));
    }),
    saveSession: vi.fn(() => Promise.resolve()),
    updateProject: vi.fn((projectId, request) => {
      const current = records.get(projectId);
      if (!current) {
        return Promise.reject(Object.assign(new Error('not found'), { status: 404 }));
      }
      if (current.revision !== request.expected_revision) {
        return Promise.reject(Object.assign(new Error('conflict'), { status: 409 }));
      }
      // A workspace sent with the content is stored in the same step, or refuses the content with it.
      if (request.workspace && (current.workspace?.revision ?? null) !== request.workspace.expected_revision) {
        return Promise.reject(Object.assign(new Error('conflict'), { status: 409 }));
      }
      const workspace = request.workspace
        ? {
            data: structuredClone(request.workspace.data),
            revision: (current.workspace?.revision ?? 0) + 1,
            updated_at: now,
          }
        : null;
      const record: ProjectWithWorkspaceDTO = {
        ...current,
        data: structuredClone(request.data),
        minimum_canvas_schema_version: Math.max(
          current.minimum_canvas_schema_version,
          request.minimum_canvas_schema_version ?? current.minimum_canvas_schema_version
        ),
        minimum_document_schema_version: Math.max(
          current.minimum_document_schema_version,
          request.minimum_document_schema_version ?? current.minimum_document_schema_version
        ),
        name: request.name,
        revision: current.revision + 1,
        updated_at: now,
        ...(workspace ? { workspace, workspace_updated_at: now } : {}),
      };
      records.set(projectId, record);
      // The answer names only the workspace this write stored.
      return Promise.resolve({ ...structuredClone(record), workspace: structuredClone(workspace) });
    }),
  };
  return api;
};

const canvasSchemaRefusal = () =>
  new ApiError(
    JSON.stringify({
      detail: { code: 'canvas_schema_unsupported', max_canvas_schema_version: 3, minimum_canvas_schema_version: 4 },
    }),
    412
  );

const createService = (
  owner: AccountScope,
  api: DurableProjectPersistenceApi,
  draftStore: ProjectDraftStore = createMemoryProjectDraftStore()
) =>
  createDurableSyncedWorkbenchPersistence(owner, {
    api,
    deleteDatabase: () => Promise.resolve({ kind: 'deleted' }),
    draftStore: Promise.resolve(draftStore),
    editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
    now: () => now,
    writerToken: 'writer-1',
  });

beforeEach(() => {
  accountLifecycle.activate('durable-sync-test');
  vi.mocked(acquireProjectMutationLock).mockReset();
  vi.mocked(acquireProjectMutationLock).mockResolvedValue({
    kind: 'acquired',
    release: () => Promise.resolve(),
  });
  vi.mocked(createAccountOwnedQueueRunJournal).mockReset();
  vi.mocked(createAccountOwnedQueueRunJournal).mockResolvedValue({
    availability: 'available',
    acknowledgeReceipt: vi.fn(),
    listPendingReceipts: vi.fn(),
    close: vi.fn(),
    deleteForProject: vi.fn(),
    listAll: vi.fn(),
    listProjectIds: listQueueRunProjectIds,
    listForProject: vi.fn(() => Promise.resolve({ entries: [], kind: 'available' as const, removedCorrupt: 0 })),
    record: vi.fn(),
    settle: vi.fn(),
  });
  listQueueRunProjectIds.mockReset();
  listQueueRunProjectIds.mockResolvedValue({ kind: 'available', projectIds: [] });
});

describe('durable project persistence', () => {
  it('awaits one acknowledged creation for concurrent uploads without saving later edits', async () => {
    const api = createApi();
    const project = createDraftProject([]);
    const service = createService(captureAccountScope(), api);
    let acknowledge!: (record: ProjectWithWorkspaceDTO) => void;
    const created = new Promise<ProjectWithWorkspaceDTO>((resolve) => {
      acknowledge = resolve;
    });
    vi.mocked(api.createProject).mockReturnValueOnce(created);
    const first = service.ensureProjectOnServer(project);
    const second = service.ensureProjectOnServer(project);
    const ready = vi.fn();
    void first.then(ready);
    await vi.waitFor(() => expect(api.createProject).toHaveBeenCalledOnce());
    expect(ready).not.toHaveBeenCalled();

    api.records.set(project.id, toSplitRecord(project));
    acknowledge(toSplitRecord(project));
    await Promise.all([first, second]);
    await service.ensureProjectOnServer({ ...project, name: 'Unsaved later edit' });
    expect(api.createProject).toHaveBeenCalledOnce();
    expect(api.updateProject).not.toHaveBeenCalled();
    expect(ready).toHaveBeenCalledOnce();
  });

  it('answers an acknowledged project without waiting behind queued saves of other projects', async () => {
    const api = createApi();
    const project = createDraftProject([]);
    const other = createDraftProject([]);
    const service = createService(captureAccountScope(), api);
    await service.ensureProjectOnServer(project);
    vi.mocked(api.createProject).mockReturnValueOnce(new Promise<never>(() => {}));
    void service.ensureProjectOnServer(other).catch(() => undefined);
    await vi.waitFor(() => expect(api.createProject).toHaveBeenCalledTimes(2));

    await expect(service.ensureProjectOnServer(project)).resolves.toBeUndefined();
  });

  it('fences an upload when its project is deleted while creation is pending', async () => {
    const api = createApi();
    const project = createDraftProject([]);
    const service = createService(captureAccountScope(), api);
    let acknowledge!: (record: ProjectWithWorkspaceDTO) => void;
    vi.mocked(api.createProject).mockReturnValueOnce(
      new Promise((resolve) => {
        acknowledge = resolve;
      })
    );
    const pending = service.ensureProjectOnServer(project);
    const rejected = expect(pending).rejects.toMatchObject({ reason: 'superseded' });
    await vi.waitFor(() => expect(api.createProject).toHaveBeenCalledOnce());
    service.markProjectDeleted(project.id);
    acknowledge(toSplitRecord(project));
    await rejected;
  });

  it('shares one failed creation across waiting and following uploads, then retries after a pause', async () => {
    const api = createApi();
    const project = createDraftProject([]);
    const service = createService(captureAccountScope(), api);
    const clock = vi.spyOn(Date, 'now').mockReturnValue(1_000);
    let fail!: (error: Error) => void;
    vi.mocked(api.createProject).mockReturnValueOnce(
      new Promise((_resolve, reject) => {
        fail = reject;
      })
    );
    try {
      const waiting = [service.ensureProjectOnServer(project), service.ensureProjectOnServer(project)];
      await vi.waitFor(() => expect(api.createProject).toHaveBeenCalledOnce());
      fail(new Error('offline'));
      for (const upload of waiting) {
        await expect(upload).rejects.toMatchObject({ reason: 'unsynced' });
      }
      await expect(service.ensureProjectOnServer(project)).rejects.toMatchObject({ reason: 'unsynced' });
      expect(api.createProject).toHaveBeenCalledOnce();

      clock.mockReturnValue(10_000);
      await service.ensureProjectOnServer(project);
      expect(api.createProject).toHaveBeenCalledTimes(2);
    } finally {
      clock.mockRestore();
    }
  });

  it('rejects deleted projects and expired accounts even after their creation was acknowledged', async () => {
    const api = createApi();
    const project = createDraftProject([]);
    const service = createService(captureAccountScope(), api);
    await service.ensureProjectOnServer(project);
    service.markProjectDeleted(project.id);
    await expect(service.ensureProjectOnServer(project)).rejects.toMatchObject({ reason: 'superseded' });
    service.unmarkProjectDeleted(project.id);
    accountLifecycle.activate('another-account');
    await expect(service.ensureProjectOnServer(project)).rejects.toMatchObject({ name: 'AccountScopeExpiredError' });
    expect(api.createProject).toHaveBeenCalledOnce();
    expect(api.updateProject).not.toHaveBeenCalled();
  });

  it('activates a requested new project and restores it after saving and reloading', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const previous = createDraftProject([]);
    api.records.set(previous.id, toRecord(previous));
    let session: WorkbenchSessionBlob = {
      account: createInitialWorkbenchState().account,
      activeProjectId: previous.id,
      openProjectIds: [previous.id],
    };
    vi.mocked(api.loadSession).mockImplementation(() => Promise.resolve(session));
    vi.mocked(api.saveSession).mockImplementation((state) => {
      session = {
        account: state.account,
        activeProjectId: state.activeProjectId,
        openProjectIds: state.projects.map((project) => project.id),
      };
      return Promise.resolve();
    });
    const service = createService(owner, api);
    const loaded = await service.loadWorkbench({ createNew: true });
    const created = loaded.state.projects.find((project) => project.id !== previous.id)!;
    expect(loaded.state.activeProjectId).toBe(created.id);
    await service.saveWorkbench(loaded.state);
    const reloaded = await createService(owner, api).loadWorkbench();
    expect(reloaded.state.activeProjectId).toBe(created.id);
  });

  it('refuses to retarget a project while it has active queue runs', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    project.queue.items = [{ status: 'running' } as Project['queue']['items'][number]];

    await expect(createService(owner, api).resolveConflictSaveAsNew(project)).rejects.toThrow(
      'Wait for active queue runs'
    );
    expect(api.createProject).not.toHaveBeenCalled();
  });

  it('refuses conflict resolution while another tab owns a project run', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    vi.mocked(acquireProjectMutationLock).mockResolvedValueOnce({ kind: 'contended' });

    await expect(createService(owner, api).resolveConflictSaveAsNew(project)).rejects.toThrow('active queue runs');

    expect(api.createProject).not.toHaveBeenCalled();
  });

  it('refuses conflict resolution when a dormant queue journal row exists', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    const release = vi.fn().mockResolvedValue(undefined);
    vi.mocked(acquireProjectMutationLock).mockResolvedValueOnce({ kind: 'acquired', release });
    vi.mocked(createAccountOwnedQueueRunJournal).mockResolvedValueOnce({
      availability: 'available',
      acknowledgeReceipt: vi.fn(),
      listPendingReceipts: vi.fn(),
      close: vi.fn(),
      deleteForProject: vi.fn(),
      listAll: vi.fn(),
      listProjectIds: vi.fn().mockResolvedValue({ kind: 'available', projectIds: [project.id] }),
      listForProject: vi.fn().mockResolvedValue({
        entries: [{ item: {}, projectId: project.id, queueItemId: 'run-1', submissionOrder: 1 }],
        kind: 'available',
        removedCorrupt: 0,
      }),
      record: vi.fn(),
      settle: vi.fn(),
    });

    await expect(createService(owner, api).resolveConflictSaveAsNew(project)).rejects.toThrow('active queue runs');

    expect(api.createProject).not.toHaveBeenCalled();
    expect(release).toHaveBeenCalledOnce();
  });

  it('deletes the legacy mirror keys before loading', async () => {
    const owner = captureAccountScope();
    const clearLegacyStorage = vi.fn();
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api: createApi(),
      clearLegacyStorage,
      draftStore: Promise.resolve(createMemoryProjectDraftStore()),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      writerToken: 'writer-1',
    });

    await service.loadWorkbench();

    expect(clearLegacyStorage).toHaveBeenCalledOnce();
  });

  it('does not fabricate a default session when the backend cannot be listed', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    vi.mocked(api.listProjects).mockRejectedValueOnce(new Error('offline'));

    await expect(createService(owner, api).loadWorkbench()).rejects.toBeInstanceOf(WorkbenchBackendUnavailableError);
    expect(api.createProject).not.toHaveBeenCalled();
  });

  it('hydrates large project documents sequentially', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const projects = Array.from({ length: 6 }, () => createDraftProject([]));
    for (const project of projects) {
      api.records.set(project.id, toRecord(project));
    }
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: projects[0]!.id,
      openProjectIds: projects.map((project) => project.id),
    });
    let active = 0;
    let maximum = 0;
    vi.mocked(api.getProject).mockImplementation(async (projectId, signal) => {
      active += 1;
      maximum = Math.max(maximum, active);
      await Promise.resolve();
      await Promise.resolve();
      void signal;
      const record = api.records.get(projectId);
      if (!record) {
        throw Object.assign(new Error('not found'), { status: 404 });
      }
      const result = structuredClone(record);
      active -= 1;
      return result;
    });

    await createService(owner, api).loadWorkbench();

    expect(maximum).toBe(1);
  });

  it('preserves the legacy session contract that omitted open project ids', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const list = vi.spyOn(draftStore, 'list');
    const listForProject = vi.spyOn(draftStore, 'listForProject');
    const projects = Array.from({ length: 3 }, () => createDraftProject([]));
    for (const project of projects) {
      api.records.set(project.id, toRecord(project));
    }
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: projects[1]!.id,
    });

    const loaded = await createService(owner, api, draftStore).loadWorkbench();

    expect(loaded.state.projects.map((project) => project.id)).toEqual(projects.map((project) => project.id));
    expect(loaded.state.activeProjectId).toBe(projects[1]!.id);
    expect(list).toHaveBeenCalledOnce();
    expect(listForProject).not.toHaveBeenCalled();
  });

  it('restores a server project with durable queue work after another tab empties the session', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: '',
      openProjectIds: [],
    });
    listQueueRunProjectIds.mockResolvedValue({ kind: 'available', projectIds: [project.id] });

    const loaded = await createService(owner, api).loadWorkbench();

    expect(loaded.state.projects.map((candidate) => candidate.id)).toEqual([project.id]);
    expect(loaded.state.activeProjectId).toBe(project.id);
  });

  it('bounds journal-driven auto-open deterministically and reports retained overflow', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const projects = ['queue-c', 'queue-a', 'queue-b'].map((id) => ({ ...createDraftProject([]), id }));
    for (const project of projects) {
      api.records.set(project.id, toRecord(project));
    }
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: '',
      openProjectIds: [],
    });
    listQueueRunProjectIds.mockResolvedValue({
      kind: 'available',
      projectIds: projects.map((project) => project.id),
    });
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      autoOpenQueueRunProjectLimit: 1,
      draftStore: Promise.resolve(createMemoryProjectDraftStore()),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      writerToken: 'writer-1',
    });

    const loaded = await service.loadWorkbench();

    expect(loaded.state.projects.map((project) => project.id)).toEqual(['queue-a']);
    expect(loaded.queueRecovery).toEqual({
      projects: [
        { projectId: 'queue-b', reason: 'open-limit' },
        { projectId: 'queue-c', reason: 'open-limit' },
      ],
      status: 'available',
    });
  });

  it('restores a draft-backed journal project after its server record was deleted', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3(project)),
      documentSchemaVersion: 3,
      editorSessionId: 'older-editor',
      generation: 1,
      projectId: project.id,
      updatedAt: 1,
      writerToken: 'older-writer',
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: '',
      openProjectIds: [],
    });
    listQueueRunProjectIds.mockResolvedValue({ kind: 'available', projectIds: [project.id] });

    const loaded = await createService(owner, api, draftStore).loadWorkbench();

    expect(loaded.state.projects.map((candidate) => candidate.id)).toEqual([project.id]);
    expect(loaded.conflicts).toEqual([expect.objectContaining({ kind: 'deleted', projectId: project.id })]);
    expect(loaded.queueRecovery.projects).toEqual([]);
  });

  it('migrates an older draft on recovery, turning its library binding into an unknown-revision source', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    const { workflows: _workflows, ...document } = serializeProjectDocumentV3(project);
    const graph = project.workflows.entries[0]!.document;
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify({
        ...document,
        documentSchemaVersion: 2,
        projectGraph: { ...graph, libraryWorkflowId: 'lib-1' },
      }),
      documentSchemaVersion: 2,
      editorSessionId: 'older-editor',
      generation: 1,
      projectId: project.id,
      updatedAt: 1,
      writerToken: 'older-writer',
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: '',
      openProjectIds: [],
    });
    listQueueRunProjectIds.mockResolvedValue({ kind: 'available', projectIds: [project.id] });

    const loaded = await createService(owner, api, draftStore).loadWorkbench();
    const restored = loaded.state.projects.find((candidate) => candidate.id === project.id);

    expect(restored?.workflows.entries.map((entry) => entry.document.id)).toEqual([graph.id]);
    expect(restored?.workflows.entries[0]?.source).toEqual({ libraryWorkflowId: 'lib-1', revision: null });
    expect(restored).not.toHaveProperty('projectGraph');
  });

  it('reports a journal project that has neither a server record nor a recoverable draft', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: '',
      openProjectIds: [],
    });
    listQueueRunProjectIds.mockResolvedValue({ kind: 'available', projectIds: ['orphan-project'] });

    const loaded = await createService(owner, api).loadWorkbench();

    expect(loaded.queueRecovery).toEqual({
      projects: [{ projectId: 'orphan-project', reason: 'project-unavailable' }],
      status: 'available',
    });
    expect(loaded.state.projects).toHaveLength(1);
    expect(loaded.state.projects[0]!.id).not.toBe('orphan-project');
  });

  it('loads backend projects when queue recovery storage is unavailable', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    listQueueRunProjectIds.mockResolvedValue({ kind: 'unavailable' });

    const loaded = await createService(owner, api).loadWorkbench();

    expect(loaded.state.projects.map((candidate) => candidate.id)).toEqual([project.id]);
    expect(loaded.queueRecovery).toEqual({ projects: [], status: 'unavailable' });
  });

  it('bounds account-wide draft auto-open while keeping overflow drafts recoverable', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const anchor = createDraftProject([]);
    const drafts = Array.from({ length: 3 }, (_, index) => ({
      ...createDraftProject([anchor]),
      name: `Recovered ${index}`,
    }));
    api.records.set(anchor.id, toRecord(anchor));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: anchor.id,
      openProjectIds: [anchor.id],
    });
    for (const [index, project] of drafts.entries()) {
      await draftStore.stage({
        baseRevision: null,
        documentJson: JSON.stringify(serializeProjectDocumentV3(project)),
        documentSchemaVersion: 3,
        editorSessionId: `old-${index}`,
        generation: 1,
        projectId: project.id,
        updatedAt: index + 1,
        writerToken: `old-writer-${index}`,
      });
    }
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      autoOpenDraftByteLimit: Number.MAX_SAFE_INTEGER,
      autoOpenDraftLimit: 1,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      writerToken: 'writer-1',
    });

    const loaded = await service.loadWorkbench();

    expect(loaded.state.projects.map((project) => project.name)).toEqual([anchor.name, 'Recovered 2']);
    expect(
      getProjectSyncSnapshot()
        .recoverableDrafts.map((draft) => draft.projectId)
        .sort()
    ).toEqual(
      drafts
        .slice(0, 2)
        .map((draft) => draft.id)
        .sort()
    );
  });

  it('charges the largest eligible lineage to the draft auto-open byte budget', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const anchor = createDraftProject([]);
    const recovered = createDraftProject([anchor]);
    api.records.set(anchor.id, toRecord(anchor));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: anchor.id,
      openProjectIds: [anchor.id],
    });
    const large = {
      ...recovered,
      promptHistory: [{ negativePrompt: null, positivePrompt: 'x'.repeat(4_096) }],
    };
    for (const [editorSessionId, project, updatedAt] of [
      ['large-editor', large, 1],
      ['newer-small-editor', recovered, 2],
    ] as const) {
      await draftStore.stage({
        baseRevision: null,
        documentJson: JSON.stringify(serializeProjectDocumentV3(project)),
        documentSchemaVersion: 3,
        editorSessionId,
        generation: 1,
        projectId: recovered.id,
        updatedAt,
        writerToken: `${editorSessionId}-writer`,
      });
    }
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      autoOpenDraftByteLimit: 2_048,
      autoOpenDraftLimit: 8,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      writerToken: 'writer-1',
    });

    const loaded = await service.loadWorkbench();

    expect(loaded.state.projects.map((project) => project.id)).toEqual([anchor.id]);
    expect(getProjectSyncSnapshot().recoverableDrafts).toHaveLength(2);
  });

  it('clears browser drafts even when backend cleanup is unavailable', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    await draftStore.stage({
      baseRevision: null,
      documentJson: JSON.stringify(serializeProjectDocumentV3(project)),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'writer-1',
    });
    vi.mocked(api.listProjects).mockRejectedValueOnce(new Error('offline'));

    await expect(createService(owner, api, draftStore).clearWorkbench()).rejects.toThrow(
      'Workbench data could not be fully cleared.'
    );

    await expect(draftStore.get(project.id, 'editor-1')).resolves.toEqual({ kind: 'unavailable' });
  });

  it('attempts every bounded backend cleanup and reports partial deletion', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const projects = Array.from({ length: 5 }, () => createDraftProject([]));
    for (const project of projects) {
      api.records.set(project.id, toRecord(project));
    }
    vi.mocked(api.deleteProject).mockImplementation((projectId) => {
      if (projectId === projects[1]!.id) {
        return Promise.reject(new Error('delete failed'));
      }
      api.records.delete(projectId);
      return Promise.resolve();
    });

    const service = createService(owner, api);
    await expect(service.clearWorkbench()).rejects.toThrow('Workbench data could not be fully cleared.');

    expect(api.deleteProject).toHaveBeenCalledTimes(projects.length);
    expect(api.deleteSession).toHaveBeenCalledOnce();
    expect(api.records.has(projects[1]!.id)).toBe(true);
    await expect(service.saveWorkbench(stateWith(projects))).rejects.toThrow('must be reloaded');
    expect(api.createProject).not.toHaveBeenCalled();
  });

  it('fences autosaves and session writes as soon as clearing starts', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    let releaseList!: () => void;
    const listReleased = new Promise<void>((resolve) => {
      releaseList = resolve;
    });
    vi.mocked(api.listProjects).mockImplementationOnce(async () => {
      await listReleased;
      return [toSummary(toRecord(project))];
    });
    const service = createService(owner, api);

    const clearing = service.clearWorkbench();

    await expect(service.saveWorkbench(stateWith([project]))).rejects.toThrow('must be reloaded');
    await expect(service.persistEmptySession(stateWith([project]))).rejects.toThrow('must be reloaded');
    releaseList();
    await clearing;
    expect(api.createProject).not.toHaveBeenCalled();
    expect(api.updateProject).not.toHaveBeenCalled();
  });

  it('waits for blocked browser-data deletion and rejects unavailable deletion', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    let finishDeletion!: (result: { kind: 'deleted' }) => void;
    const completion = new Promise<{ kind: 'deleted' }>((resolve) => {
      finishDeletion = resolve;
    });
    const blocked = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      deleteDatabase: vi.fn(() => Promise.resolve({ completion, kind: 'blocked' as const })),
      draftStore: Promise.resolve(createMemoryProjectDraftStore()),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      writerToken: 'writer-1',
    });
    let didClear = false;
    const clearing = blocked.clearWorkbench().then(() => {
      didClear = true;
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(didClear).toBe(false);
    finishDeletion({ kind: 'deleted' });
    await clearing;

    const unavailable = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      deleteDatabase: vi.fn(() => Promise.resolve({ kind: 'unavailable' as const })),
      draftStore: Promise.resolve(createMemoryProjectDraftStore()),
      editorSession: Promise.resolve({ id: 'editor-2', release: () => Promise.resolve() }),
      writerToken: 'writer-2',
    });
    await expect(unavailable.clearWorkbench()).rejects.toThrow('Workbench data could not be fully cleared');
  });

  it('fails a blocked browser-data deletion after its deadline', async () => {
    const owner = captureAccountScope();
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api: createApi(),
      databaseDeleteTimeoutMs: 1,
      deleteDatabase: vi.fn(() =>
        Promise.resolve({
          completion: new Promise<never>(() => {
            // Intentionally unresolved to exercise the deletion deadline.
          }),
          kind: 'blocked' as const,
        })
      ),
      draftStore: Promise.resolve(createMemoryProjectDraftStore()),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      writerToken: 'writer-1',
    });

    await expect(service.clearWorkbench()).rejects.toThrow('Workbench data could not be fully cleared');
  });

  it('keeps persistence resources alive across a StrictMode-style synthetic remount', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const close = vi.spyOn(draftStore, 'close');
    const releaseSession = vi.fn(() => Promise.resolve());
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: releaseSession }),
      writerToken: 'writer-1',
    });
    await service.loadWorkbench();

    const releaseFirstMount = service.retain();
    releaseFirstMount();
    const releaseRemount = service.retain();
    await Promise.resolve();

    expect(close).not.toHaveBeenCalled();
    expect(releaseSession).not.toHaveBeenCalled();

    releaseRemount();
    await Promise.resolve();
    await Promise.resolve();

    expect(close).toHaveBeenCalledOnce();
    expect(releaseSession).toHaveBeenCalledOnce();
  });

  it('resolves the last release only once the editor session lock is given back', async () => {
    const owner = captureAccountScope();
    let releaseLock!: () => void;
    const lockReleased = new Promise<void>((resolve) => {
      releaseLock = resolve;
    });
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api: createApi(),
      draftStore: Promise.resolve(createMemoryProjectDraftStore()),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => lockReleased }),
      writerToken: 'writer-1',
    });
    await service.loadWorkbench();
    let isReleased = false;

    void service
      .retain()()
      .then(() => {
        isReleased = true;
      });
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
    expect(isReleased).toBe(false);

    releaseLock();
    await vi.waitFor(() => expect(isReleased).toBe(true));
  });

  it('keeps a closed session closed until the editor reopens it', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const service = createService(owner, api);
    const project = createDraftProject([]);
    await service.loadWorkbench();

    const reopened = vi.fn();
    service.subscribeSessionReopened(reopened);
    const closing = service.persistEmptySession(stateWith([project]));
    expect(service.hasClosedSession()).toBe(true);
    await closing;
    expect(service.hasClosedSession()).toBe(true);
    expect(reopened).not.toHaveBeenCalled();

    vi.mocked(api.saveSession).mockClear();
    await service.reopenSession(stateWith([project]));

    expect(service.hasClosedSession()).toBe(false);
    expect(reopened).toHaveBeenCalledOnce();
    expect(
      vi
        .mocked(api.saveSession)
        .mock.calls.at(-1)?.[0]
        .projects.map(({ id }) => id)
    ).toEqual([project.id]);
  });

  it('gives up project locks held for an unfinished resolution when a newer editor takes over', async () => {
    const owner = captureAccountScope();
    const release = vi.fn(() => Promise.resolve());
    vi.mocked(acquireProjectMutationLock).mockResolvedValue({ kind: 'acquired', release });
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3(project)),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api, draftStore);
    await service.loadWorkbench();
    await service.resolveConflictDiscard(project.id);
    expect(release).not.toHaveBeenCalled();

    service.releaseMutationLocks();
    expect(release).toHaveBeenCalledOnce();

    service.acknowledgeProjectResolution(project.id);
    expect(release).toHaveBeenCalledOnce();
  });

  it('stages the exact V2 document before issuing a create', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const stage = vi.spyOn(draftStore, 'stage');
    const project = createDraftProject([]);
    const service = createService(owner, api, draftStore);

    await service.saveWorkbench(stateWith([project]));

    expect(stage).toHaveBeenCalledBefore(vi.mocked(api.createProject));
    expect(stage).toHaveBeenCalledWith(
      expect.objectContaining({
        baseRevision: null,
        documentJson: JSON.stringify(serializeProjectDocumentV3(project)),
        documentSchemaVersion: 3,
        projectId: project.id,
      })
    );
  });

  it('continues the authoritative save when browser draft storage is unavailable', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    const service = createService(owner, api, createUnavailableProjectDraftStore());

    const result = await service.saveWorkbench(stateWith([project]));

    expect(api.createProject).toHaveBeenCalledOnce();
    expect(result.localDraftStatus).toBe('unavailable');
    expect(result.hasPendingChanges).toBe(false);
  });

  it('clears the local recovery warning after a later draft is durably acknowledged', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backingStore = createMemoryProjectDraftStore();
    const stage = vi
      .fn(backingStore.stage)
      .mockResolvedValueOnce({ kind: 'quota' as const })
      .mockImplementation(backingStore.stage);
    const draftStore: ProjectDraftStore = { ...backingStore, stage };
    const project = createDraftProject([]);
    const service = createService(owner, api, draftStore);

    const first = await service.saveWorkbench(stateWith([project]));
    const second = await service.saveWorkbench(stateWith([{ ...project, name: 'durably backed up' }]));

    expect(first.localDraftStatus).toBe('unavailable');
    expect(second.localDraftStatus).toBe('ok');
  });

  it('removes an older durable draft after a newer quota-degraded save is acknowledged', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backingStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const stage = vi
      .fn(backingStore.stage)
      .mockImplementationOnce(backingStore.stage)
      .mockResolvedValueOnce({ kind: 'quota' });
    const settleAcknowledgement = vi
      .fn(backingStore.settleAcknowledgement)
      .mockResolvedValueOnce({ kind: 'unavailable' })
      .mockImplementation(backingStore.settleAcknowledgement);
    const draftStore: ProjectDraftStore = { ...backingStore, settleAcknowledgement, stage };
    const service = createService(owner, api, draftStore);
    const loaded = await service.loadWorkbench();

    await service.saveWorkbench(stateWith([{ ...loaded.state.projects[0]!, name: 'first edit' }]));
    await service.saveWorkbench(stateWith([{ ...loaded.state.projects[0]!, name: 'second edit' }]));

    await expect(backingStore.get(project.id, 'editor-1')).resolves.toMatchObject({ kind: 'empty' });
    const reloaded = await createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(backingStore),
      editorSession: Promise.resolve({ id: 'editor-2', release: () => Promise.resolve() }),
      now: () => now,
      writerToken: 'writer-2',
    }).loadWorkbench();
    expect(reloaded.conflicts).toEqual([]);
    expect(reloaded.state.projects).toMatchObject([{ name: 'second edit' }]);
  });

  it.each(['fenced', 'generation-conflict', 'stale'] as const)(
    'does not write to the server when draft staging reports %s',
    async (kind) => {
      const owner = captureAccountScope();
      const api = createApi();
      const backingStore = createMemoryProjectDraftStore();
      const draftStore: ProjectDraftStore = {
        ...backingStore,
        stage: vi.fn(() => Promise.resolve({ kind })),
      };

      const result = await createService(owner, api, draftStore).saveWorkbench(stateWith([createDraftProject([])]));

      expect(result.error).toContain(`rejected (${kind})`);
      expect(result.hasPendingChanges).toBe(true);
      expect(api.createProject).not.toHaveBeenCalled();
      expect(api.updateProject).not.toHaveBeenCalled();
    }
  );

  it('isolates corrupt browser storage and continues the authoritative save', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backingStore = createMemoryProjectDraftStore();
    const stage = vi.fn(backingStore.stage).mockResolvedValueOnce({ kind: 'corrupt' });
    const draftStore: ProjectDraftStore = { ...backingStore, stage };

    const result = await createService(owner, api, draftStore).saveWorkbench(stateWith([createDraftProject([])]));

    expect(stage).toHaveBeenCalledTimes(2);
    expect(stage.mock.calls[1]![0].editorSessionId).not.toBe('editor-1');
    expect(api.createProject).toHaveBeenCalledOnce();
    expect(result.localDraftStatus).toBe('unavailable');
  });

  it('refuses to create a conflict copy without durable idempotency state', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backingStore = createMemoryProjectDraftStore();
    const quotaStore: ProjectDraftStore = {
      ...backingStore,
      reserveCopyIdentity: vi.fn(() => Promise.resolve({ kind: 'quota' as const })),
      stage: vi.fn(() => Promise.resolve({ kind: 'quota' as const })),
    };
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const saveDraftAsNew = vi.fn((input) =>
      Promise.resolve({
        ...toRecord({ ...project, id: input.copyProjectId, name: input.name }),
        data: structuredClone(input.document),
        minimum_canvas_schema_version: input.minimumCanvasSchemaVersion,
      })
    );
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(quotaStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      saveDraftAsNew,
      writerToken: 'writer-1',
    });
    const loaded = await service.loadWorkbench();
    api.records.set(project.id, toRecord({ ...project, name: 'remote edit' }, 2));
    const localProject = { ...loaded.state.projects[0]!, name: 'local edit' };
    await service.saveWorkbench(stateWith([localProject]));

    await expect(service.resolveConflictSaveAsNew(localProject)).rejects.toThrow('up-to-date local recovery draft');
    expect(saveDraftAsNew).not.toHaveBeenCalled();
  });

  it('refuses to copy a volatile edit newer than its durable conflict draft', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backingStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project, 2));
    await backingStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'writer-1',
    });
    await backingStore.settleConflict(project.id, 'editor-1', 'writer-1', 1, {
      kind: 'revision',
      serverRevision: 2,
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    let rejectStages = false;
    const draftStore: ProjectDraftStore = {
      ...backingStore,
      stage: vi.fn((input) => (rejectStages ? Promise.resolve({ kind: 'quota' as const }) : backingStore.stage(input))),
    };
    const saveDraftAsNew = vi.fn();
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      saveDraftAsNew,
      writerToken: 'writer-1',
    });
    const loaded = await service.loadWorkbench();
    rejectStages = true;
    const newerProject = {
      ...loaded.state.projects[0]!,
      settings: {
        ...loaded.state.projects[0]!.settings,
        useCpuNoise: !loaded.state.projects[0]!.settings.useCpuNoise,
      },
    };
    await service.saveWorkbench(stateWith([newerProject]));

    await expect(service.resolveConflictSaveAsNew(newerProject)).rejects.toThrow('up-to-date local recovery draft');
    expect(saveDraftAsNew).not.toHaveBeenCalled();
  });

  it('stages every project before starting any network write', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const projects = [createDraftProject([]), createDraftProject([])];
    let releaseFirst!: () => void;
    const firstWrite = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    const originalCreate = api.createProject;
    vi.mocked(api.createProject).mockImplementationOnce(async (request) => {
      await firstWrite;
      return originalCreate(request, owner);
    });

    const saving = createService(owner, api, draftStore).saveWorkbench(stateWith(projects));
    await vi.waitFor(async () => {
      await expect(draftStore.get(projects[1]!.id, 'editor-1')).resolves.toMatchObject({ kind: 'found' });
    });
    expect(api.createProject).toHaveBeenCalledTimes(1);
    releaseFirst();
    await saving;
  });

  it('bounds concurrent draft staging to two projects', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backingStore = createMemoryProjectDraftStore();
    let active = 0;
    let maximum = 0;
    const draftStore: ProjectDraftStore = {
      ...backingStore,
      stage: vi.fn(async (input) => {
        active += 1;
        maximum = Math.max(maximum, active);
        await Promise.resolve();
        await Promise.resolve();
        const result = await backingStore.stage(input);
        active -= 1;
        return result;
      }),
    };
    const projects = Array.from({ length: 6 }, () => createDraftProject([]));

    await createService(owner, api, draftStore).saveWorkbench(stateWith(projects));

    expect(maximum).toBe(2);
  });

  it('coalesces overlapping saves to the newest staged workbench state', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    let releaseCreate!: () => void;
    let signalCreateStarted!: () => void;
    const createStarted = new Promise<void>((resolve) => {
      signalCreateStarted = resolve;
    });
    const createReleased = new Promise<void>((resolve) => {
      releaseCreate = resolve;
    });
    const originalCreate = vi.mocked(api.createProject).getMockImplementation()!;
    vi.mocked(api.createProject).mockImplementationOnce(async (request, requestOwner) => {
      signalCreateStarted();
      await createReleased;
      return originalCreate(request, requestOwner);
    });
    const service = createService(owner, api);
    const firstProject = { ...project, name: 'first edit' };
    const latestProject = { ...project, name: 'latest edit' };
    const first = service.saveWorkbench(stateWith([firstProject]));
    await createStarted;
    const latest = service.saveWorkbench(stateWith([latestProject]));
    releaseCreate();

    await Promise.all([first, latest]);

    expect(api.records.get(project.id)?.name).toBe('latest edit');
    expect(api.updateProject).toHaveBeenCalledOnce();
  });

  it('persists an explicitly empty session when leaving the editor', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);

    await createService(owner, api).persistEmptySession(stateWith([project]));

    expect(api.saveSession).toHaveBeenCalledWith(
      expect.objectContaining({ activeProjectId: '', projects: [] }),
      'editor-1',
      {},
      owner.signal
    );
  });

  it('preserves another tab draft when deleting a clean server project', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'other tab edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-2',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'writer-2',
    });

    await createService(owner, api, draftStore).deleteProjectOnServer(project.id);

    await expect(draftStore.get(project.id, 'editor-2')).resolves.toMatchObject({ kind: 'found' });
  });

  it('does not acknowledge deletion when its owned draft cannot be removed', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backingStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    await backingStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'writer-1',
    });
    let reads = 0;
    const draftStore: ProjectDraftStore = {
      ...backingStore,
      delete: vi.fn(() => Promise.resolve({ kind: 'unavailable' as const })),
      get: vi.fn((projectId, editorSessionId) => {
        reads += 1;
        return reads <= 2
          ? backingStore.get(projectId, editorSessionId)
          : Promise.resolve({ kind: 'unavailable' as const });
      }),
    };
    const service = createService(owner, api, draftStore);

    await expect(service.deleteProjectOnServer(project.id)).rejects.toThrow('server project was deleted');
    await expect(service.deleteProjectOnServer(project.id)).rejects.toThrow('Local project recovery must be available');
    service.unmarkProjectDeleted(project.id);
    await expect(service.flushProjectToServer({ ...project, name: 'newer edit' })).resolves.toMatchObject({
      kind: 'superseded',
    });
    expect(api.deleteProject).toHaveBeenCalledOnce();
    expect(api.createProject).not.toHaveBeenCalled();
    expect(api.updateProject).not.toHaveBeenCalled();
  });

  it('rejects an oversized document before issuing a server write', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = {
      ...createDraftProject([]),
      promptHistory: [{ negativePrompt: null, positivePrompt: 'x'.repeat(32 * 1024 * 1024) }],
    };

    const result = await createService(owner, api).saveWorkbench(stateWith([project]));

    expect(result.error).toContain('the maximum is');
    expect(result.hasPendingChanges).toBe(true);
    expect(api.createProject).not.toHaveBeenCalled();
    expect(api.updateProject).not.toHaveBeenCalled();
  });

  it('surfaces a create-time server size refusal without retrying it', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const refusal = new ApiError(
      JSON.stringify({
        detail: { actual_bytes: 1025, code: 'project_request_too_large', max_bytes: 1024 },
      }),
      413
    );
    vi.mocked(api.createProject).mockRejectedValueOnce(new ProjectCreateAbsentError(refusal));

    const result = await createService(owner, api).saveWorkbench(stateWith([createDraftProject([])]));

    expect(result.error).toContain('the maximum is 1024 bytes');
    expect(result.shouldRetry).toBe(false);
  });

  it('does not retry another deterministic create refusal', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    vi.mocked(api.createProject).mockRejectedValueOnce(
      new ProjectCreateAbsentError(new ApiError('The project request is invalid.', 422))
    );

    const result = await createService(owner, api).saveWorkbench(stateWith([createDraftProject([])]));

    expect(result.error).toBe('The project request is invalid.');
    expect(result.shouldRetry).toBe(false);
  });

  it.each([400, 422])('does not retry a deterministic update refusal (%s)', async (status) => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api);
    const loaded = await service.loadWorkbench();
    vi.mocked(api.updateProject).mockRejectedValueOnce(new ApiError(`Rejected with ${status}.`, status));

    const result = await service.saveWorkbench(
      stateWith([{ ...loaded.state.projects[0]!, name: 'deterministically rejected' }])
    );

    expect(result.error).toBe(`Rejected with ${status}.`);
    expect(result.shouldRetry).toBe(false);
    expect(api.updateProject).toHaveBeenCalledOnce();
  });

  it('keeps a newer staged generation after an older network acknowledgement', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    const service = createService(owner, api, draftStore);
    const originalCreate = api.createProject;

    vi.mocked(api.createProject).mockImplementationOnce(async (request) => {
      await draftStore.stage({
        baseRevision: null,
        documentJson: JSON.stringify({ ...request.data, name: 'newer edit' }),
        documentSchemaVersion: 3,
        editorSessionId: 'editor-1',
        generation: 2,
        projectId: project.id,
        updatedAt: Date.parse(now) + 1,
        writerToken: 'writer-1',
      });
      return originalCreate(request, owner);
    });

    await service.saveWorkbench(stateWith([project]));

    await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({
      draft: { baseRevision: 1, generation: 2 },
      kind: 'found',
    });
    expect(service.hasPendingChanges()).toBe(true);
  });

  it('persists a divergent revision conflict and does not auto-fork', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api, draftStore);
    const loaded = await service.loadWorkbench();
    const local = { ...loaded!.state.projects[0]!, name: 'local edit' };
    api.records.set(project.id, toRecord({ ...project, name: 'remote edit' }, 2));

    const result = await service.saveWorkbench(stateWith([local]));

    expect(result.conflicts).toEqual([
      expect.objectContaining({ kind: 'revision', projectId: project.id, serverRevision: 2 }),
    ]);
    // Only the user can resolve it: the save reports a decision, and nothing a retry could push.
    expect(result).toMatchObject({
      awaitsDecision: true,
      error: null,
      hasPendingChanges: true,
      hasRetryablePendingChanges: false,
      shouldRetry: false,
    });
    expect(api.createProject).not.toHaveBeenCalled();
    await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({
      draft: { conflict: { kind: 'revision', serverRevision: 2 }, state: 'conflict' },
      kind: 'found',
    });
  });

  it('reports a schema refusal as a decision that no retry can clear', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api);
    const loaded = await service.loadWorkbench();
    vi.mocked(api.updateProject).mockRejectedValueOnce(canvasSchemaRefusal());

    const result = await service.saveWorkbench(stateWith([{ ...loaded.state.projects[0]!, name: 'local edit' }]));

    expect(result).toMatchObject({
      awaitsDecision: true,
      conflicts: [],
      error: null,
      hasPendingChanges: true,
      hasRetryablePendingChanges: false,
      shouldRetry: false,
    });
    expect(getProjectSyncSnapshot().projects[project.id]).toMatchObject({
      schemaRefusal: { kind: 'canvas', maxCanvasSchemaVersion: 3, minimumCanvasSchemaVersion: 4 },
    });
  });

  it('keeps retrying another project while one awaits a decision, then reports only the decision', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const conflicted = createDraftProject([]);
    const rebased = createDraftProject([]);
    api.records.set(conflicted.id, toRecord(conflicted));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: conflicted.id,
      openProjectIds: [conflicted.id],
    });
    const service = createService(owner, api, draftStore);
    const loaded = await service.loadWorkbench();
    api.records.set(conflicted.id, toRecord({ ...conflicted, name: 'remote edit' }, 2));
    const originalCreate = api.createProject;
    // A newer draft of the second project is staged while its creation is in flight, so it stays pending once acknowledged.
    vi.mocked(api.createProject).mockImplementationOnce(async (request) => {
      await draftStore.stage({
        baseRevision: null,
        documentJson: JSON.stringify({ ...request.data, name: 'newer edit' }),
        documentSchemaVersion: 3,
        editorSessionId: 'editor-1',
        generation: 2,
        projectId: rebased.id,
        updatedAt: Date.parse(now) + 1,
        writerToken: 'writer-1',
      });
      return originalCreate(request, owner);
    });

    const first = await service.saveWorkbench(
      stateWith([{ ...loaded.state.projects[0]!, name: 'local edit' }, rebased])
    );

    expect(first.conflicts).toEqual([expect.objectContaining({ kind: 'revision', projectId: conflicted.id })]);
    expect(first).toMatchObject({ awaitsDecision: true, hasPendingChanges: true, hasRetryablePendingChanges: true });

    const second = await service.saveWorkbench(
      stateWith([
        { ...loaded.state.projects[0]!, name: 'local edit' },
        { ...rebased, name: 'newer edit' },
      ])
    );

    expect(api.records.get(rebased.id)).toMatchObject({ name: 'newer edit', revision: 2 });
    expect(second.conflicts).toEqual([expect.objectContaining({ kind: 'revision', projectId: conflicted.id })]);
    expect(second).toMatchObject({ awaitsDecision: true, hasPendingChanges: true, hasRetryablePendingChanges: false });
  });

  it('stops reporting pending work for a project whose deletion left only the local draft behind', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backingStore = createMemoryProjectDraftStore();
    const draftStore: ProjectDraftStore = {
      ...backingStore,
      delete: vi.fn(() => Promise.resolve({ kind: 'unavailable' as const })),
    };
    const project = createDraftProject([]);
    vi.mocked(api.createProject).mockRejectedValueOnce(new ApiError('Service unavailable.', 503));
    const service = createService(owner, api, draftStore);

    const unsynced = await service.saveWorkbench(stateWith([project]));
    expect(unsynced).toMatchObject({ error: null, hasPendingChanges: true, shouldRetry: true });

    // The server delete commits, so the project stays fenced as deleted even though the local removal failed.
    await expect(service.deleteProjectOnServer(project.id)).rejects.toThrow('Retry deletion.');
    expect(api.deleteProject).toHaveBeenCalledWith(project.id, expect.anything(), 'release');

    const later = await service.saveWorkbench(stateWith([project]));

    expect(later).toMatchObject({
      awaitsDecision: false,
      error: null,
      hasPendingChanges: false,
      hasRetryablePendingChanges: false,
      shouldRetry: false,
    });
    expect(service.hasPendingChanges()).toBe(false);
    expect(api.createProject).toHaveBeenCalledOnce();
  });

  it('forgets a schema refusal when its project is deleted', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api);
    const loaded = await service.loadWorkbench();
    vi.mocked(api.updateProject).mockRejectedValueOnce(canvasSchemaRefusal());
    const refused = await service.saveWorkbench(stateWith([{ ...loaded.state.projects[0]!, name: 'local edit' }]));
    expect(refused.awaitsDecision).toBe(true);

    await service.deleteProjectOnServer(project.id);
    // The library broker closes the tab separately; an autosave can land first.
    const result = await service.saveWorkbench(stateWith([{ ...loaded.state.projects[0]!, name: 'local edit' }]));

    expect(result).toMatchObject({ awaitsDecision: false, hasPendingChanges: false });
    expect(getProjectSyncSnapshot().projects[project.id]?.schemaRefusal).toBeUndefined();
  });

  it('restores an unresolved conflict and its local document after reload', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project, 2));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 4,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    await draftStore.settleConflict(project.id, 'editor-1', 'old-writer', 4, {
      kind: 'revision',
      serverRevision: 2,
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });

    const second = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-2', release: () => Promise.resolve() }),
      writerToken: 'writer-2',
    });
    const loaded = await second.loadWorkbench();

    expect(loaded?.state.projects).toMatchObject([{ id: project.id, name: 'local edit' }]);
    expect(loaded?.conflicts).toEqual([
      expect.objectContaining({ kind: 'revision', projectId: project.id, serverRevision: 2 }),
    ]);
  });

  it('does not rewrite an unchanged conflicted draft during unrelated saves', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project, 2));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    await draftStore.settleConflict(project.id, 'editor-1', 'old-writer', 1, {
      kind: 'revision',
      serverRevision: 2,
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api, draftStore);
    const loaded = await service.loadWorkbench();
    const stage = vi.spyOn(draftStore, 'stage');

    await service.saveWorkbench(loaded.state);

    expect(stage).not.toHaveBeenCalled();
    expect(api.updateProject).not.toHaveBeenCalled();
  });

  it('guards an older local draft when the authoritative document requires a newer client', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    const futureRecord = toRecord(project, 2);
    futureRecord.data = { ...futureRecord.data, documentSchemaVersion: 5 };
    api.records.set(project.id, futureRecord);
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api, draftStore);

    const loaded = await service.loadWorkbench();
    const saved = await service.saveWorkbench(loaded.state);

    expect(loaded.state.projects).toMatchObject([{ id: project.id, name: 'local edit' }]);
    expect(saved.hasPendingChanges).toBe(true);
    expect(api.createProject).not.toHaveBeenCalled();
    expect(api.updateProject).not.toHaveBeenCalled();
    await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({
      draft: {
        refusal: { documentSchemaVersion: 5, kind: 'document', maxDocumentSchemaVersion: 4 },
        state: 'schema-refused',
      },
      kind: 'found',
    });
  });

  it('keeps a future-format local draft raw and exportable without hydrating it', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    const futureDocumentJson = JSON.stringify({ ...serializeProjectDocumentV3(project), documentSchemaVersion: 4 });
    await draftStore.stage({
      baseRevision: 1,
      documentJson: futureDocumentJson,
      documentSchemaVersion: 4,
      editorSessionId: 'older-editor',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api, draftStore);

    const loaded = await service.loadWorkbench();
    const get = vi.spyOn(draftStore, 'get');
    const page = await service.listRecoverableDrafts();

    expect(loaded.state.projects).toMatchObject([{ id: project.id, name: project.name }]);
    expect(loaded.refusedProjects).toEqual([]);
    expect(page).toMatchObject({
      items: [expect.objectContaining({ editorSessionId: 'older-editor', projectId: project.id })],
      kind: 'available',
    });
    expect(get).not.toHaveBeenCalled();
    await expect(service.getRecoverableDraftDocument(project.id, 'older-editor')).resolves.toBe(futureDocumentJson);
  });

  it('does not overwrite a future-format draft when the visible server project is edited', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    const futureDocumentJson = JSON.stringify({ ...serializeProjectDocumentV3(project), documentSchemaVersion: 4 });
    await draftStore.stage({
      baseRevision: 1,
      documentJson: futureDocumentJson,
      documentSchemaVersion: 4,
      editorSessionId: 'older-editor',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api, draftStore);
    const loaded = await service.loadWorkbench();

    await service.saveWorkbench(stateWith([{ ...loaded.state.projects[0]!, name: 'safe server edit' }]));

    await expect(service.getRecoverableDraftDocument(project.id, 'older-editor')).resolves.toBe(futureDocumentJson);
    expect(api.records.get(project.id)?.name).toBe('safe server edit');
  });

  it('isolates a same-tab future draft without blocking server load or save', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    const futureDocumentJson = JSON.stringify({ ...serializeProjectDocumentV3(project), documentSchemaVersion: 4 });
    await draftStore.stage({
      baseRevision: 1,
      documentJson: futureDocumentJson,
      documentSchemaVersion: 4,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api, draftStore);
    const loaded = await service.loadWorkbench();

    await service.saveWorkbench(stateWith([{ ...loaded.state.projects[0]!, name: 'supported edit' }]));

    expect(api.records.get(project.id)?.name).toBe('supported edit');
    await expect(service.getRecoverableDraftDocument(project.id, 'editor-1')).resolves.toBe(futureDocumentJson);
  });

  it('rewrites a loaded legacy project once in the current document schema', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    const record = toRecord(project);
    delete record.data.documentSchemaVersion;
    api.records.set(project.id, record);
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api);
    const loaded = await service.loadWorkbench();

    await service.saveWorkbench(loaded.state);
    await service.saveWorkbench(loaded.state);

    // Split once, by one request: the content raises the document floor, and the caller's first workspace is stored
    // with it, so the server never holds the content alone.
    expect(api.updateProject).toHaveBeenCalledOnce();
    expect(vi.mocked(api.updateProject).mock.calls[0]![1]).toMatchObject({
      data: { documentSchemaVersion: 4 },
      minimum_document_schema_version: 4,
      workspace: { data: { documentSchemaVersion: 4 }, expected_revision: null },
    });
    expect(api.saveProjectWorkspace).not.toHaveBeenCalled();
    expect(api.records.get(project.id)).toMatchObject({
      minimum_document_schema_version: 4,
      workspace: { revision: 1 },
    });
    // Both halves acknowledged: a later edit to the workspace alone builds on the stored one.
    const edited = { ...loaded.state.projects[0]!, promptHistory: [{ negativePrompt: null, positivePrompt: 'later' }] };
    await service.saveWorkbench(stateWith([edited]));
    expect(api.updateProject).toHaveBeenCalledOnce();
    expect(vi.mocked(api.saveProjectWorkspace).mock.calls.map(([, request]) => request.expected_revision)).toEqual([1]);
    expect(api.records.get(project.id)).toMatchObject({ revision: 2, workspace: { revision: 2 } });
  });

  it.each([
    { baseRevision: 1, expectedConflict: false },
    { baseRevision: 2, expectedConflict: true },
  ])('overlays a durable draft when opening a project mid-session ($baseRevision)', async (testCase) => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    await draftStore.stage({
      baseRevision: testCase.baseRevision,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'older-editor',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    const service = createService(owner, api, draftStore);

    const result = await service.hydrateProjectFromServer(project.id);

    expect(result).toMatchObject({ project: { name: 'local edit' }, status: 'loaded' });
    await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({
      draft: { state: testCase.expectedConflict ? 'conflict' : 'dirty' },
      kind: 'found',
    });
  });

  it('adopts the newest crashed lineage and exposes every other lineage for recovery', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    for (const [editorSessionId, name, updatedAt] of [
      ['older-editor', 'older offline edit', 1],
      ['newer-editor', 'newer offline edit', 2],
    ] as const) {
      await draftStore.stage({
        baseRevision: 1,
        documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name })),
        documentSchemaVersion: 3,
        editorSessionId,
        generation: 1,
        projectId: project.id,
        updatedAt,
        writerToken: `${editorSessionId}-writer`,
      });
    }
    const service = createService(owner, api, draftStore);

    const result = await service.hydrateProjectFromServer(project.id);

    expect(result).toMatchObject({ project: { name: 'newer offline edit' }, status: 'loaded' });
    expect(getProjectSyncSnapshot().recoverableDrafts).toEqual([
      expect.objectContaining({ editorSessionId: 'older-editor', projectId: project.id }),
    ]);
    await expect(service.getRecoverableDraftDocument(project.id, 'older-editor')).resolves.toContain(
      'older offline edit'
    );
  });

  it('exposes crashed lineages beyond the per-project adapter page limit', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    for (let index = 0; index < 34; index += 1) {
      await draftStore.stage({
        baseRevision: 1,
        documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: `offline edit ${index}` })),
        documentSchemaVersion: 3,
        editorSessionId: `lineage-${index}`,
        generation: 1,
        projectId: project.id,
        updatedAt: index + 1,
        writerToken: `writer-${index}`,
      });
    }
    const list = vi.spyOn(draftStore, 'list');
    const listForProject = vi.spyOn(draftStore, 'listForProject');
    const service = createService(owner, api, draftStore);

    const result = await service.hydrateProjectFromServer(project.id);

    expect(result).toMatchObject({ project: { name: 'offline edit 33' }, status: 'loaded' });
    expect(getProjectSyncSnapshot().recoverableDrafts).toHaveLength(33);
    expect(getProjectSyncSnapshot().recoverableDrafts).toContainEqual(
      expect.objectContaining({ editorSessionId: 'lineage-0', projectId: project.id })
    );
    expect(list).not.toHaveBeenCalled();
    expect(listForProject).toHaveBeenCalledTimes(2);
  });

  it('keeps a future draft recoverable when opening its server project mid-session', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    const documentJson = JSON.stringify({ ...serializeProjectDocumentV3(project), documentSchemaVersion: 3 });
    await draftStore.stage({
      baseRevision: 1,
      documentJson,
      documentSchemaVersion: 4,
      editorSessionId: 'older-editor',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    const service = createService(owner, api, draftStore);

    const result = await service.hydrateProjectFromServer(project.id);

    expect(result).toMatchObject({ project: { name: project.name }, status: 'loaded' });
    await expect(service.getRecoverableDraftDocument(project.id, 'older-editor')).resolves.toBe(documentJson);
  });

  it('surfaces a draft when adoption storage fails instead of hiding it behind server data', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backingStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    await backingStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'unadopted edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'older-editor',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    const draftStore: ProjectDraftStore = {
      ...backingStore,
      adopt: vi.fn(() => Promise.resolve({ kind: 'unavailable' as const })),
    };
    const service = createService(owner, api, draftStore);

    const result = await service.hydrateProjectFromServer(project.id);

    expect(result).toMatchObject({ project: { name: project.name }, status: 'loaded' });
    expect(getProjectSyncSnapshot()).toMatchObject({
      localDraftStatus: 'unavailable',
      recoverableDrafts: [{ editorSessionId: 'older-editor', projectId: project.id }],
    });
  });

  it('deletes a selected future draft only when its recovery metadata is current', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    const updatedAt = Date.parse(now);
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify({ ...serializeProjectDocumentV3(project), documentSchemaVersion: 3 }),
      documentSchemaVersion: 4,
      editorSessionId: 'newer-editor',
      generation: 1,
      projectId: project.id,
      updatedAt,
      writerToken: 'newer-writer',
    });
    const service = createService(owner, api, draftStore);
    await service.hydrateProjectFromServer(project.id);

    await expect(service.deleteRecoverableDraft(project.id, 'newer-editor', 2, updatedAt)).rejects.toThrow(
      'changed before it could be deleted'
    );
    await expect(service.deleteRecoverableDraft(project.id, 'newer-editor', 1, updatedAt + 1)).rejects.toThrow(
      'changed before it could be deleted'
    );
    await service.deleteRecoverableDraft(project.id, 'newer-editor', 1, updatedAt);

    await expect(draftStore.get(project.id, 'newer-editor')).resolves.toMatchObject({ kind: 'empty' });
    await expect(service.listRecoverableDrafts()).resolves.toMatchObject({ items: [] });
  });

  it('allows exact deletion of an advertised malformed current-schema draft', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    const updatedAt = Date.parse(now);
    api.records.set(project.id, toRecord(project));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify({ documentSchemaVersion: 3, id: project.id }),
      documentSchemaVersion: 3,
      editorSessionId: 'broken-editor',
      generation: 1,
      projectId: project.id,
      updatedAt,
      writerToken: 'broken-writer',
    });
    const service = createService(owner, api, draftStore);
    await service.hydrateProjectFromServer(project.id);

    const [recoverable] = getProjectSyncSnapshot().recoverableDrafts;
    expect(recoverable).toEqual(expect.objectContaining({ projectId: project.id }));
    await service.deleteRecoverableDraft(project.id, recoverable!.editorSessionId, 1, updatedAt);

    await expect(draftStore.get(project.id, recoverable!.editorSessionId)).resolves.toMatchObject({ kind: 'empty' });
  });

  it.each([
    {
      baseRevision: 1,
      error: Object.assign(new Error('not found'), { status: 404 }),
      expectedState: 'conflict' as const,
    },
    {
      baseRevision: null,
      error: Object.assign(new Error('not found'), { status: 404 }),
      expectedState: 'dirty' as const,
    },
    {
      baseRevision: 1,
      error: new ApiError(
        JSON.stringify({
          detail: {
            code: 'canvas_schema_unsupported',
            max_canvas_schema_version: 3,
            minimum_canvas_schema_version: 4,
          },
        }),
        412
      ),
      expectedState: 'schema-refused' as const,
    },
  ])(
    'opens a durable draft mid-session when GET yields $expectedState',
    async ({ baseRevision, error, expectedState }) => {
      const owner = captureAccountScope();
      const api = createApi();
      const draftStore = createMemoryProjectDraftStore();
      const project = createDraftProject([]);
      await draftStore.stage({
        baseRevision,
        documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
        documentSchemaVersion: 3,
        editorSessionId: 'older-editor',
        generation: 1,
        projectId: project.id,
        updatedAt: Date.parse(now),
        writerToken: 'old-writer',
      });
      vi.mocked(api.getProject).mockRejectedValueOnce(error);
      const service = createService(owner, api, draftStore);

      const result = await service.hydrateProjectFromServer(project.id);

      expect(result).toMatchObject({ project: { name: 'local edit' }, status: 'loaded' });
      await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({
        draft: { state: expectedState },
        kind: 'found',
      });
    }
  );

  it('rejects a draft ownership race while aligning a mid-session load', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backingStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    await backingStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'older-editor',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'older-writer',
    });
    const draftStore: ProjectDraftStore = {
      ...backingStore,
      stage: vi.fn(() => Promise.resolve({ kind: 'fenced' as const })),
    };

    await expect(createService(owner, api, draftStore).hydrateProjectFromServer(project.id)).rejects.toBeInstanceOf(
      ProjectDraftWriteRejectedError
    );
    expect(api.updateProject).not.toHaveBeenCalled();
    expect(api.createProject).not.toHaveBeenCalled();
  });

  it('abandons a mid-session hydration when the owning account changes during the read', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    const record = toRecord(project);
    vi.mocked(api.getProject).mockImplementationOnce(() => {
      accountLifecycle.activate('different-account');
      return Promise.resolve(record);
    });

    await expect(createService(owner, api).hydrateProjectFromServer(project.id)).rejects.toThrow();

    expect(api.updateProject).not.toHaveBeenCalled();
    expect(api.createProject).not.toHaveBeenCalled();
  });

  it('reuses its durable copy identity when save-as-new is retried', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project, 2));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    await draftStore.settleConflict(project.id, 'editor-1', 'old-writer', 1, {
      kind: 'revision',
      serverRevision: 2,
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const responseLost = new TypeError('response lost');
    let committedCopy: ProjectWithWorkspaceDTO | null = null;
    const saveDraftAsNew = vi.fn().mockImplementation((input) => {
      if (!committedCopy) {
        committedCopy = {
          board_id: `board-${input.copyProjectId}`,
          created_at: now,
          data: structuredClone(input.document),
          minimum_canvas_schema_version: input.minimumCanvasSchemaVersion,
          minimum_document_schema_version: 3,
          name: input.name,
          project_id: input.copyProjectId,
          revision: 1,
          updated_at: now,
          workspace_updated_at: null,
          workspace: null,
        };
        api.records.set(input.copyProjectId, committedCopy);
        return Promise.reject(responseLost);
      }
      return Promise.resolve(structuredClone(committedCopy));
    });
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      saveDraftAsNew,
      writerToken: 'writer-1',
    });
    await service.loadWorkbench();

    await expect(service.resolveConflictSaveAsNew(project)).rejects.toBe(responseLost);
    const newerProject = { ...project, name: 'newer local edit' };
    await service.saveWorkbench(stateWith([newerProject]));
    const result = await service.resolveConflictSaveAsNew(newerProject);

    expect(saveDraftAsNew).toHaveBeenCalledTimes(2);
    expect(saveDraftAsNew.mock.calls[0]![0].copyProjectId).toBe(saveDraftAsNew.mock.calls[1]![0].copyProjectId);
    expect(result.targetProjectId).toBe(saveDraftAsNew.mock.calls[0]![0].copyProjectId);
    await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({ kind: 'retargeted' });
    await expect(draftStore.get(result.targetProjectId, 'editor-1')).resolves.toMatchObject({
      draft: { state: 'dirty' },
      kind: 'found',
    });
    expect(saveDraftAsNew.mock.calls[1]![0].document).toEqual(saveDraftAsNew.mock.calls[0]![0].document);

    await service.saveWorkbench(
      stateWith([{ ...project, settings: { ...project.settings, useCpuNoise: !project.settings.useCpuNoise } }])
    );

    expect(api.updateProject).toHaveBeenLastCalledWith(
      result.targetProjectId,
      expect.objectContaining({ expected_revision: 1 }),
      owner.signal
    );
    expect(api.records.has(project.id)).toBe(true);
  });

  it('preserves a quota-degraded background mutation while copy creation is in flight', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backingStore = createMemoryProjectDraftStore();
    const stage = vi.fn(backingStore.stage);
    const draftStore: ProjectDraftStore = { ...backingStore, stage };
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project, 2));
    await backingStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    await backingStore.settleConflict(project.id, 'editor-1', 'old-writer', 1, {
      kind: 'revision',
      serverRevision: 2,
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    let releaseCreate!: () => void;
    let signalCreateStarted!: () => void;
    const createStarted = new Promise<void>((resolve) => {
      signalCreateStarted = resolve;
    });
    const createReleased = new Promise<void>((resolve) => {
      releaseCreate = resolve;
    });
    const saveDraftAsNew = vi.fn(async (input) => {
      signalCreateStarted();
      await createReleased;
      const record: ProjectWithWorkspaceDTO = {
        board_id: `board-${input.copyProjectId}`,
        created_at: now,
        data: structuredClone(input.document),
        minimum_canvas_schema_version: input.minimumCanvasSchemaVersion,
        minimum_document_schema_version: 3,
        name: input.name,
        project_id: input.copyProjectId,
        revision: 1,
        updated_at: now,
        workspace_updated_at: null,
        workspace: null,
      };
      api.records.set(record.project_id, record);
      return record;
    });
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      saveDraftAsNew,
      writerToken: 'writer-1',
    });
    const loaded = await service.loadWorkbench();
    stage.mockClear();
    stage.mockResolvedValueOnce({ kind: 'quota' });
    const local = loaded.state.projects[0]!;
    const resolving = service.resolveConflictSaveAsNew(local);
    await createStarted;
    const backgroundEdit = {
      ...local,
      settings: { ...local.settings, useCpuNoise: !local.settings.useCpuNoise },
    };
    const saving = service.saveWorkbench(stateWith([backgroundEdit]));

    await vi.waitFor(() => expect(stage).toHaveBeenCalledOnce());
    releaseCreate();
    const copy = await resolving;
    expect(copy.project.settings).toEqual(serializeProjectDocumentV3(backgroundEdit).settings);
    expect(copy.project.name).toBe(`${local.name} (copy)`);
    await saving;

    expect(api.records.get(copy.targetProjectId)?.name).toBe(`${local.name} (copy)`);
    // Settings are the workspace's.
    expect(api.records.get(copy.targetProjectId)?.workspace?.data.settings).toEqual(
      serializeProjectDocumentV3(backgroundEdit).settings
    );
  });

  it('does not overwrite a newer captured draft when copy resolution was queued', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const source = createDraftProject([]);
    const other = createDraftProject([source]);
    api.records.set(source.id, toRecord(source, 2));
    api.records.set(other.id, toRecord(other));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...source, name: 'click-time edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: source.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    await draftStore.settleConflict(source.id, 'editor-1', 'old-writer', 1, {
      kind: 'revision',
      serverRevision: 2,
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: source.id,
      openProjectIds: [source.id],
    });
    let releaseCreate!: () => void;
    let signalCreate!: () => void;
    const createStarted = new Promise<void>((resolve) => {
      signalCreate = resolve;
    });
    const createReleased = new Promise<void>((resolve) => {
      releaseCreate = resolve;
    });
    const saveDraftAsNew = vi.fn(async (input) => {
      signalCreate();
      await createReleased;
      const record = toRecord({ ...source, id: input.copyProjectId, name: input.name });
      record.data = structuredClone(input.document);
      api.records.set(record.project_id, record);
      return record;
    });
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      saveDraftAsNew,
      writerToken: 'writer-1',
    });
    const loaded = await service.loadWorkbench();
    let releaseRead!: () => void;
    let signalRead!: () => void;
    const readStarted = new Promise<void>((resolve) => {
      signalRead = resolve;
    });
    const readReleased = new Promise<void>((resolve) => {
      releaseRead = resolve;
    });
    vi.mocked(api.getProject).mockImplementationOnce(async () => {
      signalRead();
      await readReleased;
      return structuredClone(api.records.get(other.id)!);
    });
    const blocking = service.hydrateProjectFromServer(other.id);
    await readStarted;
    const clickProject = loaded.state.projects[0]!;
    const resolving = service.resolveConflictSaveAsNew(clickProject);
    const backgroundEdit = { ...clickProject, name: 'newer background edit' };
    const saving = service.saveWorkbench(stateWith([backgroundEdit]));
    await vi.waitFor(async () => {
      await expect(draftStore.get(source.id, 'editor-1')).resolves.toMatchObject({
        draft: { documentJson: JSON.stringify(serializeProjectDocumentV3(backgroundEdit)) },
        kind: 'found',
      });
    });

    releaseRead();
    await blocking;
    await createStarted;
    await expect(draftStore.get(source.id, 'editor-1')).resolves.toMatchObject({
      draft: { documentJson: JSON.stringify(serializeProjectDocumentV3(backgroundEdit)) },
      kind: 'found',
    });
    releaseCreate();
    const [copy] = await Promise.all([resolving, saving]);

    expect(copy.project.name).toBe(`${backgroundEdit.name} (copy)`);
    expect(api.records.get(copy.targetProjectId)?.name).toBe(`${backgroundEdit.name} (copy)`);
    await expect(draftStore.get(copy.targetProjectId, 'editor-1')).resolves.toMatchObject({ kind: 'empty' });
  });

  it('does not move or alias a source draft when its copy opens during create', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const source = createDraftProject([]);
    api.records.set(source.id, toRecord(source, 2));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...source, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: source.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    await draftStore.settleConflict(source.id, 'editor-1', 'old-writer', 1, {
      kind: 'revision',
      serverRevision: 2,
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: source.id,
      openProjectIds: [source.id],
    });
    let copyProjectId = '';
    let releaseCreate!: () => void;
    let signalCreate!: () => void;
    const createStarted = new Promise<void>((resolve) => {
      signalCreate = resolve;
    });
    const createReleased = new Promise<void>((resolve) => {
      releaseCreate = resolve;
    });
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      saveDraftAsNew: async (input) => {
        copyProjectId = input.copyProjectId;
        signalCreate();
        await createReleased;
        return toRecord({ ...source, id: input.copyProjectId, name: input.name });
      },
      writerToken: 'writer-1',
    });
    const loaded = await service.loadWorkbench();
    const resolving = service.resolveConflictSaveAsNew(loaded.state.projects[0]!);
    await createStarted;
    registerOpenProject(copyProjectId, {
      close: vi.fn(),
      deleteOnServer: vi.fn(() => Promise.resolve()),
      flush: vi.fn(() => Promise.resolve({ documentJson: '', kind: 'acknowledged' as const })),
      current: vi.fn(() => undefined),
      flushPixels: vi.fn(() => Promise.resolve()),
      markDeleted: vi.fn(),
      rename: vi.fn(() => Promise.resolve()),
      unmarkDeleted: vi.fn(),
    });
    releaseCreate();

    await expect(resolving).rejects.toThrow('opened before recovery finished');
    await expect(draftStore.get(source.id, 'editor-1')).resolves.toMatchObject({
      draft: { copyProjectId, state: 'conflict' },
      kind: 'found',
    });
    await expect(draftStore.listRetargets()).resolves.toMatchObject({ items: [], kind: 'available' });
    unregisterOpenProject(copyProjectId);
  });

  it('refuses a copy identity that is already open before committing a retarget', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project, 2));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    await draftStore.settleConflict(project.id, 'editor-1', 'old-writer', 1, {
      kind: 'revision',
      serverRevision: 2,
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const saveDraftAsNew = vi.fn();
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      saveDraftAsNew,
      writerToken: 'writer-1',
    });
    const loaded = await service.loadWorkbench();
    const localProject = loaded.state.projects[0]!;
    const draftJson = await service.getProjectDraftDocument(project.id);
    const targetProjectId = await createDeterministicProjectId(
      ['workbench-conflict-copy-v1', project.id, '1', 'revision', '2', draftJson!].join('\u0000')
    );
    const target = { ...localProject, id: targetProjectId, name: `${localProject.name} (copy)` };
    service.adoptProjectRecord(toRecord(target));

    await expect(service.resolveConflictSaveAsNew(localProject)).rejects.toThrow('already open');

    expect(saveDraftAsNew).not.toHaveBeenCalled();
    expect(service.hasPendingChanges()).toBe(true);
  });

  it('requires a durable copy identity and settles one project after a lost response and reload', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backingStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project, 2));
    await backingStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    await backingStore.settleConflict(project.id, 'editor-1', 'old-writer', 1, {
      kind: 'revision',
      serverRevision: 2,
    });
    const reserveCopyIdentity = vi
      .fn(backingStore.reserveCopyIdentity)
      .mockResolvedValueOnce({ kind: 'quota' })
      .mockImplementation(backingStore.reserveCopyIdentity);
    const draftStore: ProjectDraftStore = { ...backingStore, reserveCopyIdentity };
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const createdIds: string[] = [];
    const saveDraftAsNew = vi.fn((input) => {
      createdIds.push(input.copyProjectId);
      const existing = api.records.get(input.copyProjectId);
      if (existing) {
        return Promise.resolve(existing);
      }
      const record = toRecord({ ...project, id: input.copyProjectId, name: input.name });
      api.records.set(input.copyProjectId, record);
      return Promise.reject(new TypeError('response lost'));
    });
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      saveDraftAsNew,
      writerToken: 'writer-1',
    });
    await service.loadWorkbench();

    await expect(service.resolveConflictSaveAsNew(project)).rejects.toThrow('retry-safe project identity');
    expect(saveDraftAsNew).not.toHaveBeenCalled();
    await expect(service.resolveConflictSaveAsNew(project)).rejects.toThrow('response lost');

    const reloadedService = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-2', release: () => Promise.resolve() }),
      now: () => now,
      saveDraftAsNew,
      writerToken: 'writer-2',
    });
    const reloaded = await reloadedService.loadWorkbench();
    await reloadedService.resolveConflictSaveAsNew(reloaded.state.projects.find((item) => item.id === project.id)!);

    expect(new Set(createdIds).size).toBe(1);
    expect([...api.records.keys()].filter((id) => id !== project.id)).toHaveLength(1);
  });

  it('restores an acknowledged copy when the tab closed before reducer handoff', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project, 2));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    await draftStore.settleConflict(project.id, 'editor-1', 'old-writer', 1, {
      kind: 'revision',
      serverRevision: 2,
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      editorSessionId: 'editor-1',
      openProjectIds: [project.id],
    });
    const saveDraftAsNew = vi.fn((input) => {
      const copy = toRecord({ ...project, id: input.copyProjectId, name: input.name });
      api.records.set(copy.project_id, copy);
      return Promise.resolve(copy);
    });
    const first = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      saveDraftAsNew,
      writerToken: 'writer-1',
    });
    await first.loadWorkbench();
    const copy = await first.resolveConflictSaveAsNew(project);

    const second = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-2', release: () => Promise.resolve() }),
      now: () => now,
      writerToken: 'writer-2',
    });
    const loaded = await second.loadWorkbench();

    expect(loaded.state.activeProjectId).toBe(copy.targetProjectId);
    expect(loaded.state.projects.map((item) => item.id)).toContain(copy.targetProjectId);
    expect(loaded.state.projects.map((item) => item.id)).not.toContain(project.id);

    const explicitOriginal = await createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-3', release: () => Promise.resolve() }),
      now: () => now,
      writerToken: 'writer-3',
    }).loadWorkbench({ openProjectId: project.id });
    expect(explicitOriginal.state.projects.map((item) => item.id)).toContain(project.id);
  });

  it('surfaces a durable copy once and consumes its handoff after committing the repaired session', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const source = createDraftProject([]);
    const other = createDraftProject([source]);
    api.records.set(source.id, toRecord(source, 2));
    api.records.set(other.id, toRecord(other));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...source, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: source.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    await draftStore.settleConflict(source.id, 'editor-1', 'old-writer', 1, {
      kind: 'revision',
      serverRevision: 2,
    });
    let sessionBlob: WorkbenchSessionBlob = {
      account: createInitialWorkbenchState().account,
      activeProjectId: source.id,
      editorSessionId: 'editor-1',
      openProjectIds: [source.id],
    };
    vi.mocked(api.loadSession).mockImplementation(() => Promise.resolve(structuredClone(sessionBlob)));
    vi.mocked(api.saveSession).mockImplementation((state, editorSessionId, draftEditorSessionIds) => {
      sessionBlob = {
        account: state.account,
        activeProjectId: state.activeProjectId,
        draftEditorSessionIds,
        editorSessionId,
        openProjectIds: state.projects.map((project) => project.id),
      };
      return Promise.resolve();
    });
    const saveDraftAsNew = vi.fn((input) => {
      const record = toRecord({ ...source, id: input.copyProjectId, name: input.name });
      api.records.set(record.project_id, record);
      return Promise.resolve(record);
    });
    const first = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      saveDraftAsNew,
      writerToken: 'writer-1',
    });
    const firstLoad = await first.loadWorkbench();
    const copy = await first.resolveConflictSaveAsNew(firstLoad.state.projects[0]!);
    sessionBlob = {
      account: createInitialWorkbenchState().account,
      activeProjectId: other.id,
      editorSessionId: 'other-tab',
      openProjectIds: [other.id],
    };
    vi.mocked(api.listProjects).mockResolvedValueOnce([toSummary(api.records.get(other.id)!)]);
    vi.mocked(api.saveSession).mockRejectedValueOnce(new Error('session offline'));

    const restoredService = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-2', release: () => Promise.resolve() }),
      now: () => now,
      writerToken: 'writer-2',
    });
    const restored = await restoredService.loadWorkbench();

    expect(restored.state.projects.map((project) => project.id)).toEqual([other.id, copy.targetProjectId]);
    await expect(draftStore.listRetargets()).resolves.toMatchObject({
      items: [{ targetProjectId: copy.targetProjectId }],
      kind: 'available',
    });
    restoredService.releaseProjectSync(copy.targetProjectId);
    await restoredService.persistEmptySession(restored.state);
    await expect(draftStore.listRetargets()).resolves.toMatchObject({ items: [], kind: 'available' });
    sessionBlob = {
      account: createInitialWorkbenchState().account,
      activeProjectId: other.id,
      editorSessionId: 'stale-tab',
      openProjectIds: [other.id],
    };
    const restoredAgain = await createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-3', release: () => Promise.resolve() }),
      now: () => now,
      writerToken: 'writer-3',
    }).loadWorkbench();
    expect(restoredAgain.state.projects.map((project) => project.id)).toEqual([other.id]);
  });

  it('opens and saves an explicitly requested original without redirecting it to its copy', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const source = createDraftProject([]);
    api.records.set(source.id, toRecord(source, 2));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...source, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: source.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    await draftStore.settleConflict(source.id, 'editor-1', 'old-writer', 1, {
      kind: 'revision',
      serverRevision: 2,
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: source.id,
      editorSessionId: 'editor-1',
      openProjectIds: [source.id],
    });
    const first = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      saveDraftAsNew: (input) => {
        const record = toRecord({ ...source, id: input.copyProjectId, name: input.name });
        api.records.set(record.project_id, record);
        return Promise.resolve(record);
      },
      writerToken: 'writer-1',
    });
    const firstLoad = await first.loadWorkbench();
    const copy = await first.resolveConflictSaveAsNew(firstLoad.state.projects[0]!);
    const originalService = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-2', release: () => Promise.resolve() }),
      now: () => now,
      writerToken: 'writer-2',
    });
    vi.mocked(api.saveSession).mockRejectedValueOnce(new Error('session offline'));
    const loaded = await originalService.loadWorkbench({ openProjectId: source.id });
    const original = loaded.state.projects.find((project) => project.id === source.id)!;

    await expect(draftStore.listRetargets()).resolves.toMatchObject({
      items: [{ projectId: source.id, targetProjectId: copy.targetProjectId }],
      kind: 'available',
    });

    await originalService.saveWorkbench(stateWith([{ ...original, name: 'edited original' }, copy.project]));

    expect(api.records.get(source.id)?.name).toBe('edited original');
    expect(api.records.get(copy.targetProjectId)?.name).toBe(copy.name);
    await expect(draftStore.listRetargets()).resolves.toMatchObject({ items: [], kind: 'available' });
  });

  it('stops redirecting saves when the original is opened before the copy handoff autosaves', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const source = createDraftProject([]);
    api.records.set(source.id, toRecord(source, 2));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...source, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: source.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    await draftStore.settleConflict(source.id, 'editor-1', 'old-writer', 1, {
      kind: 'revision',
      serverRevision: 2,
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: source.id,
      openProjectIds: [source.id],
    });
    const service = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      saveDraftAsNew: (input) => {
        const record = toRecord({ ...source, id: input.copyProjectId, name: input.name });
        api.records.set(record.project_id, record);
        return Promise.resolve(record);
      },
      writerToken: 'writer-1',
    });
    const loaded = await service.loadWorkbench();
    const copy = await service.resolveConflictSaveAsNew(loaded.state.projects[0]!);
    const opened = await service.hydrateProjectFromServer(source.id);
    if (opened.status !== 'loaded') {
      throw new Error('Expected the original project to load.');
    }

    await service.saveWorkbench(
      stateWith([
        { ...opened.project, name: 'original after reopen' },
        { ...copy.project, name: 'copy stays separate' },
      ])
    );

    expect(api.records.get(source.id)?.name).toBe('original after reopen');
    expect(api.records.get(copy.targetProjectId)?.name).toBe('copy stays separate');
  });

  it('restores a copy retargeted through an isolated draft lineage', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project, 2));
    const futureDocumentJson = JSON.stringify({ ...serializeProjectDocumentV3(project), documentSchemaVersion: 4 });
    await draftStore.stage({
      baseRevision: 1,
      documentJson: futureDocumentJson,
      documentSchemaVersion: 4,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now) + 1,
      writerToken: 'future-writer',
    });
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'older-editor',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    let sessionBlob: WorkbenchSessionBlob = {
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      editorSessionId: 'editor-1',
      openProjectIds: [project.id],
    };
    vi.mocked(api.loadSession).mockImplementation(() => Promise.resolve(structuredClone(sessionBlob)));
    vi.mocked(api.saveSession).mockImplementation((state, editorSessionId, draftEditorSessionIds) => {
      sessionBlob = {
        account: state.account,
        activeProjectId: state.activeProjectId,
        draftEditorSessionIds,
        editorSessionId,
        openProjectIds: state.projects.map((item) => item.id),
      };
      return Promise.resolve();
    });
    const first = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      saveDraftAsNew: (input) => {
        const record = toRecord({ ...project, id: input.copyProjectId, name: input.name });
        api.records.set(record.project_id, record);
        return Promise.resolve(record);
      },
      writerToken: 'writer-1',
    });
    await first.loadWorkbench();
    const copy = await first.resolveConflictSaveAsNew(project);

    const second = createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-2', release: () => Promise.resolve() }),
      now: () => now,
      writerToken: 'writer-2',
    });
    const loaded = await second.loadWorkbench();

    expect(loaded.state.activeProjectId).toBe(copy.targetProjectId);
    expect(loaded.state.projects.map((item) => item.id)).not.toContain(project.id);
    await expect(second.getRecoverableDraftDocument(project.id, 'editor-1')).resolves.toBe(futureDocumentJson);
  });

  it('resumes a schema-refused draft after upgrading to a compatible client', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    await draftStore.settleSchemaRefusal(project.id, 'editor-1', 'old-writer', 1, {
      kind: 'canvas',
      maxCanvasSchemaVersion: 3,
      minimumCanvasSchemaVersion: 3,
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });

    const loaded = await createService(owner, api, draftStore).loadWorkbench();

    expect(loaded.conflicts).toEqual([]);
    await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({
      draft: { state: 'dirty' },
      kind: 'found',
    });
  });

  it.each([
    { documentSchemaVersion: 3, kind: 'document' as const, maxDocumentSchemaVersion: 1 },
    { kind: 'invalid-server-document' as const },
  ])('resumes a loadable $kind refusal after the client or server is corrected', async (refusal) => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    await draftStore.settleSchemaRefusal(project.id, 'editor-1', 'old-writer', 1, refusal);
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });

    const loaded = await createService(owner, api, draftStore).loadWorkbench();

    expect(loaded.conflicts).toEqual([]);
    await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({
      draft: { state: 'dirty' },
      kind: 'found',
    });
  });

  it('deletes the draft before accepting the server version', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project, 2));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    const service = createService(owner, api, draftStore);

    await service.loadWorkbench();
    const result = await service.resolveConflictUseServer(project.id);

    expect(result).toMatchObject({ project: { id: project.id, name: project.name }, status: 'loaded' });
    await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({ kind: 'empty' });
  });

  it('serializes a queued autosave behind accepting the server version', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backingStore = createMemoryProjectDraftStore();
    const stage = vi.fn(backingStore.stage);
    const draftStore: ProjectDraftStore = { ...backingStore, stage };
    const project = createDraftProject([]);
    const remote = { ...project, name: 'remote edit' };
    api.records.set(project.id, toSplitRecord(remote, 2));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'discarded local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api, draftStore);
    const loaded = await service.loadWorkbench();
    stage.mockClear();
    let releaseRead!: () => void;
    let signalRead!: () => void;
    const readStarted = new Promise<void>((resolve) => {
      signalRead = resolve;
    });
    const readReleased = new Promise<void>((resolve) => {
      releaseRead = resolve;
    });
    vi.mocked(api.getProject).mockImplementationOnce(async () => {
      signalRead();
      await readReleased;
      return structuredClone(api.records.get(project.id)!);
    });

    const resolving = service.resolveConflictUseServer(project.id);
    await readStarted;
    const staleSave = service.saveWorkbench(stateWith(loaded.state.projects));
    expect(stage).not.toHaveBeenCalled();
    releaseRead();
    const accepted = await resolving;
    if (accepted.status !== 'loaded') {
      throw new Error('Expected the server project to load.');
    }
    await staleSave;

    expect(api.records.get(project.id)?.name).toBe('remote edit');
    expect(stage).not.toHaveBeenCalled();
    await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({ kind: 'empty' });
    service.acknowledgeProjectResolution(project.id);
    await service.saveWorkbench(stateWith([accepted.project]));
  });

  it('does not restage a deleted conflict while discard and tab closure settle', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backingStore = createMemoryProjectDraftStore();
    const stage = vi.fn(backingStore.stage);
    const remove = vi.fn(backingStore.delete);
    const draftStore: ProjectDraftStore = { ...backingStore, delete: remove, stage };
    const project = createDraftProject([]);
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'discard me' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api, draftStore);
    const loaded = await service.loadWorkbench();
    stage.mockClear();
    let releaseDelete!: () => void;
    let signalDelete!: () => void;
    const deleteStarted = new Promise<void>((resolve) => {
      signalDelete = resolve;
    });
    const deleteReleased = new Promise<void>((resolve) => {
      releaseDelete = resolve;
    });
    remove.mockImplementationOnce(async (...args) => {
      signalDelete();
      await deleteReleased;
      return backingStore.delete(...args);
    });

    const discarding = service.resolveConflictDiscard(project.id);
    await deleteStarted;
    const staleSave = service.saveWorkbench(stateWith(loaded.state.projects));
    expect(stage).not.toHaveBeenCalled();
    releaseDelete();
    await discarding;
    await staleSave;

    expect(api.createProject).not.toHaveBeenCalled();
    expect(stage).not.toHaveBeenCalled();
    await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({ kind: 'empty' });
    service.acknowledgeProjectResolution(project.id);
    await service.saveWorkbench(stateWith([]));
  });

  it('does not leave a recreated draft when delete and autosave overlap', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backingStore = createMemoryProjectDraftStore();
    const stage = vi.fn(backingStore.stage);
    const draftStore: ProjectDraftStore = { ...backingStore, stage };
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api, draftStore);
    const loaded = await service.loadWorkbench();
    await service.saveWorkbench(stateWith([{ ...loaded.state.projects[0]!, name: 'local edit' }]));
    stage.mockClear();
    let releaseDelete!: () => void;
    let signalDelete!: () => void;
    const deleteStarted = new Promise<void>((resolve) => {
      signalDelete = resolve;
    });
    const deleteReleased = new Promise<void>((resolve) => {
      releaseDelete = resolve;
    });
    vi.mocked(api.deleteProject).mockImplementationOnce(async (projectId) => {
      signalDelete();
      await deleteReleased;
      api.records.delete(projectId);
    });

    const deleting = service.deleteProjectOnServer(project.id);
    await deleteStarted;
    const staleSave = service.saveWorkbench(stateWith(loaded.state.projects));
    expect(stage).not.toHaveBeenCalled();
    releaseDelete();
    await Promise.all([deleting, staleSave]);

    expect(api.records.has(project.id)).toBe(false);
    expect(stage).not.toHaveBeenCalled();
    await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({ kind: 'empty' });
  });

  it('keeps two tabs in separate draft lineages and surfaces the losing CAS as one conflict', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const tab = (id: string) =>
      createDurableSyncedWorkbenchPersistence(owner, {
        api,
        draftStore: Promise.resolve(draftStore),
        editorSession: Promise.resolve({ id, release: () => Promise.resolve() }),
        now: () => now,
        writerToken: `writer-${id}`,
      });
    const first = tab('tab-a');
    const second = tab('tab-b');
    const [firstLoad, secondLoad] = await Promise.all([first.loadWorkbench(), second.loadWorkbench()]);

    await first.saveWorkbench(stateWith([{ ...firstLoad.state.projects[0]!, name: 'first tab' }]));
    const result = await second.saveWorkbench(stateWith([{ ...secondLoad.state.projects[0]!, name: 'second tab' }]));

    expect(result.conflicts).toEqual([
      expect.objectContaining({ kind: 'revision', projectId: project.id, serverRevision: 2 }),
    ]);
    await expect(draftStore.listForProject(project.id)).resolves.toMatchObject({
      items: [expect.objectContaining({ editorSessionId: 'tab-b', state: 'conflict' })],
      kind: 'available',
    });
  });

  it('gives divergent drafts from the same server conflict distinct idempotent copy ids', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project, 2));
    for (const [editorSessionId, writerToken, name] of [
      ['tab-a', 'old-a', 'local A'],
      ['tab-b', 'old-b', 'local B'],
    ] as const) {
      await draftStore.stage({
        baseRevision: 1,
        documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name })),
        documentSchemaVersion: 3,
        editorSessionId,
        generation: 1,
        projectId: project.id,
        updatedAt: Date.parse(now),
        writerToken,
      });
      await draftStore.settleConflict(project.id, editorSessionId, writerToken, 1, {
        kind: 'revision',
        serverRevision: 2,
      });
    }
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const saveDraftAsNew = vi.fn((input) => {
      const record: ProjectWithWorkspaceDTO = {
        board_id: `board-${input.copyProjectId}`,
        created_at: now,
        data: structuredClone(input.document),
        minimum_canvas_schema_version: input.minimumCanvasSchemaVersion,
        minimum_document_schema_version: 3,
        name: input.name,
        project_id: input.copyProjectId,
        revision: 1,
        updated_at: now,
        workspace_updated_at: null,
        workspace: null,
      };
      api.records.set(record.project_id, record);
      return Promise.resolve(record);
    });
    const tab = async (editorSessionId: string, writerToken: string) => {
      const service = createDurableSyncedWorkbenchPersistence(owner, {
        api,
        draftStore: Promise.resolve(draftStore),
        editorSession: Promise.resolve({ id: editorSessionId, release: () => Promise.resolve() }),
        now: () => now,
        saveDraftAsNew,
        writerToken,
      });
      const loaded = await service.loadWorkbench();
      return { loaded, service };
    };
    const first = await tab('tab-a', 'writer-a');
    const second = await tab('tab-b', 'writer-b');

    const [copyA, copyB] = await Promise.all([
      first.service.resolveConflictSaveAsNew(first.loaded.state.projects[0]!),
      second.service.resolveConflictSaveAsNew(second.loaded.state.projects[0]!),
    ]);

    expect(copyA.targetProjectId).not.toBe(copyB.targetProjectId);
    expect(api.records.get(copyA.targetProjectId)?.name).toContain('local A');
    expect(api.records.get(copyB.targetProjectId)?.name).toContain('local B');
  });
});

describe('unload journal', () => {
  const openSession = (api: ReturnType<typeof createApi>, project: Project): void => {
    api.records.set(project.id, toRecord(project));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
  };
  /** The next page load in the same tab: same editor session, new writer. */
  const reload = (owner: AccountScope, api: DurableProjectPersistenceApi, draftStore: ProjectDraftStore) =>
    createDurableSyncedWorkbenchPersistence(owner, {
      api,
      deleteDatabase: () => Promise.resolve({ kind: 'deleted' }),
      draftStore: Promise.resolve(draftStore),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      writerToken: 'writer-2',
    });

  it('journals an edit no save has staged, and the next load recovers it as a pending draft', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    openSession(api, project);
    const service = createService(owner, api, draftStore);
    const loaded = await service.loadWorkbench();
    const edited = stateWith([{ ...loaded.state.projects[0]!, name: 'Typed before reload' }]);

    const outcome = service.journalBeforeUnload(edited);
    expect(outcome).toMatchObject({ kind: 'journaled', projectIds: [project.id], skippedOversizedProjectIds: [] });
    await expect(outcome.kind === 'journaled' ? outcome.written : null).resolves.toBe(true);
    // `pagehide` and becoming hidden both fire on unload; the second has nothing new.
    expect(service.journalBeforeUnload(edited)).toEqual({ kind: 'nothing-to-journal' });

    const next = reload(owner, api, draftStore);
    const recovered = await next.loadWorkbench();

    expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'Typed before reload' }]);
    expect(recovered.conflicts).toEqual([]);
    expect(next.hasPendingChanges()).toBe(true);
    await next.saveWorkbench(recovered.state);
    expect(api.records.get(project.id)).toMatchObject({ name: 'Typed before reload', revision: 2 });
  });

  it('journals nothing local recovery already holds, nor for a closed session or an ended account', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    openSession(api, project);
    const service = createService(owner, api);
    const loaded = await service.loadWorkbench();

    expect(service.journalBeforeUnload(loaded.state)).toEqual({ kind: 'nothing-to-journal' });
    const saved = stateWith([{ ...loaded.state.projects[0]!, name: 'Saved' }]);
    await service.saveWorkbench(saved);
    expect(service.journalBeforeUnload(saved)).toEqual({ kind: 'nothing-to-journal' });

    await service.persistEmptySession(saved);
    expect(service.journalBeforeUnload(stateWith([{ ...saved.projects[0]!, name: 'After close' }]))).toEqual({
      kind: 'nothing-to-journal',
    });

    const other = createService(captureAccountScope(), api);
    const otherLoaded = await other.loadWorkbench();
    accountLifecycle.invalidate();
    expect(other.journalBeforeUnload(stateWith([{ ...otherLoaded.state.projects[0]!, name: 'Late' }]))).toEqual({
      kind: 'nothing-to-journal',
    });
  });

  /** A project whose document is exactly `byteSize` bytes. */
  const projectOfSize = (byteSize: number, others: Project[] = []): Project => {
    const project = { ...createDraftProject(others), name: '' };
    const base = serializeProjectDocumentV3Json(project).byteSize;
    return { ...project, name: 'x'.repeat(byteSize - base) };
  };

  it('journals documents up to the budget together, and leaves the rest to autosave', async () => {
    const service = createService(captureAccountScope(), createApi());
    await service.loadWorkbench();
    const exact = projectOfSize(PROJECT_UNLOAD_JOURNAL_MAX_BYTES);
    const over = projectOfSize(PROJECT_UNLOAD_JOURNAL_MAX_BYTES + 1, [exact]);
    const half = projectOfSize(PROJECT_UNLOAD_JOURNAL_MAX_BYTES / 2 + 1, [exact, over]);
    const otherHalf = projectOfSize(PROJECT_UNLOAD_JOURNAL_MAX_BYTES / 2 + 1, [exact, over, half]);

    expect(service.journalBeforeUnload(stateWith([over, exact]))).toMatchObject({
      kind: 'journaled',
      projectIds: [exact.id],
      skippedOversizedProjectIds: [over.id],
    });
    // Each fits on its own; together they do not.
    expect(service.journalBeforeUnload(stateWith([half, otherHalf]))).toMatchObject({
      kind: 'journaled',
      projectIds: [half.id],
      skippedOversizedProjectIds: [otherHalf.id],
    });
  });

  it('keeps a journal it could not settle and settles it on the next save', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backing = createMemoryProjectDraftStore();
    let settleFailures = 1;
    const flaky: ProjectDraftStore = {
      ...backing,
      get availability() {
        return backing.availability;
      },
      settleUnloadJournal: (...args) => {
        if (settleFailures > 0) {
          settleFailures -= 1;
          return Promise.resolve({ kind: 'unavailable' });
        }
        return backing.settleUnloadJournal(...args);
      },
    };
    const project = createDraftProject([]);
    openSession(api, project);
    const service = createService(owner, api, flaky);
    const loaded = await service.loadWorkbench();

    service.journalBeforeUnload(stateWith([{ ...loaded.state.projects[0]!, name: 'Typed, then undone' }]));
    await service.saveWorkbench(loaded.state);
    await service.saveWorkbench(loaded.state);

    const reloaded = await reload(owner, api, backing).loadWorkbench();
    expect(reloaded.state.projects).toMatchObject([{ id: project.id, name: project.name }]);
  });

  it('writes a journal again on the next hidden event when the previous write aborted', async () => {
    const backing = createMemoryProjectDraftStore();
    let isWritten = false;
    const aborting: ProjectDraftStore = {
      ...backing,
      get availability() {
        return backing.availability;
      },
      journalBeforeUnload: (write) => {
        const started = backing.journalBeforeUnload(write);
        return started.kind === 'started' ? { kind: 'started', written: Promise.resolve(isWritten) } : started;
      },
    };
    const service = createService(captureAccountScope(), createApi(), aborting);
    const loaded = await service.loadWorkbench();
    const edited = stateWith([{ ...loaded.state.projects[0]!, name: 'Hidden' }]);

    const first = service.journalBeforeUnload(edited);
    await expect(first.kind === 'journaled' ? first.written : null).resolves.toBe(false);
    isWritten = true;

    expect(service.journalBeforeUnload(edited)).toMatchObject({ kind: 'journaled' });
    expect(service.journalBeforeUnload(edited)).toEqual({ kind: 'nothing-to-journal' });
  });

  it('journals nothing for a project being deleted', async () => {
    const api = createApi();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    const service = createService(captureAccountScope(), api);
    const loaded = await service.loadWorkbench();

    service.markProjectDeleted(project.id);

    expect(service.journalBeforeUnload(stateWith([{ ...loaded.state.projects[0]!, name: 'Being deleted' }]))).toEqual({
      kind: 'nothing-to-journal',
    });
  });

  it('loads, saves and recovers drafts as before when the journal is unavailable', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const backing = createMemoryProjectDraftStore();
    const withoutJournal: ProjectDraftStore = {
      ...backing,
      get availability() {
        return backing.availability;
      },
      journalBeforeUnload: () => ({ kind: 'unavailable' }),
      reconcileUnloadJournal: () => Promise.resolve({ kind: 'unavailable' }),
    };
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api, withoutJournal);
    const loaded = await service.loadWorkbench();
    expect(loaded.localDraftStatus).toBe('ok');

    const edited = stateWith([{ ...loaded.state.projects[0]!, name: 'Staged only' }]);
    expect(service.journalBeforeUnload(edited)).toEqual({ kind: 'unavailable' });
    vi.mocked(api.updateProject).mockRejectedValueOnce(new TypeError('Failed to fetch'));
    await expect(service.saveWorkbench(edited)).resolves.toMatchObject({ localDraftStatus: 'ok', shouldRetry: true });

    const recovered = await createDurableSyncedWorkbenchPersistence(owner, {
      api,
      draftStore: Promise.resolve(withoutJournal),
      editorSession: Promise.resolve({ id: 'editor-1', release: () => Promise.resolve() }),
      now: () => now,
      writerToken: 'writer-2',
    }).loadWorkbench();
    expect(recovered.localDraftStatus).toBe('ok');
    expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'Staged only' }]);
  });

  it('retires a journal with the acknowledgement that covers it, so a later load does not revive it', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    openSession(api, project);
    const service = createService(owner, api, draftStore);
    const loaded = await service.loadWorkbench();
    const hidden = stateWith([{ ...loaded.state.projects[0]!, name: 'While hidden' }]);

    // Hidden into the back/forward cache, then shown again: the save requested on hiding, then a newer edit, land.
    service.journalBeforeUnload(hidden);
    await service.saveWorkbench(hidden);
    await service.saveWorkbench(stateWith([{ ...hidden.projects[0]!, name: 'Newer' }]));

    const reloaded = await reload(owner, api, draftStore).loadWorkbench();
    expect(reloaded.state.projects).toMatchObject([{ id: project.id, name: 'Newer' }]);
    expect(reloaded.conflicts).toEqual([]);
  });

  it('shares the generation of a stage still in flight, so its acknowledgement retires the journal', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    let releaseStage!: () => void;
    const stageGate = new Promise<void>((resolve) => {
      releaseStage = resolve;
    });
    const stage = draftStore.stage.bind(draftStore);
    const stagesStarted: number[] = [];
    const gatedStore: ProjectDraftStore = {
      ...draftStore,
      get availability() {
        return draftStore.availability;
      },
      stage: async (input) => {
        stagesStarted.push(input.generation);
        await stageGate;
        return stage(input);
      },
    };
    const project = createDraftProject([]);
    openSession(api, project);
    const service = createService(owner, api, gatedStore);
    const loaded = await service.loadWorkbench();
    const edited = stateWith([{ ...loaded.state.projects[0]!, name: 'Saving' }]);

    const saving = service.saveWorkbench(edited);
    await vi.waitFor(() => expect(stagesStarted).toHaveLength(1));
    expect(service.journalBeforeUnload(edited)).toMatchObject({ kind: 'journaled' });
    releaseStage();
    await saving;

    await expect(draftStore.reconcileUnloadJournal('durable-sync-test', Date.parse(now))).resolves.toEqual({
      kind: 'available',
      outcomes: [],
    });
  });

  it('discards its journal when the project is deleted, so the next load cannot bring it back', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    openSession(api, project);
    const service = createService(owner, api, draftStore);
    const loaded = await service.loadWorkbench();

    service.journalBeforeUnload(stateWith([{ ...loaded.state.projects[0]!, name: 'Edited, then deleted' }]));
    await service.deleteProjectOnServer(project.id);

    const reloaded = await reload(owner, api, draftStore).loadWorkbench();
    expect(reloaded.state.projects.map(({ id }) => id)).not.toContain(project.id);
    expect(reloaded.conflicts).toEqual([]);
  });
});

describe('project content and the caller’s workspace', () => {
  const openSplitProject = async (record: ProjectWithWorkspaceDTO, draftStore = createMemoryProjectDraftStore()) => {
    const owner = captureAccountScope();
    const api = createApi();
    api.records.set(record.project_id, record);
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: record.project_id,
      openProjectIds: [record.project_id],
    });
    const service = createService(owner, api, draftStore);
    const loaded = await service.loadWorkbench();
    return { api, conflicts: loaded.conflicts, project: loaded.state.projects[0]!, service };
  };
  const withOtherSettings = (project: Project): Project => ({
    ...project,
    settings: { ...project.settings, useCpuNoise: !project.settings.useCpuNoise },
  });

  it('saves a change to how one works on the project as the workspace alone', async () => {
    const { api, project, service } = await openSplitProject(toSplitRecord(createDraftProject([])));

    await service.saveWorkbench(stateWith([withOtherSettings(project)]));

    expect(api.updateProject).not.toHaveBeenCalled();
    expect(vi.mocked(api.saveProjectWorkspace).mock.calls.map(([, request]) => request.expected_revision)).toEqual([1]);
    expect(api.records.get(project.id)).toMatchObject({ revision: 1, workspace: { revision: 2 } });
  });

  it('saves a change to the shared project as the content alone', async () => {
    const { api, project, service } = await openSplitProject(toSplitRecord(createDraftProject([])));

    await service.saveWorkbench(stateWith([{ ...project, name: 'Renamed' }]));

    expect(api.saveProjectWorkspace).not.toHaveBeenCalled();
    expect(vi.mocked(api.updateProject).mock.calls.map(([, request]) => request.expected_revision)).toEqual([1]);
    expect(api.records.get(project.id)).toMatchObject({ name: 'Renamed', revision: 2, workspace: { revision: 1 } });
  });

  it('replays a workspace a closed page never saved after its content, without a conflict', async () => {
    const project = { ...createDraftProject([]), name: 'Saved before closing' };
    const draftStore = createMemoryProjectDraftStore();
    // The page saved the content (revision 2) and was gone before saving the workspace or hearing back.
    await draftStore.stage({
      baseRevision: 1,
      baseWorkspaceRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3(withOtherSettings(project))),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });

    const {
      api,
      conflicts,
      project: loaded,
      service,
    } = await openSplitProject(toSplitRecord(project, 2, 1), draftStore);
    await service.saveWorkbench(stateWith([loaded]));

    expect(conflicts).toEqual([]);
    expect(api.updateProject).not.toHaveBeenCalled();
    expect(api.records.get(project.id)?.workspace).toMatchObject({
      data: { settings: withOtherSettings(project).settings },
      revision: 2,
    });
  });

  it('reports a workspace saved meanwhile from another tab as a conflict of the workspace alone', async () => {
    const project = createDraftProject([]);
    const draftStore = createMemoryProjectDraftStore();
    await draftStore.stage({
      baseRevision: 1,
      baseWorkspaceRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3(withOtherSettings(project))),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    const elsewhere = { ...project, promptHistory: [{ negativePrompt: null, positivePrompt: 'from another tab' }] };

    const { conflicts } = await openSplitProject(toSplitRecord(elsewhere, 1, 2), draftStore);

    expect(conflicts).toMatchObject([{ half: 'workspace', kind: 'revision', projectId: project.id }]);
  });

  it('stops saving a project a newer Invoke split further, keeping the local work', async () => {
    const draftStore = createMemoryProjectDraftStore();
    const { api, project, service } = await openSplitProject(toSplitRecord(createDraftProject([])), draftStore);
    vi.mocked(api.saveProjectWorkspace).mockRejectedValueOnce(
      new ApiError(
        JSON.stringify({
          detail: {
            code: 'document_schema_unsupported',
            max_document_schema_version: 4,
            minimum_document_schema_version: 5,
          },
        }),
        412
      )
    );

    const saved = await service.saveWorkbench(stateWith([withOtherSettings(project)]));

    expect(saved.hasPendingChanges).toBe(true);
    await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({
      draft: {
        refusal: { documentSchemaVersion: 5, kind: 'document', maxDocumentSchemaVersion: 4 },
        state: 'schema-refused',
      },
    });
  });
});

describe('a push that saved one half and not the other', () => {
  it('still saves the content when an undo returns to what was last acknowledged whole', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    api.records.set(project.id, toSplitRecord(project));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(owner, api);
    const loaded = (await service.loadWorkbench()).state.projects[0]!;
    // An edit to both halves: the content lands, the workspace does not.
    const edited = {
      ...loaded,
      name: 'Edited',
      settings: { ...loaded.settings, useCpuNoise: !loaded.settings.useCpuNoise },
    };
    vi.mocked(api.saveProjectWorkspace).mockRejectedValueOnce(new TypeError('Failed to fetch'));
    await service.saveWorkbench(stateWith([edited]));
    expect(api.records.get(project.id)?.name).toBe('Edited');

    // Undone before the retry: the server holds the edited content, so it must be saved back.
    await service.saveWorkbench(stateWith([loaded]));

    expect(api.records.get(project.id)).toMatchObject({ name: loaded.name, revision: 3 });
  });
});

describe('recovering a project whose halves were acknowledged apart', () => {
  const openWith = async (
    api: ReturnType<typeof createApi>,
    projectId: string,
    draftStore: ProjectDraftStore = createMemoryProjectDraftStore()
  ) => {
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: projectId,
      openProjectIds: [projectId],
    });
    const service = createService(captureAccountScope(), api, draftStore);
    const loaded = await service.loadWorkbench();
    return { conflicts: loaded.conflicts, project: loaded.state.projects[0]!, service };
  };

  it('builds on its own content save when the workspace save after it failed and the page went away', async () => {
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    api.records.set(project.id, toSplitRecord(project));
    const { project: loaded, service } = await openWith(api, project.id, draftStore);
    // The content lands; the workspace save after it does not.
    vi.mocked(api.saveProjectWorkspace).mockRejectedValueOnce(new TypeError('Failed to fetch'));
    await service.saveWorkbench(
      stateWith([
        { ...loaded, name: 'First', settings: { ...loaded.settings, useCpuNoise: !loaded.settings.useCpuNoise } },
      ])
    );
    expect(api.records.get(project.id)).toMatchObject({ name: 'First', revision: 2 });
    // A further edit reaches only the draft before the page goes away.
    vi.mocked(api.updateProject).mockRejectedValueOnce(new TypeError('Failed to fetch'));
    await service.saveWorkbench(stateWith([{ ...loaded, name: 'Second' }]));

    const reloaded = await openWith(api, project.id, draftStore);

    expect(reloaded.conflicts).toEqual([]);
    await reloaded.service.saveWorkbench(stateWith([reloaded.project]));
    expect(api.records.get(project.id)).toMatchObject({ name: 'Second', revision: 3 });
  });

  it('keeps this tab’s project edits when a workspace conflict is resolved with the saved workspace', async () => {
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    await draftStore.stage({
      baseRevision: 1,
      baseWorkspaceRevision: 1,
      documentJson: JSON.stringify(
        serializeProjectDocumentV3({
          ...project,
          name: 'Edited here',
          settings: { ...project.settings, useCpuNoise: !project.settings.useCpuNoise },
        })
      ),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    const elsewhere = { ...project, promptHistory: [{ negativePrompt: null, positivePrompt: 'from another tab' }] };
    api.records.set(project.id, toSplitRecord(elsewhere, 1, 2));
    const { conflicts, service } = await openWith(api, project.id, draftStore);
    expect(conflicts).toMatchObject([{ half: 'workspace' }]);

    const resolved = await service.resolveConflictUseServer(project.id);
    service.acknowledgeProjectResolution(project.id);
    if (resolved.status !== 'loaded') {
      throw new Error('Expected the resolution to load.');
    }
    await service.saveWorkbench(stateWith([resolved.project]));

    expect(resolved.project).toMatchObject({ name: 'Edited here', promptHistory: elsewhere.promptHistory });
    expect(api.records.get(project.id)).toMatchObject({ name: 'Edited here', revision: 2, workspace: { revision: 2 } });
  });
});

describe('a save the server answers with a revision conflict', () => {
  const open = async (record: ProjectWithWorkspaceDTO) => {
    const api = createApi();
    api.records.set(record.project_id, record);
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: record.project_id,
      openProjectIds: [record.project_id],
    });
    const service = createService(captureAccountScope(), api);
    const project = (await service.loadWorkbench()).state.projects[0]!;
    return { api, project, service };
  };
  const prompted = (project: Project, prompt: string): Project => ({
    ...project,
    promptHistory: [{ negativePrompt: null, positivePrompt: prompt }],
  });
  const conflict = () => Object.assign(new Error('conflict'), { status: 409 });

  it('reports another tab’s workspace as a conflict of the workspace, and sends it nothing more', async () => {
    const project = createDraftProject([]);
    const { api, project: loaded, service } = await open(toSplitRecord(project));
    // Another tab saves its own workspace meanwhile.
    const elsewhere = toSplitRecord(prompted(project, 'another tab'), 1, 2);
    api.records.set(project.id, elsewhere);

    const saved = await service.saveWorkbench(stateWith([prompted(loaded, 'this tab')]));

    expect(saved.conflicts).toMatchObject([{ half: 'workspace', kind: 'revision' }]);
    expect(api.saveProjectWorkspace).toHaveBeenCalledOnce();
    expect(api.records.get(project.id)?.workspace).toEqual(elsewhere.workspace);
  });

  it('acknowledges a workspace the server already holds as this tab’s', async () => {
    const project = createDraftProject([]);
    const { api, project: loaded, service } = await open(toSplitRecord(project));
    const edited = prompted(loaded, 'this tab');
    // The same bytes reached the server another way (another tab of this account typed the same).
    api.records.set(project.id, toSplitRecord(edited, 1, 2));

    const saved = await service.saveWorkbench(stateWith([edited]));

    expect(saved.conflicts).toEqual([]);
    expect(saved.hasPendingChanges).toBe(false);
    expect(api.saveProjectWorkspace).toHaveBeenCalledOnce();
  });

  it('retries a content save once when only the revision moved, and calls a second refusal a conflict', async () => {
    const project = createDraftProject([]);
    const retried = await open(toSplitRecord(project));
    // A save of the same bytes moved the revision on.
    retried.api.records.set(project.id, toSplitRecord(project, 2));
    await retried.service.saveWorkbench(stateWith([{ ...retried.project, name: 'Renamed' }]));
    expect(retried.api.updateProject).toHaveBeenCalledTimes(2);
    expect(retried.api.records.get(project.id)).toMatchObject({ name: 'Renamed', revision: 3 });

    const refused = await open(toSplitRecord(project));
    vi.mocked(refused.api.updateProject).mockRejectedValueOnce(conflict()).mockRejectedValueOnce(conflict());
    const saved = await refused.service.saveWorkbench(stateWith([{ ...refused.project, name: 'Renamed' }]));
    expect(refused.api.updateProject).toHaveBeenCalledTimes(2);
    expect(saved.conflicts).toMatchObject([{ kind: 'revision' }]);
    expect(saved.conflicts[0]).not.toHaveProperty('half');
  });

  it('retries the split of a project the server holds whole when only its revision moved', async () => {
    const project = createDraftProject([]);
    const { api, project: loaded, service } = await open(toRecord(project));
    api.records.set(project.id, toRecord(project, 2));

    const saved = await service.saveWorkbench(stateWith([{ ...loaded, name: 'Renamed' }]));

    expect(saved.conflicts).toEqual([]);
    // Refused once, read back as still whole, then split again by one request: never the content on its own.
    expect(vi.mocked(api.updateProject).mock.calls.map(([, request]) => request.expected_revision)).toEqual([1, 2]);
    expect(vi.mocked(api.updateProject).mock.calls.every(([, request]) => request.workspace)).toBe(true);
    expect(api.saveProjectWorkspace).not.toHaveBeenCalled();
    expect(api.records.get(project.id)).toMatchObject({
      minimum_document_schema_version: 4,
      name: 'Renamed',
      revision: 3,
      workspace: { revision: 1 },
    });
  });

  it('reports a workspace another tab stored under the whole project as a conflict of the workspace', async () => {
    const project = createDraftProject([]);
    const { api, project: loaded, service } = await open(toRecord(project));
    // A split by another tab of this account landed meanwhile, with its own workspace.
    api.records.set(project.id, toSplitRecord(prompted(project, 'another tab'), 2, 1));

    const saved = await service.saveWorkbench(stateWith([prompted(loaded, 'this tab')]));

    expect(saved.conflicts).toMatchObject([{ half: 'workspace', kind: 'revision' }]);
    expect(api.updateProject).toHaveBeenCalledOnce();
    expect(api.records.get(project.id)).toMatchObject({ revision: 2, workspace: { revision: 1 } });
  });
});

describe('splitting a project the server holds whole', () => {
  const open = async (record: ProjectWithWorkspaceDTO) => {
    const api = createApi();
    api.records.set(record.project_id, record);
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: record.project_id,
      openProjectIds: [record.project_id],
    });
    const service = createService(captureAccountScope(), api);
    const project = (await service.loadWorkbench()).state.projects[0]!;
    return { api, project, service };
  };

  it('leaves the server with both halves, never the content alone, when the account ends as the split lands', async () => {
    const project = createDraftProject([]);
    const { api, project: loaded, service } = await open(toRecord(project));
    const store = vi.mocked(api.updateProject).getMockImplementation()!;
    vi.mocked(api.updateProject).mockImplementationOnce(async (projectId, request, signal) => {
      const answer = await store(projectId, request, signal);
      // Signed out as the answer arrives: the account's drafts are deleted, and nothing more may be sent for it.
      accountLifecycle.activate('another-account');
      return answer;
    });

    const saved = await service.saveWorkbench(stateWith([{ ...loaded, name: 'Renamed' }]));

    expect(saved.error).toMatch(/account scope/);
    expect(api.updateProject).toHaveBeenCalledOnce();
    expect(api.saveProjectWorkspace).not.toHaveBeenCalled();
    expect(api.records.get(project.id)).toMatchObject({
      minimum_document_schema_version: 4,
      name: 'Renamed',
      workspace: { data: { documentSchemaVersion: 4, settings: loaded.settings }, revision: 1 },
    });
  });

  it('keeps the whole document when the account ends before the split is sent', async () => {
    const project = createDraftProject([]);
    const { api, project: loaded, service } = await open(toRecord(project));
    const before = structuredClone(api.records.get(project.id));
    vi.mocked(api.updateProject).mockImplementationOnce(() => {
      accountLifecycle.activate('another-account');
      return Promise.reject(new DOMException('Aborted', 'AbortError'));
    });

    const saved = await service.saveWorkbench(stateWith([{ ...loaded, name: 'Renamed' }]));

    expect(saved.error).toMatch(/account scope/);
    expect(api.saveProjectWorkspace).not.toHaveBeenCalled();
    expect(api.records.get(project.id)).toEqual(before);
  });

  it('sends the workspace on its own when the split’s answer names none', async () => {
    const project = createDraftProject([]);
    const { api, project: loaded, service } = await open(toRecord(project));
    // A server that stored the content and ignored the workspace sent with it.
    const store = vi.mocked(api.updateProject).getMockImplementation()!;
    vi.mocked(api.updateProject).mockImplementationOnce((projectId, request, signal) => {
      const { workspace: _workspace, ...content } = request;
      return store(projectId, content, signal);
    });

    const saved = await service.saveWorkbench(stateWith([{ ...loaded, name: 'Renamed' }]));

    expect(saved.conflicts).toEqual([]);
    expect(saved.hasPendingChanges).toBe(false);
    expect(api.updateProject).toHaveBeenCalledOnce();
    expect(vi.mocked(api.saveProjectWorkspace).mock.calls.map(([, request]) => request.expected_revision)).toEqual([
      null,
    ]);
    expect(api.records.get(project.id)).toMatchObject({
      minimum_document_schema_version: 4,
      name: 'Renamed',
      revision: 2,
      workspace: { revision: 1 },
    });
  });
});

describe('saving the two halves', () => {
  const open = async (record: ProjectWithWorkspaceDTO) => {
    const api = createApi();
    api.records.set(record.project_id, record);
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: record.project_id,
      openProjectIds: [record.project_id],
    });
    const service = createService(captureAccountScope(), api);
    const project = (await service.loadWorkbench()).state.projects[0]!;
    return { api, project, service };
  };
  const bothEdited = (project: Project): Project => ({
    ...project,
    name: 'Renamed',
    settings: { ...project.settings, useCpuNoise: !project.settings.useCpuNoise },
  });

  it('sends only the workspace on the next save when the content already landed', async () => {
    const { api, project, service } = await open(toSplitRecord(createDraftProject([])));
    vi.mocked(api.saveProjectWorkspace).mockRejectedValueOnce(new TypeError('Failed to fetch'));
    await service.saveWorkbench(stateWith([bothEdited(project)]));

    await service.saveWorkbench(stateWith([bothEdited(project)]));

    expect(api.updateProject).toHaveBeenCalledOnce();
    expect(api.saveProjectWorkspace).toHaveBeenCalledTimes(2);
    expect(api.records.get(project.id)).toMatchObject({ revision: 2, workspace: { revision: 2 } });
  });

  it('sends nothing of the workspace while the content it follows has not landed', async () => {
    const { api, project, service } = await open(toSplitRecord(createDraftProject([])));
    vi.mocked(api.updateProject).mockRejectedValueOnce(new TypeError('Failed to fetch'));

    const saved = await service.saveWorkbench(stateWith([bothEdited(project)]));

    expect(saved.hasPendingChanges).toBe(true);
    expect(api.saveProjectWorkspace).not.toHaveBeenCalled();
  });

  it('creates a new project with its content and the caller’s workspace in one request', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    const service = createService(owner, api);
    const halves = splitProjectDocument(serializeProjectDocumentV3(project));

    const saved = await service.saveWorkbench(stateWith([project]));

    // One request stored both halves: the server never held the content alone.
    expect(api.createProject).toHaveBeenCalledOnce();
    expect(vi.mocked(api.createProject).mock.calls[0]![0]).toMatchObject({
      data: { documentSchemaVersion: 4 },
      minimum_document_schema_version: 4,
      project_id: project.id,
      workspace: halves.workspace,
    });
    expect(api.saveProjectWorkspace).not.toHaveBeenCalled();
    expect(api.updateProject).not.toHaveBeenCalled();
    expect(saved.hasPendingChanges).toBe(false);
    expect(saved.projectBoardAssignments).toEqual([{ boardId: `board-${project.id}`, projectId: project.id }]);
    expect(api.records.get(project.id)).toMatchObject({ revision: 1, workspace: { revision: 1 } });

    // The board the server minted lands in the workspace, whose save builds on the revision the create answered.
    const onBoard = withAuthoritativeProjectBoard(project, `board-${project.id}`);
    await service.saveWorkbench(stateWith([onBoard]));
    expect(api.updateProject).not.toHaveBeenCalled();
    expect(vi.mocked(api.saveProjectWorkspace).mock.calls.map(([, request]) => request.expected_revision)).toEqual([1]);
    expect(api.records.get(project.id)).toMatchObject({ revision: 1, workspace: { revision: 2 } });
  });
});

describe('saving a conflicted draft as a new project', () => {
  it('creates the copy’s content and this tab’s workspace in one request', async () => {
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const source = createDraftProject([]);
    api.records.set(source.id, toSplitRecord(source, 2));
    const local = {
      ...source,
      name: 'local edit',
      settings: { ...source.settings, useCpuNoise: !source.settings.useCpuNoise },
    };
    await draftStore.stage({
      baseRevision: 1,
      baseWorkspaceRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3(local)),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: source.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: source.id,
      openProjectIds: [source.id],
    });
    const service = createService(captureAccountScope(), api, draftStore);
    const loaded = await service.loadWorkbench();
    expect(loaded.conflicts).toMatchObject([{ kind: 'revision' }]);

    const copy = await service.resolveConflictSaveAsNew(loaded.state.projects[0]!);

    expect(api.createProject).toHaveBeenCalledOnce();
    expect(vi.mocked(api.createProject).mock.calls[0]![0]).toMatchObject({
      data: { documentSchemaVersion: 4 },
      minimum_document_schema_version: 4,
      workspace: { documentSchemaVersion: 4, settings: local.settings },
    });
    expect(api.saveProjectWorkspace).not.toHaveBeenCalled();
    expect(api.records.get(copy.targetProjectId)?.workspace).toMatchObject({
      data: { settings: local.settings },
      revision: 1,
    });
    expect(copy.project.settings).toEqual(local.settings);
  });
});

describe('recovering a draft against a split project', () => {
  const recover = async (
    record: ProjectWithWorkspaceDTO,
    draft: { base: number; workspaceBase?: number; project: Project }
  ) => {
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    api.records.set(record.project_id, record);
    await draftStore.stage({
      baseRevision: draft.base,
      ...(draft.workspaceBase === undefined ? {} : { baseWorkspaceRevision: draft.workspaceBase }),
      documentJson: JSON.stringify(serializeProjectDocumentV3(draft.project)),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: record.project_id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: record.project_id,
      openProjectIds: [record.project_id],
    });
    const service = createService(captureAccountScope(), api, draftStore);
    return { api, loaded: await service.loadWorkbench(), service };
  };

  it('needs no base for a workspace the server already holds as the draft’s', async () => {
    // As a page writes a draft: from the project it loaded from the server.
    const loaded = deserializeProjectRecord(toSplitRecord(createDraftProject([])));
    if (loaded.status !== 'loaded') {
      throw new Error('Expected the project to load.');
    }
    const project = loaded.project;
    const settled = { ...project, settings: { ...project.settings, useCpuNoise: !project.settings.useCpuNoise } };
    // The page saved its workspace (revision 2) and was gone before it heard back; a rename after it never left.
    const recovered = await recover(toSplitRecord(settled, 1, 2), {
      base: 1,
      project: { ...settled, name: 'Renamed after' },
      workspaceBase: 1,
    });
    await recovered.service.saveWorkbench(recovered.loaded.state);

    expect(recovered.loaded.conflicts).toEqual([]);
    expect(recovered.api.records.get(project.id)).toMatchObject({
      name: 'Renamed after',
      revision: 2,
      workspace: { revision: 2 },
    });
  });

  it('replays a draft from before the split onto a project split meanwhile, with no workspace yet', async () => {
    const project = createDraftProject([]);
    const edited = { ...project, settings: { ...project.settings, useCpuNoise: !project.settings.useCpuNoise } };
    // Its content landed as the split (revision 2); the page went before saving the workspace. Its draft is a
    // record from before the split: no workspace base at all.
    const record = { ...toSplitRecord(project, 2), workspace: null, workspace_updated_at: null };

    const { api, loaded, service } = await recover(record, { base: 1, project: edited });
    await service.saveWorkbench(loaded.state);

    expect(loaded.conflicts).toEqual([]);
    expect(api.records.get(project.id)?.workspace).toMatchObject({ data: { settings: edited.settings }, revision: 1 });
  });

  it.each([
    [
      'canvas',
      { code: 'canvas_schema_unsupported', max_canvas_schema_version: 4, minimum_canvas_schema_version: 5 },
      { kind: 'canvas', minimumCanvasSchemaVersion: 5 },
    ],
    [
      'document',
      { code: 'document_schema_unsupported', max_document_schema_version: 4, minimum_document_schema_version: 5 },
      { documentSchemaVersion: 5, kind: 'document' },
    ],
  ])('keeps a draft refused by a newer %s schema on open, and saves nothing', async (_floor, detail, refusal) => {
    const project = createDraftProject([]);
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    api.records.set(project.id, toSplitRecord(project));
    await draftStore.stage({
      baseRevision: 1,
      documentJson: JSON.stringify(serializeProjectDocumentV3({ ...project, name: 'local edit' })),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    const service = createService(captureAccountScope(), api, draftStore);
    await service.loadWorkbench();
    vi.mocked(api.getProject).mockRejectedValueOnce(new ApiError(JSON.stringify({ detail }), 412));

    const opened = await service.hydrateProjectFromServer(project.id);

    expect(opened).toMatchObject({ project: { name: 'local edit' }, status: 'loaded' });
    await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({
      draft: { refusal, state: 'schema-refused' },
    });
  });

  it('refuses a listed project a newer document schema wrote without reading it', async () => {
    const project = createDraftProject([]);
    const api = createApi();
    api.records.set(project.id, { ...toSplitRecord(project), minimum_document_schema_version: 5 });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });

    const loaded = await createService(captureAccountScope(), api).loadWorkbench();

    expect(api.getProject).not.toHaveBeenCalled();
    expect(loaded.refusedProjects).toMatchObject([
      { projectId: project.id, refusal: { status: 'unsupported-version', version: 5 }, source: 'project-document' },
    ]);
  });
});

describe('resolving a workspace conflict with the saved workspace', () => {
  const conflicted = async () => {
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    await draftStore.stage({
      baseRevision: 1,
      baseWorkspaceRevision: 1,
      documentJson: JSON.stringify(
        serializeProjectDocumentV3({
          ...project,
          name: 'Edited here',
          settings: { ...project.settings, useCpuNoise: !project.settings.useCpuNoise },
        })
      ),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    const elsewhere = { ...project, promptHistory: [{ negativePrompt: null, positivePrompt: 'from another tab' }] };
    api.records.set(project.id, toSplitRecord(elsewhere, 1, 2));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(captureAccountScope(), api, draftStore);
    const loaded = await service.loadWorkbench();
    expect(loaded.conflicts).toMatchObject([{ half: 'workspace' }]);
    return { api, draftStore, elsewhere, project, service };
  };

  it('reports content another tab saved meanwhile as a conflict, and overwrites nothing', async () => {
    const { api, draftStore, elsewhere, project, service } = await conflicted();
    // Another tab saves the project's content after the conflict was raised.
    api.records.set(project.id, toSplitRecord({ ...elsewhere, name: 'Other tab content' }, 2, 2));

    await expect(service.resolveConflictUseServer(project.id)).rejects.toThrow('changed elsewhere as well');
    service.abortProjectResolution(project.id);
    await service.saveWorkbench(stateWith([{ ...project, name: 'Edited here' }]));

    expect(api.records.get(project.id)).toMatchObject({ name: 'Other tab content', revision: 2 });
    await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({
      draft: { conflict: { kind: 'revision', serverRevision: 2 }, state: 'conflict' },
    });
  });

  it('builds later edits on the saved workspace, so a page gone before saving them replays them', async () => {
    const { api, draftStore, project, service } = await conflicted();
    const resolved = await service.resolveConflictUseServer(project.id);
    service.acknowledgeProjectResolution(project.id);
    if (resolved.status !== 'loaded') {
      throw new Error('Expected the resolution to load.');
    }
    // A further workspace edit reaches only the draft before the page goes away.
    vi.mocked(api.saveProjectWorkspace).mockRejectedValueOnce(new TypeError('Failed to fetch'));
    await service.saveWorkbench(
      stateWith([{ ...resolved.project, promptHistory: [{ negativePrompt: null, positivePrompt: 'after' }] }])
    );

    const reloaded = await createService(captureAccountScope(), api, draftStore).loadWorkbench();

    expect(reloaded.conflicts).toEqual([]);
    expect(reloaded.state.projects[0]).toMatchObject({
      name: 'Edited here',
      promptHistory: [{ positivePrompt: 'after' }],
    });
  });

  it('keeps this tab’s project edits when the server holds no workspace of its own', async () => {
    const api = createApi();
    const draftStore = createMemoryProjectDraftStore();
    const project = createDraftProject([]);
    await draftStore.stage({
      baseRevision: 1,
      baseWorkspaceRevision: 1,
      documentJson: JSON.stringify(
        serializeProjectDocumentV3({
          ...project,
          name: 'Edited here',
          settings: { ...project.settings, useCpuNoise: !project.settings.useCpuNoise },
        })
      ),
      documentSchemaVersion: 3,
      editorSessionId: 'editor-1',
      generation: 1,
      projectId: project.id,
      updatedAt: Date.parse(now),
      writerToken: 'old-writer',
    });
    api.records.set(project.id, { ...toSplitRecord(project), workspace: null, workspace_updated_at: null });
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const service = createService(captureAccountScope(), api, draftStore);
    expect((await service.loadWorkbench()).conflicts).toMatchObject([{ half: 'workspace' }]);

    const resolved = await service.resolveConflictUseServer(project.id);
    service.acknowledgeProjectResolution(project.id);
    if (resolved.status !== 'loaded') {
      throw new Error('Expected the resolution to load.');
    }
    await service.saveWorkbench(stateWith([resolved.project]));

    expect(resolved.project).toMatchObject({ name: 'Edited here', settings: project.settings });
    expect(api.records.get(project.id)).toMatchObject({ name: 'Edited here', revision: 2, workspace: { revision: 1 } });
  });

  it('keeps the conflict when the merge cannot be staged', async () => {
    const { api, draftStore, project, service } = await conflicted();
    vi.spyOn(draftStore, 'stage').mockResolvedValueOnce({ kind: 'fenced' });

    await expect(service.resolveConflictUseServer(project.id)).rejects.toThrow();
    service.abortProjectResolution(project.id);
    const saved = await service.saveWorkbench(stateWith([{ ...project, name: 'Edited here' }]));

    expect(saved.conflicts).toMatchObject([{ half: 'workspace' }]);
    expect(api.saveProjectWorkspace).not.toHaveBeenCalled();
  });
});

describe('answers lost to the network', () => {
  it.each([
    [
      'content',
      'updateProject',
      (project: Project, edit: string): Project => ({ ...project, name: edit }),
      { name: 'third', revision: 3 },
    ],
    [
      'workspace',
      'saveProjectWorkspace',
      (project: Project, edit: string): Project => ({
        ...project,
        promptHistory: [{ negativePrompt: null, positivePrompt: edit }],
      }),
      { revision: 1, workspace: { data: { promptHistory: [{ positivePrompt: 'third' }] }, revision: 3 } },
    ],
  ] as const)(
    'knows an earlier %s save of its own when the edits after it are saved',
    async (_half, method, edit, expected) => {
      const api = createApi();
      const project = createDraftProject([]);
      api.records.set(project.id, toSplitRecord(project));
      vi.mocked(api.loadSession).mockResolvedValue({
        account: createInitialWorkbenchState().account,
        activeProjectId: project.id,
        openProjectIds: [project.id],
      });
      const service = createService(captureAccountScope(), api);
      const loaded = (await service.loadWorkbench()).state.projects[0]!;
      // The first save lands but its answer is lost; the next, of a further edit, is lost before it lands.
      const save = vi.mocked(api[method]).getMockImplementation()! as (...args: unknown[]) => Promise<unknown>;
      vi.mocked(api[method])
        .mockImplementationOnce(async (...args: unknown[]) => {
          await save(...args);
          throw new TypeError('Failed to fetch');
        })
        .mockRejectedValueOnce(new TypeError('Failed to fetch'));
      await service.saveWorkbench(stateWith([edit(loaded, 'first')]));
      await service.saveWorkbench(stateWith([edit(loaded, 'second')]));

      const saved = await service.saveWorkbench(stateWith([edit(loaded, 'third')]));

      expect(saved.conflicts).toEqual([]);
      expect(api.records.get(project.id)).toMatchObject(expected);
    }
  );
});

describe('a project from before the split whose workflow repeats an item id', () => {
  const withRepeatedNode = (base: Project): Project => {
    const [entry, ...rest] = base.workflows.entries;
    const node = { data: { label: '', notes: '' }, id: 'node-1', position: { x: 0, y: 0 }, type: 'notes' };
    return {
      ...base,
      workflows: {
        ...base.workflows,
        entries: [{ ...entry!, document: { ...entry!.document, nodes: [node, node] } }, ...rest],
      },
    } as unknown as Project;
  };
  const open = async (record: ProjectWithWorkspaceDTO) => {
    const api = createApi();
    api.records.set(record.project_id, record);
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: record.project_id,
      openProjectIds: [record.project_id],
    });
    const service = createService(captureAccountScope(), api);
    return { api, project: (await service.loadWorkbench()).state.projects[0]!, service };
  };

  it('keeps saving whole rather than writing content no load would accept', async () => {
    const project = withRepeatedNode(createDraftProject([]));
    const { api, project: loaded, service } = await open(toRecord(project));
    // Nothing to split, so nothing to send until it changes.
    await service.saveWorkbench(stateWith([loaded]));
    expect(api.updateProject).not.toHaveBeenCalled();

    const saved = await service.saveWorkbench(stateWith([{ ...loaded, name: 'Still whole' }]));

    expect(saved.error).toBeNull();
    expect(api.saveProjectWorkspace).not.toHaveBeenCalled();
    expect(api.records.get(project.id)).toMatchObject({
      data: { documentSchemaVersion: 3 },
      minimum_document_schema_version: 3,
      name: 'Still whole',
      revision: 2,
    });
  });

  it('is created whole, as an import or a recovered draft of one brings it back', async () => {
    const api = createApi();
    const project = withRepeatedNode(createDraftProject([]));
    const service = createService(captureAccountScope(), api);

    await service.saveWorkbench(stateWith([project]));
    const saved = await service.saveWorkbench(stateWith([{ ...project, name: 'Saved again' }]));

    expect(saved.error).toBeNull();
    expect(api.saveProjectWorkspace).not.toHaveBeenCalled();
    expect(vi.mocked(api.createProject).mock.calls[0]![0]).toMatchObject({
      data: { documentSchemaVersion: 3 },
      minimum_document_schema_version: 3,
    });
    expect(vi.mocked(api.createProject).mock.calls[0]![0]).not.toHaveProperty('workspace');
    expect(api.records.get(project.id)).toMatchObject({ name: 'Saved again', revision: 2, workspace: null });
  });

  it('refuses to save a split project once a workflow repeats an id, sending nothing', async () => {
    const { api, project, service } = await open(toSplitRecord(createDraftProject([])));

    const saved = await service.saveWorkbench(stateWith([withRepeatedNode({ ...project, name: 'Repeats' })]));

    expect(saved.error).toMatch(/repeats an item id/);
    expect(saved.hasPendingChanges).toBe(true);
    expect(api.updateProject).not.toHaveBeenCalled();
    expect(api.saveProjectWorkspace).not.toHaveBeenCalled();
  });
});
