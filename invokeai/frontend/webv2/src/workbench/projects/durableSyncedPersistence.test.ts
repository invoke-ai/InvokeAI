import type { AccountScope } from '@platform/state/accountLifecycle';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { ApiError, RequestDeadlineError, withRequestDeadline } from '@platform/transport/http';
import { stackTopAnchor } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { MAX_SUPPORTED_CANVAS_SCHEMA_VERSION } from '@workbench/canvasSchemaVersion';
import {
  createWorkbenchPersistenceRuntime,
  type PersistenceAggregatePort,
  type PersistenceClock,
} from '@workbench/persistenceRuntime';
import { createAccountOwnedQueueRunJournal } from '@workbench/queue-integration/queueRunJournal';
import { createEmptyPaintLayer } from '@workbench/widgets/layers/layerOps';
import {
  createDraftProject,
  createInitialWorkbenchState,
  withAuthoritativeProjectBoard,
} from '@workbench/workbenchState';
import { beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest';

import type { ProjectSummaryDTO, ProjectWithWorkspaceDTO } from './api';
import type { ProjectDraftStore } from './draftStore';
import type * as projectLifecycleLocksModule from './projectLifecycleLocks';
import type { WorkbenchSessionBlob } from './session';

import {
  createMemoryProjectDraftStore,
  createUnavailableProjectDraftStore,
  PROJECT_UNLOAD_JOURNAL_MAX_BYTES,
} from './draftStore';
import {
  createDurableSyncedWorkbenchPersistence,
  ProjectDraftWriteRejectedError,
  type DurableProjectPersistenceApi,
} from './durableSyncedPersistence';
import { serializeProjectDocumentV3, serializeProjectDocumentV3Json } from './projectDocument';
import { splitProjectDocument } from './projectHalves';
import { acquireProjectMutationLock } from './projectLifecycleLocks';
import { getProjectSyncSnapshot } from './syncStore';

const listQueueRunProjectIds = vi.hoisted(() =>
  vi.fn<() => Promise<{ kind: 'available'; projectIds: string[] } | { kind: 'unavailable' }>>(() =>
    Promise.resolve({ kind: 'available', projectIds: [] })
  )
);

vi.mock('./projectLifecycleLocks', async (importOriginal) => ({
  ...(await importOriginal<typeof projectLifecycleLocksModule>()),
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

/** A fake server; `clientMaxCanvasSchemaVersion` is the canvas schema its reader declares, refused below a floor. */
const createApi = (clientMaxCanvasSchemaVersion = MAX_SUPPORTED_CANVAS_SCHEMA_VERSION) => {
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
      if (record.minimum_canvas_schema_version > clientMaxCanvasSchemaVersion) {
        return Promise.reject(
          new ApiError(
            JSON.stringify({
              detail: {
                code: 'canvas_schema_unsupported',
                max_canvas_schema_version: clientMaxCanvasSchemaVersion,
                minimum_canvas_schema_version: record.minimum_canvas_schema_version,
              },
            }),
            412
          )
        );
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
  // These tests own their persistence API. Unrelated lifecycle reads must not open real HTTP sockets.
  vi.stubGlobal(
    'fetch',
    vi.fn(() => Promise.reject(new TypeError('Network unavailable in persistence unit test')))
  );
  onTestFinished(() => {
    vi.unstubAllGlobals();
  });
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

  it('sends an edit to the server only after its draft write has finished', async () => {
    const owner = captureAccountScope();
    const api = createApi();
    const project = createDraftProject([]);
    api.records.set(project.id, toRecord(project));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: project.id,
      openProjectIds: [project.id],
    });
    const backingStore = createMemoryProjectDraftStore();
    let held: Promise<void> | null = null;
    let finishWrite!: () => void;
    const stage = vi.fn(async (input: Parameters<ProjectDraftStore['stage']>[0]) => {
      await held;
      return backingStore.stage(input);
    });
    const service = createService(owner, api, { ...backingStore, stage });
    const loaded = await service.loadWorkbench();
    held = new Promise((resolve) => {
      finishWrite = resolve;
    });
    stage.mockClear();

    const saved = service.saveWorkbench(stateWith([{ ...loaded.state.projects[0]!, name: 'Renamed' }]));
    await vi.waitFor(() => expect(stage).toHaveBeenCalledOnce());
    expect(api.updateProject).not.toHaveBeenCalled();
    finishWrite();
    await saved;

    expect(api.updateProject).toHaveBeenCalledOnce();
    expect(api.records.get(project.id)?.name).toBe('Renamed');
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

  it.each([false, true])(
    'recovers a real mounted runtime and durable service after a hung save (queued edits: %s)',
    async (queuedEdits) => {
      vi.useFakeTimers();
      const owner = captureAccountScope();
      const api = createApi();
      const original = createDraftProject([]);
      const other = createDraftProject([original]);
      api.records.set(original.id, toSplitRecord(original));
      api.records.set(other.id, toSplitRecord(other));
      vi.mocked(api.loadSession).mockResolvedValue({
        account: createInitialWorkbenchState().account,
        activeProjectId: original.id,
        openProjectIds: [original.id],
      });
      const service = createService(owner, api);
      let state = stateWith([original]);
      let revision = 0;
      let hydrated = false;
      const events: string[] = [];
      const listeners = new Set<() => void>();
      const emit = () => {
        for (const listener of listeners) {
          listener();
        }
      };
      const aggregate: PersistenceAggregatePort = {
        assignProjectBoard: () => {},
        getPersistedRevision: () => revision,
        getState: () => state,
        hydrate: (next) => {
          state = next;
          revision += 1;
          emit();
        },
        notifyProjectNotFound: () => {},
        reportLoadAvailable: () => {},
        reportRefusedProjects: () => {},
        reportLoadError: () => {},
        reportLoadUnavailable: () => {},
        saveFailed: () => events.push('failed'),
        saveNeedsAttention: () => events.push('attention'),
        savePending: () => events.push('pending'),
        saveScheduled: () => {},
        saveStarted: () => events.push('started'),
        saveSucceeded: () => events.push('saved'),
        setHasHydrated: (next) => {
          hydrated = next;
          emit();
        },
        subscribe: (listener) => {
          listeners.add(listener);
          return () => {
            listeners.delete(listener);
          };
        },
      };
      const callbacks = new Map<number, () => void>();
      let clockId = 0;
      const clock: PersistenceClock = {
        clearTimeout: (id) => {
          callbacks.delete(id as number);
        },
        setTimeout: (callback) => {
          callbacks.set(++clockId, callback);
          return clockId;
        },
      };
      const runScheduled = () => {
        const pending = [...callbacks.values()];
        callbacks.clear();
        for (const callback of pending) {
          callback();
        }
      };
      const runtime = createWorkbenchPersistenceRuntime({
        aggregate,
        clock,
        persistence: service,
        signal: owner.signal,
      });
      try {
        runtime.start();
        await vi.waitFor(() => expect(hydrated).toBe(true));
        let stalledSignal!: AbortSignal;
        vi.mocked(api.updateProject).mockImplementationOnce((_id, _request, signal) =>
          withRequestDeadline(
            (requestSignal) => {
              stalledSignal = requestSignal;
              vi.mocked(api.getProject).mockRejectedValueOnce(new RequestDeadlineError());
              return new Promise(() => {});
            },
            150_000,
            signal
          )
        );
        state = { ...state, projects: [{ ...state.projects[0]!, name: 'First edit' }] };
        revision += 1;
        emit();
        runScheduled();
        await vi.waitFor(() => expect(api.updateProject).toHaveBeenCalledTimes(1));
        const switchAndEdit = () => {
          state = {
            ...state,
            activeProjectId: other.id,
            projects: [
              { ...state.projects[0]!, name: 'After timeout' },
              withAuthoritativeProjectBoard(other, `board-${other.id}`),
            ],
          };
          revision += 1;
          emit();
          runScheduled();
        };
        if (queuedEdits) {
          switchAndEdit();
        }
        await vi.advanceTimersByTimeAsync(150_000);
        expect(stalledSignal.aborted).toBe(true);
        if (!queuedEdits) {
          await vi.waitFor(() => expect(events).toContain('pending'));
          switchAndEdit();
        }
        await vi.waitFor(() => expect(api.records.get(original.id)?.name).toBe('After timeout'));
        await vi.waitFor(() => expect(events).toContain('saved'));
        expect(events).not.toContain('failed');
        expect(events).not.toContain('attention');
        expect(api.saveSession).toHaveBeenLastCalledWith(
          expect.objectContaining({ activeProjectId: other.id }),
          'editor-1',
          expect.anything(),
          owner.signal
        );
        expect(service.hasPendingChanges()).toBe(false);
      } finally {
        await runtime.exit().settled;
        await service.close();
        vi.useRealTimers();
      }
    }
  );

  it('retires answered content sends before resolving a later workspace conflict', async () => {
    const api = createApi();
    const original = createDraftProject([]);
    api.records.set(original.id, toSplitRecord(original));
    vi.mocked(api.loadSession).mockResolvedValue({
      account: createInitialWorkbenchState().account,
      activeProjectId: original.id,
      openProjectIds: [original.id],
    });
    const service = createService(captureAccountScope(), api);
    const loaded = (await service.loadWorkbench()).state.projects[0]!;
    const workspace = [{ negativePrompt: null, positivePrompt: 'Workspace X' }];
    vi.mocked(api.saveProjectWorkspace).mockRejectedValueOnce(new TypeError('Failed to fetch'));
    expect(
      await service.saveWorkbench(stateWith([{ ...loaded, name: 'Content B', promptHistory: workspace }]))
    ).toMatchObject({
      conflicts: [],
      shouldRetry: true,
    });
    api.records.set(
      original.id,
      toSplitRecord(
        {
          ...loaded,
          name: 'Content B',
          promptHistory: [{ negativePrompt: null, positivePrompt: 'Remote workspace Y' }],
        },
        2,
        2
      )
    );
    expect(
      (await service.saveWorkbench(stateWith([{ ...loaded, name: 'Content C', promptHistory: workspace }]))).conflicts
    ).toEqual([expect.objectContaining({ half: 'workspace' })]);
    const resolved = await service.resolveConflictUseServer(original.id);
    service.acknowledgeProjectResolution(original.id);
    if (resolved.status !== 'loaded') {
      throw new Error('Expected the workspace resolution to load.');
    }
    expect(resolved.project).toMatchObject({
      name: 'Content C',
      promptHistory: [{ positivePrompt: 'Remote workspace Y' }],
    });
    const contentWrites = vi.mocked(api.updateProject).mock.calls.length;
    const workspaceWrites = vi.mocked(api.saveProjectWorkspace).mock.calls.length;
    api.records.set(original.id, toSplitRecord({ ...resolved.project, name: 'Remote content D' }, 4, 2));
    expect(await service.saveWorkbench(stateWith([resolved.project]))).toMatchObject({
      conflicts: [],
      hasPendingChanges: false,
    });
    expect(api.updateProject).toHaveBeenCalledTimes(contentWrites);
    expect(api.saveProjectWorkspace).toHaveBeenCalledTimes(workspaceWrites);
    expect(api.records.get(original.id)?.name).toBe('Remote content D');
  });

  it.each(['content', 'workspace'] as const)(
    'acknowledges a committed %s undo fence from a higher revision despite a lost answer',
    async (half) => {
      const api = createApi();
      const original = createDraftProject([]);
      api.records.set(original.id, toSplitRecord(original));
      vi.mocked(api.loadSession).mockResolvedValue({
        account: createInitialWorkbenchState().account,
        activeProjectId: original.id,
        openProjectIds: [original.id],
      });
      const service = createService(captureAccountScope(), api);
      const loaded = (await service.loadWorkbench()).state.projects[0]!;
      const changed =
        half === 'content'
          ? { ...loaded, name: 'Undone edit' }
          : { ...loaded, promptHistory: [{ negativePrompt: null, positivePrompt: 'Undone edit' }] };
      const method = half === 'content' ? 'updateProject' : 'saveProjectWorkspace';
      const write = vi.mocked(api[method]).getMockImplementation()! as (...args: unknown[]) => Promise<unknown>;
      vi.mocked(api[method]).mockImplementationOnce(async (...args: unknown[]) => {
        await write(...args);
        vi.mocked(api.getProject).mockRejectedValueOnce(new RequestDeadlineError());
        throw new RequestDeadlineError();
      });
      expect(await service.saveWorkbench(stateWith([changed]))).toMatchObject({ shouldRetry: true });
      vi.mocked(api[method]).mockImplementation(async (...args: unknown[]) => {
        await write(...args);
        throw new RequestDeadlineError();
      });
      expect(await service.saveWorkbench(stateWith([loaded]))).toMatchObject({
        conflicts: [],
        hasPendingChanges: false,
      });
      const calls = vi.mocked(api[method]).mock.calls.length;
      expect(await service.saveWorkbench(stateWith([loaded]))).toMatchObject({
        conflicts: [],
        hasPendingChanges: false,
      });
      expect(api[method]).toHaveBeenCalledTimes(calls);
    }
  );

  it.each([false, true])(
    'bounds an unanswered create through API and persistence reconciliation (busy: %s)',
    async (busy) => {
      vi.useFakeTimers();
      const { createProjectSettled, getProject } = await import('./api');
      const api = createApi();
      api.createProject = createProjectSettled;
      api.getProject = getProject;
      const fetch = vi.fn(
        (): Promise<Response> =>
          new Promise(() => {
            /* Simulate an unresponsive request. */
          })
      );
      if (busy) {
        for (let attempt = 0; attempt < 8; attempt += 1) {
          fetch.mockResolvedValueOnce(
            new Response(JSON.stringify({ detail: { code: 'project_write_busy' } }), {
              status: 429,
              headers: { 'Retry-After': '0' },
            })
          );
        }
      }
      vi.stubGlobal('fetch', fetch);
      const random = vi.spyOn(Math, 'random').mockReturnValue(0);
      try {
        const service = createService(captureAccountScope(), api);
        const save = service.saveWorkbench(stateWith([createDraftProject([])]));
        let settled = false;
        void save.then(() => {
          settled = true;
        });
        // Each layer performs one bounded read after the final unanswered POST.
        const duration = busy ? 574_000 : 450_000;
        await vi.advanceTimersByTimeAsync(duration - 1);
        expect(settled).toBe(false);
        await vi.advanceTimersByTimeAsync(1);
        expect(await save).toMatchObject({ conflicts: [], hasPendingChanges: true, shouldRetry: true });
        expect(fetch).toHaveBeenCalledTimes(busy ? 11 : 3);
        expect(vi.getTimerCount()).toBe(0);
      } finally {
        random.mockRestore();
        vi.unstubAllGlobals();
        vi.useRealTimers();
      }
    }
  );

  it.each(['committed', 'uncommitted', 'diverged', 'unavailable', 'repeated-timeout', 'missing'] as const)(
    'reconciles a timed-out update with %s server state and preserves newer edits',
    async (outcome) => {
      const owner = captureAccountScope();
      const api = createApi();
      const draftStore = createMemoryProjectDraftStore();
      const project = createDraftProject([]);
      api.records.set(project.id, toSplitRecord(project));
      vi.mocked(api.loadSession).mockResolvedValue({
        account: createInitialWorkbenchState().account,
        activeProjectId: project.id,
        openProjectIds: [project.id],
      });
      const service = createService(owner, api, draftStore);
      const loaded = await service.loadWorkbench();
      const local = { ...loaded.state.projects[0]!, name: 'First edit' };
      const write = vi.mocked(api.updateProject).getMockImplementation()!;
      vi.mocked(api.updateProject).mockImplementationOnce(async (...args) => {
        if (outcome === 'committed') {
          await write(...args);
        }
        if (outcome === 'diverged') {
          api.records.set(project.id, toSplitRecord({ ...project, name: 'Other writer' }, 2));
        }
        if (outcome === 'unavailable') {
          vi.mocked(api.getProject).mockRejectedValueOnce(new RequestDeadlineError());
        }
        if (outcome === 'missing') {
          vi.mocked(api.getProject).mockRejectedValueOnce(new ApiError('not yet visible', 404));
        }
        if (outcome === 'repeated-timeout') {
          vi.mocked(api.updateProject).mockRejectedValueOnce(new RequestDeadlineError());
        }
        throw new RequestDeadlineError();
      });
      const result = await service.saveWorkbench(stateWith([local]));
      if (outcome === 'diverged') {
        expect(result.conflicts).toEqual([expect.objectContaining({ kind: 'revision' })]);
      } else if (['unavailable', 'repeated-timeout', 'missing'].includes(outcome)) {
        expect(result).toMatchObject({ hasPendingChanges: true, shouldRetry: true });
        await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({
          kind: 'found',
          draft: { state: 'dirty' },
        });
      } else {
        expect(result).toMatchObject({ hasPendingChanges: false, conflicts: [] });
        expect(api.updateProject).toHaveBeenCalledTimes(outcome === 'committed' ? 1 : 2);
      }
      if (outcome !== 'diverged') {
        const next = await service.saveWorkbench(stateWith([{ ...local, name: 'Newer edit' }]));
        expect(next).toMatchObject({ hasPendingChanges: false, conflicts: [] });
        expect(api.records.get(project.id)?.name).toBe('Newer edit');
      }
    }
  );

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

  describe('what an unload would lose', () => {
    it('counts an over-budget unsaved project until a save stages it', async () => {
      const service = createService(captureAccountScope(), createApi());
      await service.loadWorkbench();
      const small = createDraftProject([]);
      const over = projectOfSize(PROJECT_UNLOAD_JOURNAL_MAX_BYTES + 1, [small]);

      expect(service.journalBeforeUnload(stateWith([small]))).toMatchObject({ kind: 'journaled' });
      expect(service.hasChangesUnloadWouldLose()).toBe(false);

      const state = stateWith([small, over]);
      expect(service.journalBeforeUnload(state)).toMatchObject({ skippedOversizedProjectIds: [over.id] });
      expect(service.hasChangesUnloadWouldLose()).toBe(true);

      // A staged draft has no budget of its own below the document limit: local recovery now holds it.
      await service.saveWorkbench(state);
      expect(service.hasChangesUnloadWouldLose()).toBe(false);
      expect(service.journalBeforeUnload(state)).toEqual({ kind: 'nothing-to-journal' });
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
    // What the copy here builds on, as the media usage check compares it with the server's.
    expect(getProjectSyncSnapshot().projects[project.id]).toMatchObject({ revision: 1, workspaceRevision: 2 });
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
  const conflict = () => Object.assign(new Error('conflict'), { status: 409 });

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

  it('raises the canvas floor to 5 with the content that adds a noise adjustment', async () => {
    const { api, project, service } = await open(toSplitRecord(createDraftProject([])));
    const grained = applyCanvasProjectMutation(project, {
      anchor: stackTopAnchor(project.id),
      layer: {
        ...createEmptyPaintLayer('Grain', 'grain'),
        adjustments: [
          {
            amount: 0.1,
            distribution: 'gaussian',
            grain: 'monochrome',
            id: 'noise',
            isEnabled: false,
            response: 'flat',
            seed: 1,
            size: 1,
            type: 'noise',
          },
        ],
      },
      type: 'addCanvasLayer',
    });

    const saved = await service.saveWorkbench(stateWith([grained]));

    expect(saved.error).toBeNull();
    expect(vi.mocked(api.updateProject).mock.calls.map(([, request]) => request.minimum_canvas_schema_version)).toEqual(
      [5]
    );
    // A build that reads at most schema 4 is now refused the project by the server instead of loading it.
    expect(api.records.get(project.id)).toMatchObject({
      data: { canvas: { document: { version: 5 }, version: 5 } },
      minimum_canvas_schema_version: 5,
    });
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

  it('replays a draft from before the split onto a project split meanwhile, with no workspace yet', async () => {
    const project = createDraftProject([]);
    const edited = { ...project, settings: { ...project.settings, useCpuNoise: !project.settings.useCpuNoise } };
    // Its content landed as the split (revision 2); the page went before saving the workspace. Its draft is a record
    // from before the split: no workspace base at all.
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

  it('refuses to save a split project once a workflow repeats an id, sending nothing', async () => {
    const { api, project, service } = await open(toSplitRecord(createDraftProject([])));

    const saved = await service.saveWorkbench(stateWith([withRepeatedNode({ ...project, name: 'Repeats' })]));

    expect(saved.error).toMatch(/repeats an item id/);
    expect(saved.hasPendingChanges).toBe(true);
    expect(api.updateProject).not.toHaveBeenCalled();
    expect(api.saveProjectWorkspace).not.toHaveBeenCalled();
  });
});
