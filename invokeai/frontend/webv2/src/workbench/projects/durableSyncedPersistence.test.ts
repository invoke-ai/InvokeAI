import type { AccountScope } from '@platform/state/accountLifecycle';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { createAccountOwnedQueueRunJournal } from '@workbench/queue-integration/queueRunJournal';
import { createDraftProject, createInitialWorkbenchState } from '@workbench/workbenchState';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ProjectRecordDTO, ProjectSummaryDTO } from './api';
import type { ProjectDraftStore } from './draftStore';
import type { WorkbenchSessionBlob } from './session';

import { ProjectCreateAbsentError } from './api';
import { createMemoryProjectDraftStore, createUnavailableProjectDraftStore } from './draftStore';
import {
  createDurableSyncedWorkbenchPersistence,
  ProjectDraftWriteRejectedError,
  WorkbenchBackendUnavailableError,
  type DurableProjectPersistenceApi,
} from './durableSyncedPersistence';
import { createDeterministicProjectId } from './ids';
import { serializeProjectDocumentV3 } from './projectDocument';
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

const toRecord = (project: Project, revision = 1): ProjectRecordDTO => ({
  board_id: `board-${project.id}`,
  created_at: now,
  data: serializeProjectDocumentV3(project),
  minimum_canvas_schema_version: 3,
  name: project.name,
  project_id: project.id,
  revision,
  updated_at: now,
});

const toSummary = (record: ProjectRecordDTO): ProjectSummaryDTO => ({
  board_id: record.board_id,
  created_at: record.created_at,
  minimum_canvas_schema_version: record.minimum_canvas_schema_version,
  name: record.name,
  project_id: record.project_id,
  revision: record.revision,
  updated_at: record.updated_at,
});

const createApi = () => {
  const records = new Map<string, ProjectRecordDTO>();
  const api: DurableProjectPersistenceApi & { records: Map<string, ProjectRecordDTO> } = {
    records,
    createProject: vi.fn((request) => {
      const id = request.project_id!;
      const record: ProjectRecordDTO = {
        board_id: `board-${id}`,
        created_at: now,
        data: structuredClone(request.data),
        minimum_canvas_schema_version: request.minimum_canvas_schema_version ?? 3,
        name: request.name,
        project_id: id,
        revision: 1,
        updated_at: now,
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
    saveSession: vi.fn(() => Promise.resolve()),
    updateProject: vi.fn((projectId, request) => {
      const current = records.get(projectId);
      if (!current) {
        return Promise.reject(Object.assign(new Error('not found'), { status: 404 }));
      }
      if (current.revision !== request.expected_revision) {
        return Promise.reject(Object.assign(new Error('conflict'), { status: 409 }));
      }
      const record: ProjectRecordDTO = {
        ...current,
        data: structuredClone(request.data),
        minimum_canvas_schema_version: Math.max(
          current.minimum_canvas_schema_version,
          request.minimum_canvas_schema_version ?? current.minimum_canvas_schema_version
        ),
        name: request.name,
        revision: current.revision + 1,
        updated_at: now,
      };
      records.set(projectId, record);
      return Promise.resolve(structuredClone(record));
    }),
  };
  return api;
};

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
    let acknowledge!: (record: ProjectRecordDTO) => void;
    const created = new Promise<ProjectRecordDTO>((resolve) => {
      acknowledge = resolve;
    });
    vi.mocked(api.createProject).mockReturnValueOnce(created);
    const first = service.ensureProjectOnServer(project);
    const second = service.ensureProjectOnServer(project);
    const ready = vi.fn();
    void first.then(ready);
    await vi.waitFor(() => expect(api.createProject).toHaveBeenCalledOnce());
    expect(ready).not.toHaveBeenCalled();

    acknowledge(toRecord(project));
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
    let acknowledge!: (record: ProjectRecordDTO) => void;
    vi.mocked(api.createProject).mockReturnValueOnce(
      new Promise((resolve) => {
        acknowledge = resolve;
      })
    );
    const pending = service.ensureProjectOnServer(project);
    const rejected = expect(pending).rejects.toMatchObject({ reason: 'superseded' });
    await vi.waitFor(() => expect(api.createProject).toHaveBeenCalledOnce());
    service.markProjectDeleted(project.id);
    acknowledge(toRecord(project));
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
    expect(api.createProject).not.toHaveBeenCalled();
    await expect(draftStore.get(project.id, 'editor-1')).resolves.toMatchObject({
      draft: { conflict: { kind: 'revision', serverRevision: 2 }, state: 'conflict' },
      kind: 'found',
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
    futureRecord.data = { ...futureRecord.data, documentSchemaVersion: 4 };
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
        refusal: { documentSchemaVersion: 4, kind: 'document', maxDocumentSchemaVersion: 3 },
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

    expect(api.updateProject).toHaveBeenCalledOnce();
    expect(vi.mocked(api.updateProject).mock.calls[0]![1].data.documentSchemaVersion).toBe(3);
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
    let committedCopy: ProjectRecordDTO | null = null;
    const saveDraftAsNew = vi.fn().mockImplementation((input) => {
      if (!committedCopy) {
        committedCopy = {
          board_id: `board-${input.copyProjectId}`,
          created_at: now,
          data: structuredClone(input.document),
          minimum_canvas_schema_version: input.minimumCanvasSchemaVersion,
          name: input.name,
          project_id: input.copyProjectId,
          revision: 1,
          updated_at: now,
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
      const record: ProjectRecordDTO = {
        board_id: `board-${input.copyProjectId}`,
        created_at: now,
        data: structuredClone(input.document),
        minimum_canvas_schema_version: input.minimumCanvasSchemaVersion,
        name: input.name,
        project_id: input.copyProjectId,
        revision: 1,
        updated_at: now,
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
    expect(api.records.get(copy.targetProjectId)?.data.settings).toEqual(
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
    api.records.set(project.id, toRecord(remote, 2));
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
      const record: ProjectRecordDTO = {
        board_id: `board-${input.copyProjectId}`,
        created_at: now,
        data: structuredClone(input.document),
        minimum_canvas_schema_version: input.minimumCanvasSchemaVersion,
        name: input.name,
        project_id: input.copyProjectId,
        revision: 1,
        updated_at: now,
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
