import type { AccountScope } from '@platform/state/accountLifecycle';
import type { ProjectBoardItemDTO, ProjectWithWorkspaceDTO } from '@workbench/projects/api';
import type * as apiModule from '@workbench/projects/api';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { ProjectCreateAbsentError } from '@workbench/projects/api';
import { createDraftProject } from '@workbench/workbenchState';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as assetTransportModule from './assetTransport';
import type * as duplicateProjectModule from './duplicateProject';

/**
 * Verify fresh board identities, remapped references, isolated copy failures, and exact rollback through mocked
 * transport.
 */

// Mock settled creation because its verdict determines whether rollback is safe.
const api = vi.hoisted(() => ({
  createProjectSettled: vi.fn(),
  getProject: vi.fn(),
  invalidateBoardLists: vi.fn(),
  isProjectNotFoundError: (error: unknown) =>
    typeof error === 'object' && error !== null && 'status' in error && error.status === 404,
}));

const transport = vi.hoisted(() => ({
  copyImagesToBoard: vi.fn(),
  copyVideosToBoard: vi.fn(),
  createStagingBoard: vi.fn(() => Promise.resolve('staging-board')),
  deleteArchiveImages: vi.fn(() => Promise.resolve()),
  deleteArchiveVideos: vi.fn(() => Promise.resolve()),
  deleteStagingBoard: vi.fn(() => Promise.resolve()),
  findExistingImageNames: vi.fn((names: readonly string[]) => Promise.resolve(new Set(names))),
  findExistingVideoNames: vi.fn((names: readonly string[]) => Promise.resolve(new Set(names))),
  getBoardProjectId: vi.fn((): Promise<string | null> => Promise.resolve(null)),
  mimeForEntryName: () => 'image/png',
  placeBoardInProject: vi.fn(() => Promise.resolve()),
  starImages: vi.fn(() => Promise.resolve({ failed: [] as string[] })),
  starVideos: vi.fn(() => Promise.resolve({ failed: [] as string[] })),
  uploadArchiveImage: vi.fn(() => Promise.reject(new Error('duplication uploads nothing'))),
  uploadArchiveVideo: vi.fn(() => Promise.reject(new Error('duplication uploads nothing'))),
}));

vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof apiModule>()),
  ...api,
}));
// Preserve real cancellation predicates in partial mocks.
vi.mock('./assetTransport', async (importOriginal) => ({
  ...(await importOriginal<typeof assetTransportModule>()),
  ...transport,
}));

let duplicateProject: typeof duplicateProjectModule;
let owner: AccountScope;

const rasterImageLayer = (id: string, imageName: string) => ({
  blendMode: 'normal' as const,
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height: 1, imageName, width: 1 }, type: 'image' as const },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster' as const,
});

const sourceRecord = (document: Record<string, unknown> = {}): ProjectWithWorkspaceDTO => {
  const project = createDraftProject([]);

  return {
    board_id: 'source-board',
    created_at: '2026-06-10 10:00:00.000',
    data: {
      ...project,
      canvas: {
        ...project.canvas,
        document: { ...project.canvas.document, stacks: stacksFrom([rasterImageLayer('l1', 'shared.png')]) },
      },
      id: 'source',
      name: 'Source',
      ...document,
    },
    name: 'Source',
    minimum_canvas_schema_version: 2,
    minimum_document_schema_version: 3,
    project_id: 'source',
    revision: 4,
    updated_at: '2026-06-10 10:00:00.000',
    workspace_updated_at: null,
    workspace: null,
  };
};

const boardItem = (overrides: Partial<ProjectBoardItemDTO> = {}): ProjectBoardItemDTO => ({
  category: 'general',
  kind: 'image',
  name: 'shared.png',
  starred: false,
  ...overrides,
});

const copiesOf = (names: readonly string[]) => ({
  copied: names.map((name) => ({ name: `copy-${name}`, sourceName: name })),
  failed: [] as string[],
});

const createdData = (): Record<string, unknown> =>
  (api.createProjectSettled.mock.calls[0]![0] as { data: Record<string, unknown> }).data;

const restoredLayerName = (): string =>
  (
    createdData().canvas as {
      document: { stacks: { raster: { source: { image: { imageName: string } } }[] } };
    }
  ).document.stacks.raster[0]!.source.image.imageName;

beforeEach(async () => {
  vi.resetModules();
  vi.resetAllMocks();

  const account = await import('@platform/state/accountLifecycle');

  account.accountLifecycle.activate('duplicate-user');
  owner = account.captureAccountScope();

  transport.copyImagesToBoard.mockImplementation((names: readonly string[]) => Promise.resolve(copiesOf(names)));
  transport.copyVideosToBoard.mockImplementation((names: readonly string[]) => Promise.resolve(copiesOf(names)));
  api.createProjectSettled.mockImplementation(
    (request: {
      board_id?: string;
      data: Record<string, unknown>;
      minimum_canvas_schema_version?: number;
      name: string;
      project_id: string;
      workspace?: Record<string, unknown>;
    }) =>
      Promise.resolve({
        board_id: request.board_id ?? 'server-created-board',
        created_at: '2026-06-10 11:00:00.000',
        data: request.data,
        minimum_canvas_schema_version: request.minimum_canvas_schema_version ?? 2,
        name: request.name,
        project_id: request.project_id,
        revision: 1,
        updated_at: '2026-06-10 11:00:00.000',
        workspace: request.workspace
          ? { data: request.workspace, revision: 1, updated_at: '2026-06-10 11:00:00.000' }
          : null,
      })
  );

  duplicateProject = await import('./duplicateProject');
});

/** The source's boards as the snapshot lists them: the inbox alone unless a test adds members. */
const inboxOf = (items: ReturnType<typeof boardItem>[]) => [
  { archived: false, board_id: 'source-board', is_inbox: true, items, name: 'Source' },
];

describe('duplicateProjectRecord', () => {
  it('copies every board item onto a staging board the create then claims', async () => {
    const result = await duplicateProject.duplicateProjectRecord({
      boards: inboxOf([boardItem(), boardItem({ category: 'user', name: 'unreferenced.png' })]),
      owner,
      record: sourceRecord(),
    });

    expect(transport.createStagingBoard).toHaveBeenCalledWith('Source copy', owner.signal);
    expect(transport.copyImagesToBoard).toHaveBeenCalledWith(
      ['shared.png', 'unreferenced.png'],
      'staging-board',
      owner.signal
    );
    expect(api.createProjectSettled.mock.calls[0]![0]).toMatchObject({
      board_id: 'staging-board',
      minimum_canvas_schema_version: 2,
      name: 'Source copy',
    });
    expect(result.record.project_id).not.toBe('source');
    expect(result.boardItemIssues).toEqual([]);
  });

  it('uses a caller-reserved identity for retry-safe conflict copies', async () => {
    await duplicateProject.duplicateProjectRecord({
      boards: inboxOf([]),
      identity: { id: 'reserved-copy', name: 'Source (copy)' },
      owner,
      record: sourceRecord(),
    });

    expect(api.createProjectSettled.mock.calls[0]![0]).toMatchObject({
      name: 'Source (copy)',
      project_id: 'reserved-copy',
    });
  });

  /** The whole reason duplication needs its own copies rather than a second reference. */
  it('points the copy at its own media, never the original name', async () => {
    await duplicateProject.duplicateProjectRecord({
      boards: inboxOf([boardItem()]),
      owner,
      record: sourceRecord(),
    });

    expect(restoredLayerName()).toBe('copy-shared.png');
  });

  it('stars the copies whose descriptor was starred', async () => {
    await duplicateProject.duplicateProjectRecord({
      boards: inboxOf([boardItem({ name: 'plain.png' }), boardItem({ name: 'starred.png', starred: true })]),
      owner,
      record: sourceRecord(),
    });

    expect(transport.starImages).toHaveBeenCalledWith(['copy-starred.png'], owner.signal);
  });

  it('copies videos through the video endpoint with their category', async () => {
    await duplicateProject.duplicateProjectRecord({
      boards: inboxOf([boardItem({ category: 'user', kind: 'video', name: 'clip.mp4' })]),
      owner,
      record: sourceRecord(),
    });

    expect(transport.copyVideosToBoard).toHaveBeenCalledWith(['clip.mp4'], 'staging-board', owner.signal);
    expect(transport.copyImagesToBoard).toHaveBeenCalledWith([], 'staging-board', owner.signal);
  });

  /** Same-server duplicates share external references. */
  it('reuses a document reference the board does not own, copying nothing', async () => {
    const project = createDraftProject([]);
    const record = sourceRecord({
      canvas: {
        ...project.canvas,
        document: {
          ...project.canvas.document,
          stacks: stacksFrom([rasterImageLayer('l1', 'shared.png'), rasterImageLayer('l2', 'external.png')]),
        },
      },
    });

    await duplicateProject.duplicateProjectRecord({ boards: inboxOf([boardItem()]), owner, record });

    expect(transport.copyImagesToBoard).toHaveBeenCalledWith(['shared.png'], 'staging-board', owner.signal);
    expect(JSON.stringify(createdData().canvas)).toContain('"external.png"');
  });

  /** A failed copy must never fall back to source-owned media. */
  it('forces a reference dangling when its board copy fails', async () => {
    transport.copyImagesToBoard.mockResolvedValue({ copied: [], failed: ['shared.png'] });

    const result = await duplicateProject.duplicateProjectRecord({
      boards: inboxOf([boardItem()]),
      owner,
      record: sourceRecord(),
    });

    expect(restoredLayerName()).not.toBe('shared.png');
    expect(restoredLayerName()).toContain('-missing-image-');
    expect(result.boardItemIssues).toEqual([{ kind: 'image', name: 'shared.png', reason: 'upload-failed' }]);
    expect(result.documentReferenceIssues).toEqual([{ kind: 'image', name: 'shared.png', reason: 'upload-failed' }]);
  });

  it('canonicalizes the copy under its own identity and drops installation state', async () => {
    const project = createDraftProject([]);
    const gallery = Object.values(project.widgetInstances).find((instance) => instance.typeId === 'gallery')!;
    const record = sourceRecord({
      widgetInstances: {
        ...project.widgetInstances,
        [gallery.id]: {
          ...gallery,
          state: {
            ...gallery.state,
            values: { ...gallery.state.values, projectBoardId: 'source-board', selectedBoardId: 'source-board' },
          },
        },
      },
    });

    const result = await duplicateProject.duplicateProjectRecord({ boards: inboxOf([boardItem()]), owner, record });

    // The content carries no gallery state at all, and is the copy's under its own identity.
    expect(createdData().widgetInstances).toBeUndefined();
    expect(createdData().id).toBe(result.record.project_id);
    expect(createdData().name).toBe('Source copy');
    // The duplicating account's workspace goes in the same create, opening on the staged inbox it claims, never the
    // original's board; the record answers with the workspace the create stored.
    expect(api.createProjectSettled).toHaveBeenCalledOnce();
    const sent = (api.createProjectSettled.mock.calls[0]![0] as { workspace: Record<string, unknown> }).workspace;
    expect((sent.widgetInstances as typeof project.widgetInstances)[gallery.id]!.state.values).toMatchObject({
      projectBoardId: 'staging-board',
      selectedBoardId: 'staging-board',
    });
    expect(api.createProjectSettled.mock.calls[0]![0]).toMatchObject({
      board_id: 'staging-board',
      minimum_document_schema_version: 4,
    });
    expect(result.record.workspace).toMatchObject({ data: sent, revision: 1 });
  });

  it('stages an inbox for a source whose snapshot lists no boards, so the create still carries both halves', async () => {
    const result = await duplicateProject.duplicateProjectRecord({ boards: [], owner, record: sourceRecord() });

    expect(transport.createStagingBoard).toHaveBeenCalledExactlyOnceWith('Source copy', owner.signal);
    expect(api.createProjectSettled.mock.calls[0]![0]).toMatchObject({
      board_id: 'staging-board',
      workspace: expect.any(Object),
    });
    expect(result.record.workspace).not.toBeNull();
  });

  it('copies a project whose workflow repeats an item id whole, as no load would accept it split', async () => {
    const project = createDraftProject([]);
    const node = { data: { label: '', notes: '' }, id: 'node-1', position: { x: 0, y: 0 }, type: 'notes' };
    const record = sourceRecord({
      documentSchemaVersion: 3,
      workflows: {
        ...project.workflows,
        entries: project.workflows.entries.map((entry) => ({
          ...entry,
          document: { ...entry.document, nodes: [node, node] },
        })),
      },
    });

    const result = await duplicateProject.duplicateProjectRecord({ boards: inboxOf([]), owner, record });

    expect(api.createProjectSettled.mock.calls[0]![0]).toMatchObject({ minimum_document_schema_version: 3 });
    expect(api.createProjectSettled.mock.calls[0]![0]).not.toHaveProperty('workspace');
    expect(createdData()).toMatchObject({ documentSchemaVersion: 3, widgetInstances: expect.any(Object) });
    expect(result.record.workspace).toBeNull();
  });

  it('keeps each workflow linked to its library template, since the copy lives on the same server', async () => {
    const source = { libraryWorkflowId: 'lib-1', revision: 4 };
    const project = createDraftProject([]);
    const record = sourceRecord({
      documentSchemaVersion: 3,
      workflows: {
        ...project.workflows,
        entries: project.workflows.entries.map((entry) => ({ ...entry, source })),
      },
    });

    await duplicateProject.duplicateProjectRecord({ boards: inboxOf([]), owner, record });

    const workflows = createdData().workflows as { entries: { source?: unknown }[] };

    expect(workflows.entries[0]?.source).toEqual(source);
    expect(createdData().documentSchemaVersion).toBe(4);
  });

  it('stages, copies and places every other board of the source, empty ones included', async () => {
    transport.createStagingBoard
      .mockResolvedValueOnce('staging-board')
      .mockResolvedValueOnce('member-staging')
      .mockResolvedValueOnce('empty-staging');

    const onProgress = vi.fn();
    const result = await duplicateProject.duplicateProjectRecord(
      {
        boards: [
          ...inboxOf([boardItem()]),
          { archived: true, board_id: 'old', is_inbox: false, items: [boardItem({ name: 'old.png' })], name: 'Old' },
          { archived: false, board_id: 'empty', is_inbox: false, items: [], name: 'Empty' },
        ],
        owner,
        record: sourceRecord(),
      },
      { onProgress }
    );

    // In source order, so the copy's boards keep their creation order; the empty one has nothing to copy.
    expect((transport.createStagingBoard.mock.calls as unknown[][]).map((call) => call[0])).toEqual([
      'Source copy',
      'Old',
      'Empty',
    ]);
    expect(transport.copyImagesToBoard).toHaveBeenCalledWith(['old.png'], 'member-staging', owner.signal);
    expect(transport.copyImagesToBoard).toHaveBeenCalledTimes(2);
    expect(api.createProjectSettled.mock.calls[0]![0]).toMatchObject({ board_id: 'staging-board' });
    // Only once the copy exists: the staged members move in, archived as they were.
    expect(transport.placeBoardInProject.mock.calls).toEqual([
      ['member-staging', result.record.project_id, true, owner.signal],
      ['empty-staging', result.record.project_id, false, owner.signal],
    ]);
    // Copying is counted by item, then the moves by board, so progress does not sit at its end while boards move.
    expect(onProgress.mock.calls.map(([progress]) => progress).slice(-3)).toEqual([
      { completed: 2, phase: 'restoring', total: 2 },
      { completed: 1, phase: 'placing-boards', total: 2 },
      { completed: 2, phase: 'placing-boards', total: 2 },
    ]);
    // Board lists refreshed before the moves finished would still show the members in the Library.
    expect(api.invalidateBoardLists).toHaveBeenCalledExactlyOnceWith(owner);
    expect(api.invalidateBoardLists.mock.invocationCallOrder[0]).toBeGreaterThan(
      transport.placeBoardInProject.mock.invocationCallOrder[1]!
    );
    expect(result.boardIssues).toEqual([]);
  });

  it('reports a board that could not be placed and keeps the copy', async () => {
    transport.placeBoardInProject.mockRejectedValueOnce(new Error('move refused'));
    transport.createStagingBoard.mockResolvedValueOnce('staging-board').mockResolvedValueOnce('member-staging');

    const result = await duplicateProject.duplicateProjectRecord({
      boards: [
        ...inboxOf([]),
        { archived: false, board_id: 'old', is_inbox: false, items: [boardItem({ name: 'old.png' })], name: 'Old' },
      ],
      owner,
      record: sourceRecord(),
    });

    expect(result.record.project_id).not.toBe('source');
    expect(result.boardIssues).toEqual([{ name: 'Old' }]);
    expect(transport.deleteStagingBoard).not.toHaveBeenCalled();
  });

  it('refreshes board lists even when placing the boards is cancelled part way', async () => {
    transport.placeBoardInProject.mockRejectedValueOnce(new DOMException('cancelled', 'AbortError'));
    transport.createStagingBoard.mockResolvedValueOnce('staging-board').mockResolvedValueOnce('member-staging');

    await expect(
      duplicateProject.duplicateProjectRecord({
        boards: [...inboxOf([]), { archived: false, board_id: 'old', is_inbox: false, items: [], name: 'Old' }],
        owner,
        record: sourceRecord(),
      })
    ).rejects.toThrow('cancelled');

    expect(api.invalidateBoardLists).toHaveBeenCalledExactlyOnceWith(owner);
  });

  it('stages and claims the inbox even when it is empty, copying nothing', async () => {
    await duplicateProject.duplicateProjectRecord({ boards: inboxOf([]), owner, record: sourceRecord() });

    expect(transport.createStagingBoard).toHaveBeenCalledTimes(1);
    expect(transport.copyImagesToBoard).not.toHaveBeenCalled();
    expect(api.createProjectSettled.mock.calls[0]![0]).toMatchObject({ board_id: 'staging-board' });
  });

  it('refuses a source with more boards than a copy may carry, before staging anything', async () => {
    const members = Array.from({ length: 1000 }, (_unused, index) => ({
      archived: false,
      board_id: `b${String(index)}`,
      is_inbox: false,
      items: [],
      name: `Board ${String(index)}`,
    }));

    await expect(
      duplicateProject.duplicateProjectRecord({ boards: [...inboxOf([]), ...members], owner, record: sourceRecord() })
    ).rejects.toMatchObject({ reason: 'too-large' });

    expect(transport.createStagingBoard).not.toHaveBeenCalled();
    expect(api.createProjectSettled).not.toHaveBeenCalled();
  });

  it('deletes the boards it had staged when staging a later one fails', async () => {
    transport.createStagingBoard.mockResolvedValueOnce('staging-board').mockRejectedValueOnce(new Error('refused'));

    await expect(
      duplicateProject.duplicateProjectRecord({
        boards: [
          ...inboxOf([boardItem()]),
          { archived: false, board_id: 'old', is_inbox: false, items: [boardItem({ name: 'old.png' })], name: 'Old' },
        ],
        owner,
        record: sourceRecord(),
      })
    ).rejects.toThrow('refused');

    expect(transport.copyImagesToBoard).not.toHaveBeenCalled();
    expect(transport.deleteStagingBoard).toHaveBeenCalledExactlyOnceWith('staging-board', owner.signal);
    expect(api.createProjectSettled).not.toHaveBeenCalled();
  });

  it('deletes the copies it made and the staging board when the create provably did not happen', async () => {
    const failure = new ProjectCreateAbsentError(new Error('the server refused the create'));

    api.createProjectSettled.mockRejectedValue(failure);

    await expect(
      duplicateProject.duplicateProjectRecord({
        boards: inboxOf([boardItem(), boardItem({ kind: 'video', name: 'clip.mp4' })]),
        owner,
        record: sourceRecord(),
      })
    ).rejects.toBe(failure);

    expect(transport.deleteArchiveImages).toHaveBeenCalledWith(['copy-shared.png'], owner.signal);
    expect(transport.deleteArchiveVideos).toHaveBeenCalledWith(['copy-clip.mp4'], owner.signal);
    expect(transport.deleteStagingBoard).toHaveBeenCalledWith('staging-board', owner.signal);
    // Uploads announce their board, so lists can already show the staging board it deleted.
    expect(api.invalidateBoardLists).toHaveBeenCalledExactlyOnceWith(owner);
    expect(api.invalidateBoardLists.mock.invocationCallOrder[0]).toBeGreaterThan(
      transport.deleteStagingBoard.mock.invocationCallOrder[0]!
    );
  });

  /** Retain media for unproven create outcomes; GET 404 can race commit. */
  it('leaves the copies alone when a rejected create may already have landed', async () => {
    const failure = new Error('connection ended after create');

    api.createProjectSettled.mockRejectedValue(failure);

    await expect(
      duplicateProject.duplicateProjectRecord({ boards: inboxOf([boardItem()]), owner, record: sourceRecord() })
    ).rejects.toBe(failure);

    expect(transport.deleteArchiveImages).not.toHaveBeenCalled();
    expect(transport.deleteStagingBoard).not.toHaveBeenCalled();
  });

  it('does not issue destructive cleanup under a different account', async () => {
    const account = await import('@platform/state/accountLifecycle');

    api.createProjectSettled.mockImplementation(() => {
      account.accountLifecycle.activate('duplicate-user-b');

      return Promise.reject(new ProjectCreateAbsentError(new Error('create rejected')));
    });

    await expect(
      duplicateProject.duplicateProjectRecord({ boards: inboxOf([boardItem()]), owner, record: sourceRecord() })
    ).rejects.toThrow();

    expect(transport.deleteArchiveImages).not.toHaveBeenCalled();
    expect(transport.deleteStagingBoard).not.toHaveBeenCalled();
    account.accountLifecycle.invalidate();
  });
});
