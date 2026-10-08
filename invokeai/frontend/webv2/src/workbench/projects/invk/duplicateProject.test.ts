import type { AccountScope } from '@platform/state/accountLifecycle';
import type { ProjectBoardItemDTO, ProjectRecordDTO } from '@workbench/projects/api';
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

const sourceRecord = (document: Record<string, unknown> = {}): ProjectRecordDTO => {
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
    project_id: 'source',
    revision: 4,
    updated_at: '2026-06-10 10:00:00.000',
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
    const record = sourceRecord({ futureImageInput: { image_name: 'external.png' } });

    await duplicateProject.duplicateProjectRecord({ boards: inboxOf([boardItem()]), owner, record });

    expect(transport.copyImagesToBoard).toHaveBeenCalledWith(['shared.png'], 'staging-board', owner.signal);
    expect((createdData().futureImageInput as { image_name: string }).image_name).toBe('external.png');
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
    const record = sourceRecord({
      widgetStates: { gallery: { values: { projectBoardId: 'source-board', selectedBoardId: 'source-board' } } },
    });

    const result = await duplicateProject.duplicateProjectRecord({ boards: inboxOf([boardItem()]), owner, record });
    const gallery = (createdData().widgetStates as { gallery: { values: Record<string, unknown> } }).gallery.values;

    // The document that goes to the server carries neither the original's board nor its selection.
    expect(gallery.projectBoardId).toBeUndefined();
    expect(gallery.selectedBoardId).toBeUndefined();
    expect(createdData().id).toBe(result.record.project_id);
    expect(createdData().name).toBe('Source copy');

    // The returned record is patched with the board the server says it claimed, and points at it.
    const claimed = (result.record.data.widgetStates as { gallery: { values: Record<string, unknown> } }).gallery
      .values;

    expect(claimed).toMatchObject({ projectBoardId: 'staging-board', selectedBoardId: 'staging-board' });
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
    expect(createdData().documentSchemaVersion).toBe(3);
  });

  it('stages, copies and places every other board of the source, empty ones included', async () => {
    transport.createStagingBoard
      .mockResolvedValueOnce('staging-board')
      .mockResolvedValueOnce('member-staging')
      .mockResolvedValueOnce('empty-staging');

    const result = await duplicateProject.duplicateProjectRecord({
      boards: [
        ...inboxOf([boardItem()]),
        { archived: true, board_id: 'old', is_inbox: false, items: [boardItem({ name: 'old.png' })], name: 'Old' },
        { archived: false, board_id: 'empty', is_inbox: false, items: [], name: 'Empty' },
      ],
      owner,
      record: sourceRecord(),
    });

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

  it('stages and claims the inbox even when it is empty, copying nothing', async () => {
    await duplicateProject.duplicateProjectRecord({ boards: inboxOf([]), owner, record: sourceRecord() });

    expect(transport.createStagingBoard).toHaveBeenCalledTimes(1);
    expect(transport.copyImagesToBoard).not.toHaveBeenCalled();
    expect(api.createProjectSettled.mock.calls[0]![0]).toMatchObject({ board_id: 'staging-board' });
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
