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

describe('duplicateProjectRecord', () => {
  it('copies every board item onto a staging board the create then claims', async () => {
    const result = await duplicateProject.duplicateProjectRecord({
      boardItems: [boardItem(), boardItem({ category: 'user', name: 'unreferenced.png' })],
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
      boardItems: [],
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
      boardItems: [boardItem()],
      owner,
      record: sourceRecord(),
    });

    expect(restoredLayerName()).toBe('copy-shared.png');
  });

  it('stars the copies whose descriptor was starred', async () => {
    await duplicateProject.duplicateProjectRecord({
      boardItems: [boardItem({ name: 'plain.png' }), boardItem({ name: 'starred.png', starred: true })],
      owner,
      record: sourceRecord(),
    });

    expect(transport.starImages).toHaveBeenCalledWith(['copy-starred.png'], owner.signal);
  });

  it('copies videos through the video endpoint with their category', async () => {
    await duplicateProject.duplicateProjectRecord({
      boardItems: [boardItem({ category: 'user', kind: 'video', name: 'clip.mp4' })],
      owner,
      record: sourceRecord(),
    });

    expect(transport.copyVideosToBoard).toHaveBeenCalledWith(['clip.mp4'], 'staging-board', owner.signal);
    expect(transport.copyImagesToBoard).toHaveBeenCalledWith([], 'staging-board', owner.signal);
  });

  /** Same-server duplicates share external references. */
  it('reuses a document reference the board does not own, copying nothing', async () => {
    const record = sourceRecord({ futureImageInput: { image_name: 'external.png' } });

    await duplicateProject.duplicateProjectRecord({ boardItems: [boardItem()], owner, record });

    expect(transport.copyImagesToBoard).toHaveBeenCalledWith(['shared.png'], 'staging-board', owner.signal);
    expect((createdData().futureImageInput as { image_name: string }).image_name).toBe('external.png');
  });

  /** A failed copy must never fall back to source-owned media. */
  it('forces a reference dangling when its board copy fails', async () => {
    transport.copyImagesToBoard.mockResolvedValue({ copied: [], failed: ['shared.png'] });

    const result = await duplicateProject.duplicateProjectRecord({
      boardItems: [boardItem()],
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

    const result = await duplicateProject.duplicateProjectRecord({ boardItems: [boardItem()], owner, record });
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

    await duplicateProject.duplicateProjectRecord({ boardItems: [], owner, record });

    const workflows = createdData().workflows as { entries: { source?: unknown }[] };

    expect(workflows.entries[0]?.source).toEqual(source);
    expect(createdData().documentSchemaVersion).toBe(3);
  });

  it('creates no staging board for a project whose board is empty', async () => {
    await duplicateProject.duplicateProjectRecord({ boardItems: [], owner, record: sourceRecord() });

    expect(transport.createStagingBoard).not.toHaveBeenCalled();
    expect(api.createProjectSettled.mock.calls[0]![0]).not.toHaveProperty('board_id');
  });

  it('deletes the copies it made and the staging board when the create provably did not happen', async () => {
    const failure = new ProjectCreateAbsentError(new Error('the server refused the create'));

    api.createProjectSettled.mockRejectedValue(failure);

    await expect(
      duplicateProject.duplicateProjectRecord({
        boardItems: [boardItem(), boardItem({ kind: 'video', name: 'clip.mp4' })],
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
      duplicateProject.duplicateProjectRecord({ boardItems: [boardItem()], owner, record: sourceRecord() })
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
      duplicateProject.duplicateProjectRecord({ boardItems: [boardItem()], owner, record: sourceRecord() })
    ).rejects.toThrow();

    expect(transport.deleteArchiveImages).not.toHaveBeenCalled();
    expect(transport.deleteStagingBoard).not.toHaveBeenCalled();
    account.accountLifecycle.invalidate();
  });
});
