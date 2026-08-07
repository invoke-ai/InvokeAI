import type { AccountScope } from '@platform/state/accountLifecycle';
import type { ProjectBoardItemDTO, ProjectRecordDTO } from '@workbench/projects/api';
import type * as apiModule from '@workbench/projects/api';

import { createDraftProject } from '@workbench/workbenchState';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as assetTransportModule from './assetTransport';
import type * as duplicateProjectModule from './duplicateProject';

/**
 * Duplication end to end, with the copy endpoints and the project API replaced.
 *
 * The rules it has to keep are import's rules, because it is import's engine: board media takes new
 * identities, the document follows them, a failed copy cannot leave a reference pointing at the
 * original's image, and a failure before the create undoes exactly what it made.
 */

const api = vi.hoisted(() => ({
  createProject: vi.fn(),
  getProject: vi.fn(),
  isProjectConfirmedAbsent: vi.fn(),
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
// Partial, so the module's constants and pure predicates stay real — a stub of
// `isRequestCancellation` would let the tests agree with a duplication that mistook a
// cancellation for a board's worth of individually failed copies.
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
        document: { ...project.canvas.document, layers: [rasterImageLayer('l1', 'shared.png')] },
      },
      id: 'source',
      name: 'Source',
      ...document,
    },
    name: 'Source',
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
  (api.createProject.mock.calls[0]![0] as { data: Record<string, unknown> }).data;

const restoredLayerName = (): string =>
  (
    createdData().canvas as {
      document: { layers: { source: { image: { imageName: string } } }[] };
    }
  ).document.layers[0]!.source.image.imageName;

beforeEach(async () => {
  vi.resetModules();
  vi.clearAllMocks();

  const account = await import('@platform/state/accountLifecycle');

  account.accountLifecycle.activate('duplicate-user');
  owner = account.captureAccountScope();

  transport.createStagingBoard.mockImplementation(() => Promise.resolve('staging-board'));
  transport.copyImagesToBoard.mockImplementation((names: readonly string[]) => Promise.resolve(copiesOf(names)));
  transport.copyVideosToBoard.mockImplementation((names: readonly string[]) => Promise.resolve(copiesOf(names)));
  transport.findExistingImageNames.mockImplementation((names: readonly string[]) => Promise.resolve(new Set(names)));
  transport.findExistingVideoNames.mockImplementation((names: readonly string[]) => Promise.resolve(new Set(names)));
  transport.starImages.mockImplementation(() => Promise.resolve({ failed: [] }));
  transport.starVideos.mockImplementation(() => Promise.resolve({ failed: [] }));
  api.createProject.mockImplementation(
    (request: { board_id?: string; data: Record<string, unknown>; name: string; project_id: string }) =>
      Promise.resolve({
        board_id: request.board_id ?? 'server-created-board',
        created_at: '2026-06-10 11:00:00.000',
        data: request.data,
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
    expect(api.createProject.mock.calls[0]![0]).toMatchObject({ board_id: 'staging-board', name: 'Source copy' });
    expect(result.record.project_id).not.toBe('source');
    expect(result.boardItemIssues).toEqual([]);
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

  /**
   * A reference to media outside the project's own board is a pointer, and both projects live on
   * this one server, so the copy simply keeps pointing at it.
   */
  it('reuses a document reference the board does not own, copying nothing', async () => {
    const record = sourceRecord({ futureImageInput: { image_name: 'external.png' } });

    await duplicateProject.duplicateProjectRecord({ boardItems: [boardItem()], owner, record });

    expect(transport.copyImagesToBoard).toHaveBeenCalledWith(['shared.png'], 'staging-board', owner.signal);
    expect((createdData().futureImageInput as { image_name: string }).image_name).toBe('external.png');
  });

  /**
   * The failure that must not bind to a stranger: on this server the original's `shared.png` is
   * right there, and falling back to it would leave the copy quietly sharing the original's image.
   */
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

  it('creates no staging board for a project whose board is empty', async () => {
    await duplicateProject.duplicateProjectRecord({ boardItems: [], owner, record: sourceRecord() });

    expect(transport.createStagingBoard).not.toHaveBeenCalled();
    expect(api.createProject.mock.calls[0]![0]).not.toHaveProperty('board_id');
  });

  it('deletes the copies it made and the staging board when the create fails', async () => {
    const failure = new Error('create rejected');

    api.createProject.mockRejectedValue(failure);
    api.isProjectConfirmedAbsent.mockResolvedValueOnce(true);

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

  it('leaves the copies alone when a rejected create may already have claimed the board', async () => {
    const failure = new Error('connection ended after create');

    api.createProject.mockRejectedValue(failure);
    api.isProjectConfirmedAbsent.mockResolvedValueOnce(false);

    await expect(
      duplicateProject.duplicateProjectRecord({ boardItems: [boardItem()], owner, record: sourceRecord() })
    ).rejects.toBe(failure);

    expect(transport.deleteArchiveImages).not.toHaveBeenCalled();
    expect(transport.deleteStagingBoard).not.toHaveBeenCalled();
  });

  it('does not issue destructive cleanup under a different account', async () => {
    const account = await import('@platform/state/accountLifecycle');

    api.createProject.mockImplementation(() => {
      account.accountLifecycle.activate('duplicate-user-b');

      return Promise.reject(new Error('create rejected'));
    });

    await expect(
      duplicateProject.duplicateProjectRecord({ boardItems: [boardItem()], owner, record: sourceRecord() })
    ).rejects.toThrow();

    expect(transport.deleteArchiveImages).not.toHaveBeenCalled();
    expect(transport.deleteStagingBoard).not.toHaveBeenCalled();
    account.accountLifecycle.invalidate();
  });
});
