import { describe, expect, it, vi } from 'vitest';

import type { InvkBoardItem } from './board';
import type { MaterializeResult, MediaMaterializer, RestoreProjectMediaDeps } from './restoreProjectMedia';
import type { InvkMediaRef } from './transfer';

import { createRestoredMediaLedger, restoreProjectMedia, rollbackRestoredMedia } from './restoreProjectMedia';

/** Verify fresh ownership, failed-reference isolation, and role-aware loss reporting. */

const BOARD_ID = 'staging-board';
const PROJECT_ID = 'project-new';

const boardItem = (overrides: Partial<InvkBoardItem> = {}): InvkBoardItem => ({
  category: 'general',
  kind: 'image',
  name: 'board.png',
  starred: false,
  ...overrides,
});

const imageRef = (name: string): InvkMediaRef => ({ kind: 'image', name });

/** Materializes everything, under the fresh names the destination server would assign. */
const freshNameMaterializer =
  (options: { fail?: ReadonlySet<string> } = {}): MediaMaterializer =>
  (boards, onItemSettled) => {
    const result: MaterializeResult = { failed: [], materialized: [] };

    for (const { items, stagingBoardId } of boards) {
      for (const item of items) {
        if (options.fail?.has(item.name)) {
          result.failed.push({ kind: item.kind, name: item.name, reason: 'upload-failed' });
        } else {
          result.materialized.push({
            kind: item.kind,
            name: `${stagingBoardId}-${item.name}`,
            sourceName: item.name,
          });
        }

        onItemSettled();
      }
    }

    return Promise.resolve(result);
  };

const restore = (
  input: {
    boardItems?: readonly InvkBoardItem[];
    /** Several boards at once; `boardItems` describes the one-board case. */
    boards?: readonly { items: readonly InvkBoardItem[]; stagingBoardId: string }[];
    coverBytes?: { bytes: Uint8Array; entryName: string } | null;
    coverSourceImageName?: string | null;
    documentRefs?: readonly InvkMediaRef[];
  },
  deps: Partial<RestoreProjectMediaDeps> = {}
) => {
  const boards = input.boards ?? [{ items: input.boardItems ?? [], stagingBoardId: BOARD_ID }];
  const ledger = createRestoredMediaLedger(boards.map((board) => board.stagingBoardId));

  return restoreProjectMedia(
    {
      boards,
      coverBytes: input.coverBytes ?? null,
      coverSourceImageName: input.coverSourceImageName ?? null,
      documentRefs: input.documentRefs ?? [],
      ledger,
      projectId: PROJECT_ID,
    },
    {
      findExistingImageNames: () => Promise.resolve(new Set<string>()),
      findExistingVideoNames: () => Promise.resolve(new Set<string>()),
      materializeBoardMedia: freshNameMaterializer(),
      starImages: () => Promise.resolve({ failed: [] }),
      starVideos: () => Promise.resolve({ failed: [] }),
      uploadImage: (_bytes, fileName) => Promise.resolve({ height: 1, imageName: `uploaded-${fileName}`, width: 1 }),
      uploadVideo: (_bytes, fileName) => Promise.resolve({ videoName: `uploaded-${fileName}` }),
      ...deps,
    }
  ).then((result) => ({ ...result, ledger }));
};

describe('board media', () => {
  /** Board media needs fresh copies: adopting an existing image-name primary key would move another board's media. */
  it('takes a fresh identity even when the destination already has that name', async () => {
    const findExisting = vi.fn(() => Promise.resolve(new Set(['board.png'])));
    const result = await restore(
      { boardItems: [boardItem()], documentRefs: [imageRef('board.png')] },
      { findExistingImageNames: findExisting }
    );

    expect(findExisting).not.toHaveBeenCalled();
    expect(result.mappings.images.get('board.png')).toBe('staging-board-board.png');
    expect(result.ledger.boardImageNames).toEqual(['staging-board-board.png']);
  });

  it('stars exactly the descriptors that were starred, per kind', async () => {
    const starImages = vi.fn(() => Promise.resolve({ failed: [] }));
    const starVideos = vi.fn(() => Promise.resolve({ failed: [] }));

    await restore(
      {
        boardItems: [
          boardItem({ name: 'plain.png' }),
          boardItem({ name: 'starred.png', starred: true }),
          boardItem({ kind: 'video', name: 'starred.mp4', starred: true }),
        ],
      },
      { starImages, starVideos }
    );

    expect(starImages).toHaveBeenCalledWith(['staging-board-starred.png'], undefined);
    expect(starVideos).toHaveBeenCalledWith(['staging-board-starred.mp4'], undefined);
  });

  /** A star is a flag. Losing it is worth a warning, never worth losing the media or the mapping. */
  it('keeps the media and the mapping when starring fails', async () => {
    const result = await restore(
      { boardItems: [boardItem({ starred: true })], documentRefs: [imageRef('board.png')] },
      { starImages: () => Promise.resolve({ failed: ['staging-board-board.png'] }) }
    );

    expect(result.boardItemIssues).toEqual([{ kind: 'image', name: 'board.png', reason: 'star-failed' }]);
    expect(result.documentReferenceIssues).toEqual([]);
    expect(result.mappings.images.get('board.png')).toBe('staging-board-board.png');
  });

  /** After a failed copy, the existing source name remains unsafe to reuse. */
  it('forces an overlapping reference dangling when its board item fails', async () => {
    const result = await restore(
      { boardItems: [boardItem({ name: 'both.png' })], documentRefs: [imageRef('both.png')] },
      { materializeBoardMedia: freshNameMaterializer({ fail: new Set(['both.png']) }) }
    );

    expect(result.mappings.images.get('both.png')).toBe(`${PROJECT_ID}-missing-image-0`);
    expect(result.boardItemIssues).toEqual([{ kind: 'image', name: 'both.png', reason: 'upload-failed' }]);
    expect(result.documentReferenceIssues).toEqual([{ kind: 'image', name: 'both.png', reason: 'upload-failed' }]);
  });
});

describe('document-only references', () => {
  /** Failed existence probes may cause extra uploads, but must not fail the import. */
  it('uploads everything when the existence check fails', async () => {
    const uploadImage = vi.fn((_bytes: Uint8Array, fileName: string) =>
      Promise.resolve({ height: 1, imageName: `uploaded-${fileName}`, width: 1 })
    );
    const result = await restore(
      { documentRefs: [imageRef('external.png')] },
      {
        documentMediaBytes: () => new Uint8Array([1]),
        findExistingImageNames: () => Promise.reject(new Error('500')),
        uploadImage,
      }
    );

    expect(uploadImage).toHaveBeenCalledTimes(1);
    expect(result.mappings.images.get('external.png')).toBe('uploaded-external.png');
    expect(result.documentReferenceIssues).toEqual([]);
  });

  it('reuses media the destination already has instead of uploading it again', async () => {
    const uploadImage = vi.fn(() => Promise.resolve({ height: 1, imageName: 'never', width: 1 }));
    const result = await restore(
      { documentRefs: [imageRef('external.png')] },
      { findExistingImageNames: () => Promise.resolve(new Set(['external.png'])), uploadImage }
    );

    expect(uploadImage).not.toHaveBeenCalled();
    expect(result.mappings.images.size).toBe(0);
    expect(result.documentReferenceIssues).toEqual([]);
  });

  it('reports a failed upload as dangling rather than failing the restore', async () => {
    const result = await restore(
      { documentRefs: [imageRef('external.png')] },
      { documentMediaBytes: () => new Uint8Array([1]), uploadImage: () => Promise.reject(new Error('rejected')) }
    );

    expect(result.documentReferenceIssues).toEqual([{ kind: 'image', name: 'external.png', reason: 'upload-failed' }]);
    expect(result.ledger.imageNames).toEqual([]);
  });
});

describe('the cover', () => {
  const cover = { bytes: new Uint8Array([9]), entryName: 'cover.webp' };

  it('points at the restored board image rather than uploading the thumbnail again', async () => {
    const result = await restore({
      boardItems: [boardItem()],
      coverBytes: cover,
      coverSourceImageName: 'board.png',
    });

    expect(result.coverImageName).toBe('staging-board-board.png');
    expect(result.ledger.coverImageName).toBeNull();
  });
});

describe('progress', () => {
  it('counts each board item as it settles, then each upload', async () => {
    const onProgress = vi.fn();

    await restore(
      {
        boardItems: [boardItem({ name: 'one.png' }), boardItem({ name: 'two.png' })],
        documentRefs: [imageRef('external.png')],
      },
      { documentMediaBytes: () => new Uint8Array([1]), onProgress }
    );

    expect(onProgress.mock.calls.map(([progress]) => progress)).toEqual([
      { completed: 1, total: 3 },
      { completed: 2, total: 3 },
      { completed: 3, total: 3 },
    ]);
  });
});

describe('rollbackRestoredMedia', () => {
  const ledger = () => ({
    boardIds: [BOARD_ID],
    boardImageNames: ['board-image'],
    boardVideoNames: ['board-video'],
    coverImageName: 'cover-image',
    imageNames: ['document-image'],
    videoNames: ['document-video'],
  });

  it('deletes exactly what the restore created, then the board it created them on', async () => {
    const order: string[] = [];
    const deleteImages = vi.fn((names: string[]) => {
      order.push(`images:${names.join(',')}`);

      return Promise.resolve();
    });
    const deleteVideos = vi.fn((names: string[]) => {
      order.push(`videos:${names.join(',')}`);

      return Promise.resolve();
    });
    const deleteBoard = vi.fn((boardId: string) => {
      order.push(`board:${boardId}`);

      return Promise.resolve();
    });

    await rollbackRestoredMedia(ledger(), { deleteBoard, deleteImages, deleteVideos });

    expect(deleteImages).toHaveBeenCalledWith(['board-image', 'cover-image', 'document-image'], undefined);
    expect(deleteVideos).toHaveBeenCalledWith(['board-video', 'document-video'], undefined);
    // Delete the board last with include_images=false to preserve unrelated media.
    expect(order.at(-1)).toBe(`board:${BOARD_ID}`);
  });

  it('still drops the board when the media deletions fail', async () => {
    const deleteBoard = vi.fn(() => Promise.resolve());

    await rollbackRestoredMedia(ledger(), {
      deleteBoard,
      deleteImages: () => Promise.reject(new Error('image cleanup failed')),
      deleteVideos: () => Promise.reject(new Error('video cleanup failed')),
    });

    expect(deleteBoard).toHaveBeenCalledWith(BOARD_ID, undefined);
  });

  it('tries every staging board even when deleting one of them fails', async () => {
    const deleteBoard = vi
      .fn<(boardId: string) => Promise<void>>()
      .mockRejectedValueOnce(new Error('first board refused'))
      .mockResolvedValue(undefined);

    await rollbackRestoredMedia(
      { ...ledger(), boardIds: ['staged-inbox', 'staged-old'] },
      { deleteBoard, deleteImages: () => Promise.resolve(), deleteVideos: () => Promise.resolve() }
    );

    expect(deleteBoard.mock.calls.map(([boardId]) => boardId)).toEqual(['staged-inbox', 'staged-old']);
  });

  it('issues no requests for a restore that created nothing', async () => {
    const deleteImages = vi.fn(() => Promise.resolve());
    const deleteVideos = vi.fn(() => Promise.resolve());
    const deleteBoard = vi.fn(() => Promise.resolve());

    await rollbackRestoredMedia(createRestoredMediaLedger([]), { deleteBoard, deleteImages, deleteVideos });

    expect(deleteImages).not.toHaveBeenCalled();
    expect(deleteVideos).not.toHaveBeenCalled();
    expect(deleteBoard).not.toHaveBeenCalled();
  });
});
