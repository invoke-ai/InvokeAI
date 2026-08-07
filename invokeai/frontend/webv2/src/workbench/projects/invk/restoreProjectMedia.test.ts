import { describe, expect, it, vi } from 'vitest';

import type { InvkBoardItem } from './board';
import type { MaterializeResult, MediaMaterializer, RestoreProjectMediaDeps } from './restoreProjectMedia';
import type { InvkMediaRef } from './transfer';

import { createRestoredMediaLedger, restoreProjectMedia, rollbackRestoredMedia } from './restoreProjectMedia';

/**
 * The restore engine, with the byte source and the transport replaced.
 *
 * These are the rules that make a restored project *its own* project rather than a second view of
 * somebody else's: board media always gets a new identity, a board item that fails takes its
 * document references down with it rather than letting them resolve to a stranger, and everything
 * that could not be carried is reported against the role it failed in.
 */

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
  (items, boardId, onItemSettled) => {
    const result: MaterializeResult = { failed: [], materialized: [] };

    for (const item of items) {
      if (options.fail?.has(item.name)) {
        result.failed.push({ kind: item.kind, name: item.name, reason: 'upload-failed' });
      } else {
        result.materialized.push({ kind: item.kind, name: `${boardId}-${item.name}`, sourceName: item.name });
      }

      onItemSettled();
    }

    return Promise.resolve(result);
  };

const restore = (
  input: {
    boardItems?: readonly InvkBoardItem[];
    coverBytes?: { bytes: Uint8Array; entryName: string } | null;
    coverSourceImageName?: string | null;
    documentRefs?: readonly InvkMediaRef[];
    boardId?: string | null;
  },
  deps: Partial<RestoreProjectMediaDeps> = {}
) => {
  const ledger = createRestoredMediaLedger(input.boardId === undefined ? BOARD_ID : input.boardId);

  return restoreProjectMedia(
    {
      boardId: input.boardId === undefined ? BOARD_ID : input.boardId,
      boardItems: input.boardItems ?? [],
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
  /**
   * `board_images` has `PRIMARY KEY (image_name)`, so reusing a name would move the destination's
   * image onto this project's board rather than copy it — and deleting either board would then take
   * it from both. The existence check that document references get is deliberately not run here.
   */
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

  it('files images and videos in their own ledgers', async () => {
    const result = await restore({
      boardItems: [boardItem({ category: 'user' }), boardItem({ kind: 'video', name: 'clip.mp4' })],
    });

    expect(result.ledger).toMatchObject({
      boardImageNames: ['staging-board-board.png'],
      boardVideoNames: ['staging-board-clip.mp4'],
    });
    expect(result.boardItemIssues).toEqual([]);
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

  /**
   * A rejected star call reaches the caller's rollback, which deletes every asset the restore just
   * uploaded. Trading a whole project's media for a flag is the one thing this must not do.
   */
  it('does not lose the restore when the star request itself fails', async () => {
    const result = await restore(
      { boardItems: [boardItem({ starred: true })], documentRefs: [imageRef('board.png')] },
      { starImages: () => Promise.reject(new Error('500')) }
    );

    expect(result.boardItemIssues).toEqual([{ kind: 'image', name: 'board.png', reason: 'star-failed' }]);
    expect(result.mappings.images.get('board.png')).toBe('staging-board-board.png');
    expect(result.ledger.boardImageNames).toEqual(['staging-board-board.png']);
  });

  it('ends the restore when starring fails because the operation was cancelled', async () => {
    const cancelled = Object.assign(new Error('aborted'), { name: 'AbortError' });

    await expect(
      restore({ boardItems: [boardItem({ starred: true })] }, { starImages: () => Promise.reject(cancelled) })
    ).rejects.toBe(cancelled);
  });

  it('reports a descriptor the source could not materialize', async () => {
    const result = await restore(
      { boardItems: [boardItem({ name: 'gone.png' })] },
      {
        materializeBoardMedia: (items, _boardId, onItemSettled) => {
          onItemSettled();

          return Promise.resolve({
            failed: items.map((item) => ({ kind: item.kind, name: item.name, reason: 'missing-entry' as const })),
            materialized: [],
          });
        },
      }
    );

    expect(result.boardItemIssues).toEqual([{ kind: 'image', name: 'gone.png', reason: 'missing-entry' }]);
    expect(result.documentReferenceIssues).toEqual([]);
    expect(result.ledger.boardImageNames).toEqual([]);
  });

  /**
   * The invariant this module exists for. The old name is not neutral: on the same server it is
   * already taken, by the source project's own image.
   */
  it('forces an overlapping reference dangling when its board item fails', async () => {
    const result = await restore(
      { boardItems: [boardItem({ name: 'both.png' })], documentRefs: [imageRef('both.png')] },
      { materializeBoardMedia: freshNameMaterializer({ fail: new Set(['both.png']) }) }
    );

    expect(result.mappings.images.get('both.png')).toBe(`${PROJECT_ID}-missing-image-0`);
    expect(result.boardItemIssues).toEqual([{ kind: 'image', name: 'both.png', reason: 'upload-failed' }]);
    expect(result.documentReferenceIssues).toEqual([{ kind: 'image', name: 'both.png', reason: 'upload-failed' }]);
  });

  it('gives every occurrence of a failed reference the same placeholder', async () => {
    const items = [boardItem({ name: 'a.png' }), boardItem({ name: 'b.png' })];
    const result = await restore(
      { boardItems: items, documentRefs: [imageRef('a.png'), imageRef('b.png')] },
      { materializeBoardMedia: freshNameMaterializer({ fail: new Set(['a.png', 'b.png']) }) }
    );

    // Derived from the descriptor's position, not from the order the failures arrived in, so two
    // failed items can never collapse onto one placeholder.
    expect([...result.mappings.images]).toEqual([
      ['a.png', `${PROJECT_ID}-missing-image-0`],
      ['b.png', `${PROJECT_ID}-missing-image-1`],
    ]);
  });

  /**
   * The live reference set skips history — `collectLiveAssetRefs` walks past `recentImages`, canvas
   * snapshots and the queue — but `remapAssetRefs` rewrites the whole document. Gating the *mapping*
   * on that set therefore left the newest generated result, which is on the board and in the gallery
   * recents but not yet on canvas, pointing at the source project's image: the copy renders the
   * original's picture, and deleting the original breaks it with no explanation.
   */
  it('remaps a failed board item the document names only from history', async () => {
    const result = await restore(
      // No document ref: this name lives in `recentImages`, which the live-ref walker skips.
      { boardItems: [boardItem({ name: 'newest.png' })], documentRefs: [] },
      { materializeBoardMedia: freshNameMaterializer({ fail: new Set(['newest.png']) }) }
    );

    expect(result.mappings.images.get('newest.png')).toBe(`${PROJECT_ID}-missing-image-0`);
    expect(result.boardItemIssues).toEqual([{ kind: 'image', name: 'newest.png', reason: 'upload-failed' }]);
    // Still not *reported*: a gallery recent that stops resolving is not something the person lost
    // from this project, and counting it would inflate every report.
    expect(result.documentReferenceIssues).toEqual([]);
  });

  it('reports a failure the materializer raised twice only once', async () => {
    const result = await restore(
      { boardItems: [boardItem({ name: 'twice.png' })], documentRefs: [imageRef('twice.png')] },
      {
        materializeBoardMedia: (items, _boardId, onItemSettled) => {
          onItemSettled();

          const failed = items.map((item) => ({ kind: item.kind, name: item.name, reason: 'upload-failed' as const }));

          return Promise.resolve({ failed: [...failed, ...failed], materialized: [] });
        },
      }
    );

    expect(result.boardItemIssues).toEqual([{ kind: 'image', name: 'twice.png', reason: 'upload-failed' }]);
    expect(result.documentReferenceIssues).toEqual([{ kind: 'image', name: 'twice.png', reason: 'upload-failed' }]);
  });

  it('keeps two failures the board never described apart', async () => {
    // No descriptor position to derive from. Collapsing both onto index 0 would merge two unrelated
    // missing items into one dangling reference.
    const result = await restore(
      { boardItems: [boardItem({ name: 'described.png' })] },
      {
        materializeBoardMedia: (_items, _boardId, onItemSettled) => {
          onItemSettled();

          return Promise.resolve({
            failed: [
              { kind: 'image' as const, name: 'stray-a.png', reason: 'upload-failed' as const },
              { kind: 'image' as const, name: 'stray-b.png', reason: 'upload-failed' as const },
            ],
            materialized: [],
          });
        },
      }
    );

    expect(result.mappings.images.get('stray-a.png')).not.toBe(result.mappings.images.get('stray-b.png'));
  });

  /** A materializer that answers neither way has still not delivered the media. */
  it('treats an unaccounted descriptor as a failure rather than a silent success', async () => {
    const result = await restore(
      { boardItems: [boardItem({ name: 'ignored.png' })], documentRefs: [imageRef('ignored.png')] },
      { materializeBoardMedia: () => Promise.resolve({ failed: [], materialized: [] }) }
    );

    expect(result.boardItemIssues).toEqual([{ kind: 'image', name: 'ignored.png', reason: 'upload-failed' }]);
    expect(result.mappings.images.get('ignored.png')).toBe(`${PROJECT_ID}-missing-image-0`);
  });
});

describe('document-only references', () => {
  /**
   * The existence check is an optimization — it only decides whether an upload can be skipped. A
   * failed probe should cost bandwidth, never the import.
   */
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

  it('ends the restore when the existence check fails because the operation was cancelled', async () => {
    const cancelled = Object.assign(new Error('aborted'), { name: 'AbortError' });

    await expect(
      restore({ documentRefs: [imageRef('external.png')] }, { findExistingImageNames: () => Promise.reject(cancelled) })
    ).rejects.toBe(cancelled);
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

  it('uploads a missing reference and records the rename', async () => {
    const result = await restore(
      { documentRefs: [imageRef('external.png')] },
      { documentMediaBytes: () => new Uint8Array([1]) }
    );

    expect([...result.mappings.images]).toEqual([['external.png', 'uploaded-external.png']]);
    expect(result.ledger.imageNames).toEqual(['uploaded-external.png']);
  });

  it('reports a reference with no bytes and carries on', async () => {
    const result = await restore(
      { documentRefs: [imageRef('a.png'), imageRef('b.png')] },
      {
        documentMediaBytes: (ref) => (ref.name === 'b.png' ? new Uint8Array([1]) : undefined),
      }
    );

    expect(result.documentReferenceIssues).toEqual([{ kind: 'image', name: 'a.png', reason: 'missing-entry' }]);
    expect(result.ledger.imageNames).toEqual(['uploaded-b.png']);
  });

  it('reports a failed upload as dangling rather than failing the restore', async () => {
    const result = await restore(
      { documentRefs: [imageRef('external.png')] },
      { documentMediaBytes: () => new Uint8Array([1]), uploadImage: () => Promise.reject(new Error('rejected')) }
    );

    expect(result.documentReferenceIssues).toEqual([{ kind: 'image', name: 'external.png', reason: 'upload-failed' }]);
    expect(result.ledger.imageNames).toEqual([]);
  });

  /** An item that is both is board media; asking the destination about it would be the wrong question. */
  it('never checks the destination for a reference the board already owns', async () => {
    const findExisting = vi.fn(() => Promise.resolve(new Set<string>()));

    await restore(
      { boardItems: [boardItem({ name: 'both.png' })], documentRefs: [imageRef('both.png')] },
      { findExistingImageNames: findExisting }
    );

    expect(findExisting).not.toHaveBeenCalled();
  });

  it('leaves the video endpoints alone for an images-only restore', async () => {
    const findVideos = vi.fn(() => Promise.resolve(new Set<string>()));

    await restore({ documentRefs: [imageRef('a.png')] }, { findExistingVideoNames: findVideos });

    expect(findVideos).not.toHaveBeenCalled();
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

  it('falls back to the bundled thumbnail when its source could not be restored', async () => {
    const result = await restore(
      { boardItems: [boardItem()], coverBytes: cover, coverSourceImageName: 'board.png' },
      { materializeBoardMedia: freshNameMaterializer({ fail: new Set(['board.png']) }) }
    );

    expect(result.coverImageName).toBe('uploaded-cover.webp');
    expect(result.ledger.coverImageName).toBe('uploaded-cover.webp');
  });

  it('survives a fallback cover that will not upload', async () => {
    const result = await restore(
      { coverBytes: cover, coverSourceImageName: 'gone.png' },
      { uploadImage: () => Promise.reject(new Error('nope')) }
    );

    expect(result.coverImageName).toBeNull();
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

  it('drops a deduplicated reference out of the total rather than leaving it unfinished', async () => {
    const onProgress = vi.fn();

    await restore(
      { boardItems: [boardItem()], documentRefs: [imageRef('external.png')] },
      { findExistingImageNames: () => Promise.resolve(new Set(['external.png'])), onProgress }
    );

    expect(onProgress.mock.calls.at(-1)?.[0]).toEqual({ completed: 1, total: 1 });
  });
});

describe('rollbackRestoredMedia', () => {
  const ledger = () => ({
    boardId: BOARD_ID,
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
    // The board goes last, and without `include_images`, so anything that wandered onto it while
    // the import ran survives as Uncategorized.
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

  it('never rejects, whatever the cleanup does', async () => {
    await expect(
      rollbackRestoredMedia(ledger(), {
        deleteBoard: () => Promise.reject(new Error('board cleanup failed')),
        deleteImages: () => Promise.resolve(),
        deleteVideos: () => Promise.resolve(),
      })
    ).resolves.toBeUndefined();
  });

  it('issues no requests for a restore that created nothing', async () => {
    const deleteImages = vi.fn(() => Promise.resolve());
    const deleteVideos = vi.fn(() => Promise.resolve());
    const deleteBoard = vi.fn(() => Promise.resolve());

    await rollbackRestoredMedia(createRestoredMediaLedger(null), { deleteBoard, deleteImages, deleteVideos });

    expect(deleteImages).not.toHaveBeenCalled();
    expect(deleteVideos).not.toHaveBeenCalled();
    expect(deleteBoard).not.toHaveBeenCalled();
  });
});
