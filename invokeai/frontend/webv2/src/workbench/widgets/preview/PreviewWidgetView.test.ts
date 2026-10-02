import type { GalleryItem } from '@features/gallery';

import { GALLERY_MAX_ROWS } from '@features/gallery/queries';
import { describe, expect, it } from 'vitest';

import { getVideoFrameCopyNotice } from './PreviewWidgetView';
import { mergePreviewBoardItems } from './usePreviewNavigation';

describe('mergePreviewBoardItems', () => {
  const item = (kind: GalleryItem['kind'], name: string, createdAt: string, starred = false): GalleryItem => {
    const base = {
      boardId: 'none',
      category: 'general' as const,
      createdAt,
      fullUrl: `/${kind}/${name}`,
      height: 64,
      isIntermediate: false,
      name,
      starred,
      thumbnailUrl: `/${kind}/${name}/thumbnail`,
      width: 64,
    };

    return kind === 'video' ? { ...base, durationSeconds: 1, kind } : { ...base, kind };
  };

  it('deduplicates and chronologically merges optimistic items in either direction', () => {
    const oldest = item('image', 'oldest', '2026-07-21T00:00:01.000Z');
    const middle = item('image', 'middle', '2026-07-21T00:00:02.000Z');
    const newest = item('image', 'newest', '2026-07-21T00:00:03.000Z');

    expect(mergePreviewBoardItems([newest, oldest], [middle, newest], 'DESC')).toEqual([newest, middle, oldest]);
    expect(mergePreviewBoardItems([oldest, newest], [middle, oldest], 'ASC')).toEqual([oldest, middle, newest]);
  });

  it('preserves relevance order for a ranked list', () => {
    // Preserve relevance order; ranked mode receives selection only, not local generations.
    const first = item('image', 'first', '2026-07-21T00:00:01.000Z');
    const second = item('image', 'second', '2026-07-21T00:00:03.000Z', true);
    const third = item('image', 'third', '2026-07-21T00:00:02.000Z');
    const optimistic = item('image', 'optimistic', '2026-07-21T00:00:09.000Z');

    expect(mergePreviewBoardItems([first, second, third], [], 'DESC', { isRanked: true })).toEqual([
      first,
      second,
      third,
    ]);
    // Without the flag the same input is re-sorted newest-first and the
    // optimistic item is spliced in, so the ranked path really is what
    // preserves the list.
    expect(mergePreviewBoardItems([first, second, third], [optimistic], 'DESC')).toEqual([
      optimistic,
      second,
      third,
      first,
    ]);
  });

  it('anchors a selection that the ranking does not contain, rather than losing the cursor', () => {
    // Keep out-of-ranking selection at the head so navigation can enter the ranked list.
    const ranked = item('image', 'ranked', '2026-07-21T00:00:01.000Z');
    const outsider = item('image', 'outsider', '2026-07-21T00:00:09.000Z');

    expect(mergePreviewBoardItems([ranked], [outsider], 'DESC', { isRanked: true })).toEqual([outsider, ranked]);
    // Already a member: kept once, in its ranked position.
    expect(mergePreviewBoardItems([ranked], [ranked], 'DESC', { isRanked: true })).toEqual([ranked]);
  });

  it('merges chronologically past starred items', () => {
    // The listing is flat: navigation must walk the same order the grid
    // shows, not lift starred items to the front.
    const starred = item('video', 'starred', '2026-07-21T00:00:01.000Z', true);
    const optimistic = item('image', 'optimistic', '2026-07-21T00:00:03.000Z');
    const existing = item('video', 'existing', '2026-07-21T00:00:02.000Z');

    expect(mergePreviewBoardItems([starred, existing], [optimistic], 'DESC')).toEqual([optimistic, existing, starred]);
  });

  it('uses the server kind/name tie-breakers for equal timestamps in both directions', () => {
    const createdAt = '2026-07-21T00:00:01.000Z';
    const imageA = item('image', 'a', createdAt);
    const imageZ = item('image', 'z', createdAt);
    const videoA = item('video', 'a', createdAt);
    const videoZ = item('video', 'z', createdAt);

    expect(mergePreviewBoardItems([videoA, imageZ], [videoZ, imageA], 'ASC')).toEqual([imageA, imageZ, videoA, videoZ]);
    expect(mergePreviewBoardItems([imageA, videoZ], [imageZ, videoA], 'DESC')).toEqual([
      videoZ,
      videoA,
      imageZ,
      imageA,
    ]);
  });

  it('keeps same-name media independent and bounds the merged Gallery window', () => {
    const backend = Array.from({ length: GALLERY_MAX_ROWS }, (_, index) =>
      item('image', `backend-${index}`, new Date(index * 1_000).toISOString())
    );
    const optimistic = Array.from({ length: 60 }, (_, index) =>
      item('video', `optimistic-${index}`, new Date((GALLERY_MAX_ROWS + index) * 1_000).toISOString())
    );
    backend[GALLERY_MAX_ROWS - 1] = item('image', 'shared', new Date((GALLERY_MAX_ROWS - 1) * 1_000).toISOString());
    optimistic[0] = item('video', 'shared', new Date((GALLERY_MAX_ROWS + 1) * 1_000).toISOString());

    const merged = mergePreviewBoardItems(backend, optimistic, 'DESC');

    expect(merged).toHaveLength(GALLERY_MAX_ROWS);
    expect(merged[0]?.name).toBe('optimistic-59');
    expect(merged).toContainEqual(expect.objectContaining({ kind: 'image', name: 'shared' }));
    expect(merged).toContainEqual(expect.objectContaining({ kind: 'video', name: 'shared' }));
  });
});

describe('getVideoFrameCopyNotice', () => {
  it.each([
    [{ ok: true } as const, 'success', 'widgets.preview.copyCurrentFrameSuccess'],
    [{ ok: false, reason: 'unsupported' } as const, 'error', 'widgets.preview.copyCurrentFrameUnsupported'],
    [{ ok: false, reason: 'not-ready' } as const, 'error', 'widgets.preview.copyCurrentFrameNotReady'],
    [{ ok: false, reason: 'draw-failed' } as const, 'error', 'widgets.preview.copyCurrentFrameDrawFailed'],
    [{ ok: false, reason: 'encode-failed' } as const, 'error', 'widgets.preview.copyCurrentFrameEncodeFailed'],
    [{ ok: false, reason: 'clipboard-failed' } as const, 'error', 'widgets.preview.copyCurrentFrameWriteFailed'],
    [{ ok: false, reason: 'stale' } as const, 'error', 'widgets.preview.copyCurrentFrameStale'],
  ])('maps %o to one localized notification', (result, kind, key) => {
    expect(getVideoFrameCopyNotice(result, (translationKey) => translationKey)).toEqual({ kind, title: key });
  });
});
