import type { GalleryItem } from '@features/gallery';

import { GALLERY_MAX_ROWS } from '@features/gallery/queries';
import { describe, expect, it } from 'vitest';

import { getMatchingProgressImage, mergePreviewBoardItems } from './PreviewWidgetView';

describe('getMatchingProgressImage', () => {
  const placeholder = {
    backendItemId: null,
    boardId: 'none',
    height: 768,
    id: 'queue-1:1',
    itemIndex: 2,
    queueItemId: 'queue-1',
    width: 512,
  };
  const progressImage = {
    dataUrl: 'data:image/png;base64,abc',
    height: 768,
    target: { itemIndex: 2, queueItemId: 'queue-1' },
    width: 512,
  };

  it('returns progress only when it belongs to the current placeholder', () => {
    expect(getMatchingProgressImage(progressImage, placeholder)).toBe(progressImage);
    expect(
      getMatchingProgressImage({ ...progressImage, target: { itemIndex: 1, queueItemId: 'queue-1' } }, placeholder)
    ).toBeNull();
    expect(
      getMatchingProgressImage({ ...progressImage, target: { itemIndex: 2, queueItemId: 'queue-2' } }, placeholder)
    ).toBeNull();
  });
});

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

    expect(mergePreviewBoardItems([newest, oldest], [middle, newest], 'DESC', false)).toEqual([newest, middle, oldest]);
    expect(mergePreviewBoardItems([oldest, newest], [middle, oldest], 'ASC', false)).toEqual([oldest, middle, newest]);
  });

  it('keeps starred backend items ahead of optimistic unstarred items', () => {
    const starred = item('video', 'starred', '2026-07-21T00:00:01.000Z', true);
    const optimistic = item('image', 'optimistic', '2026-07-21T00:00:03.000Z');
    const existing = item('video', 'existing', '2026-07-21T00:00:02.000Z');

    expect(mergePreviewBoardItems([starred, existing], [optimistic], 'DESC', true)).toEqual([
      starred,
      optimistic,
      existing,
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

    const merged = mergePreviewBoardItems(backend, optimistic, 'DESC', false);

    expect(merged).toHaveLength(GALLERY_MAX_ROWS);
    expect(merged[0]?.name).toBe('optimistic-59');
    expect(merged).toContainEqual(expect.objectContaining({ kind: 'image', name: 'shared' }));
    expect(merged).toContainEqual(expect.objectContaining({ kind: 'video', name: 'shared' }));
  });
});
