import type { GalleryItemKind, GalleryItemKey, GalleryQueuePlaceholder } from '@features/gallery/contracts';

import { describe, expect, it } from 'vitest';

import {
  getPreviewNavigationCursor,
  getPreviewNavigationSequence,
  getPreviewNavigationTarget,
  type PreviewNavigationItem,
} from './previewNavigation';

interface TestItem {
  kind: GalleryItemKind;
  name: string;
  starred?: boolean;
}

const placeholder: GalleryQueuePlaceholder = {
  backendItemId: null,
  boardId: 'none',
  height: 1024,
  id: 'queue-1:1',
  itemIndex: 1,
  queueItemId: 'queue-1',
  width: 1024,
};

const item = (kind: GalleryItemKind, name: string, starred = false): TestItem => ({ kind, name, starred });

const keys = (sequence: PreviewNavigationItem<TestItem>[]): string[] =>
  sequence.map((entry) => (entry.kind === 'item' ? `${entry.item.kind}:${entry.item.name}` : 'placeholder'));

const buildSequence = (overrides: Partial<Parameters<typeof getPreviewNavigationSequence<TestItem>>[0]> = {}) =>
  getPreviewNavigationSequence<TestItem>({
    activePlaceholder: placeholder,
    boardId: 'none',
    boardImages: [item('image', 'newest'), item('video', 'middle'), item('image', 'oldest')],
    galleryView: 'images',
    imageOrderDir: 'DESC',
    ...overrides,
  });

describe('getPreviewNavigationSequence', () => {
  it('places the placeholder at the newest position in descending order', () => {
    expect(keys(buildSequence())).toEqual(['placeholder', 'image:newest', 'video:middle', 'image:oldest']);
  });

  it('places the placeholder at the latest chronological position in ascending order', () => {
    expect(keys(buildSequence({ imageOrderDir: 'ASC' }))).toEqual([
      'image:newest',
      'video:middle',
      'image:oldest',
      'placeholder',
    ]);
  });

  it('places the placeholder after the leading starred block when starred-first sorts descending', () => {
    expect(
      keys(
        buildSequence({
          boardImages: [
            item('video', 'starred-a', true),
            item('image', 'starred-b', true),
            item('image', 'newest'),
            item('video', 'oldest'),
          ],
        })
      )
    ).toEqual(['video:starred-a', 'image:starred-b', 'placeholder', 'image:newest', 'video:oldest']);
  });

  it('places the placeholder last when every item is starred', () => {
    expect(
      keys(
        buildSequence({
          boardImages: [item('image', 'starred-a', true), item('video', 'starred-b', true)],
        })
      )
    ).toEqual(['image:starred-a', 'video:starred-b', 'placeholder']);
  });

  it('keeps the placeholder last in ascending order even with starred-first', () => {
    expect(
      keys(
        buildSequence({
          boardImages: [item('video', 'starred-a', true), item('image', 'newest')],
          imageOrderDir: 'ASC',
        })
      )
    ).toEqual(['video:starred-a', 'image:newest', 'placeholder']);
  });

  it('excludes the placeholder when it belongs to another board or the assets view', () => {
    expect(keys(buildSequence({ boardId: 'board-2' }))).toEqual(['image:newest', 'video:middle', 'image:oldest']);
    expect(keys(buildSequence({ galleryView: 'assets' }))).toEqual(['image:newest', 'video:middle', 'image:oldest']);
  });

  it('returns only completed media items when there is no active placeholder', () => {
    expect(keys(buildSequence({ activePlaceholder: null }))).toEqual(['image:newest', 'video:middle', 'image:oldest']);
  });

  it('keeps same-name images and videos as independent navigation entries', () => {
    expect(
      keys(
        buildSequence({
          activePlaceholder: null,
          boardImages: [item('image', 'shared'), item('video', 'shared')],
        })
      )
    ).toEqual(['image:shared', 'video:shared']);
  });
});

describe('getPreviewNavigationCursor', () => {
  const sequence = buildSequence();

  it('resolves to the placeholder while following live', () => {
    expect(getPreviewNavigationCursor(sequence, { isFollowingLive: true, selectedItemKey: 'video:middle' })).toBe(0);
  });

  it('resolves the selected media kind and name otherwise', () => {
    expect(getPreviewNavigationCursor(sequence, { isFollowingLive: false, selectedItemKey: 'video:middle' })).toBe(2);

    const sameName = buildSequence({
      activePlaceholder: null,
      boardImages: [item('image', 'shared'), item('video', 'shared')],
    });
    expect(getPreviewNavigationCursor(sameName, { isFollowingLive: false, selectedItemKey: 'image:shared' })).toBe(0);
    expect(getPreviewNavigationCursor(sameName, { isFollowingLive: false, selectedItemKey: 'video:shared' })).toBe(1);
  });

  it('is unresolved when the selected item is absent from the sequence', () => {
    expect(
      getPreviewNavigationCursor(sequence, {
        isFollowingLive: false,
        selectedItemKey: 'image:missing' as GalleryItemKey,
      })
    ).toBe(-1);
    expect(getPreviewNavigationCursor(sequence, { isFollowingLive: false, selectedItemKey: null })).toBe(-1);
  });
});

describe('getPreviewNavigationTarget', () => {
  const sequence = buildSequence();

  it('moves exactly one position in either direction', () => {
    expect(getPreviewNavigationTarget(sequence, 2, -1)).toEqual({
      item: item('image', 'newest'),
      kind: 'item',
    });
    expect(getPreviewNavigationTarget(sequence, 2, 1)).toEqual({
      item: item('image', 'oldest'),
      kind: 'item',
    });
  });

  it('steps between the newest item and the placeholder in descending order', () => {
    expect(getPreviewNavigationTarget(sequence, 1, -1)).toEqual({ kind: 'placeholder', placeholder });
    expect(getPreviewNavigationTarget(sequence, 0, 1)).toEqual({
      item: item('image', 'newest'),
      kind: 'item',
    });
  });

  it('steps between the oldest item and the placeholder in ascending order', () => {
    const ascending = buildSequence({ imageOrderDir: 'ASC' });

    expect(getPreviewNavigationTarget(ascending, 2, 1)).toEqual({ kind: 'placeholder', placeholder });
    expect(getPreviewNavigationTarget(ascending, 3, -1)).toEqual({
      item: item('image', 'oldest'),
      kind: 'item',
    });
  });

  it('clamps at both sequence boundaries', () => {
    expect(getPreviewNavigationTarget(sequence, 0, -1)).toBeNull();
    expect(getPreviewNavigationTarget(sequence, sequence.length - 1, 1)).toBeNull();
  });

  it('is a no-op for an unresolved cursor', () => {
    expect(getPreviewNavigationTarget(sequence, -1, -1)).toBeNull();
    expect(getPreviewNavigationTarget(sequence, -1, 1)).toBeNull();
  });
});
