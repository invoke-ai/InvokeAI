import type { GalleryQueuePlaceholder } from '@features/gallery/contracts';

import { describe, expect, it } from 'vitest';

import {
  getPreviewNavigationCursor,
  getPreviewNavigationSequence,
  getPreviewNavigationTarget,
  type PreviewNavigationItem,
} from './previewNavigation';

interface TestImage {
  imageName: string;
  starred?: boolean;
}

const placeholder: GalleryQueuePlaceholder = {
  boardId: 'none',
  height: 1024,
  id: 'queue-1:1',
  itemIndex: 1,
  queueItemId: 'queue-1',
  width: 1024,
};

const image = (imageName: string, starred = false): TestImage => ({ imageName, starred });

const names = (sequence: PreviewNavigationItem<TestImage>[]): string[] =>
  sequence.map((item) => (item.kind === 'image' ? item.image.imageName : 'placeholder'));

const buildSequence = (overrides: Partial<Parameters<typeof getPreviewNavigationSequence<TestImage>>[0]> = {}) =>
  getPreviewNavigationSequence<TestImage>({
    activePlaceholder: placeholder,
    boardId: 'none',
    boardImages: [image('newest'), image('middle'), image('oldest')],
    galleryView: 'images',
    imageOrderDir: 'DESC',
    starredFirst: false,
    ...overrides,
  });

describe('getPreviewNavigationSequence', () => {
  it('places the placeholder at the newest position in descending order', () => {
    expect(names(buildSequence())).toEqual(['placeholder', 'newest', 'middle', 'oldest']);
  });

  it('places the placeholder at the latest chronological position in ascending order', () => {
    expect(names(buildSequence({ imageOrderDir: 'ASC' }))).toEqual(['newest', 'middle', 'oldest', 'placeholder']);
  });

  it('places the placeholder after the leading starred block when starred-first sorts descending', () => {
    expect(
      names(
        buildSequence({
          boardImages: [image('starred-a', true), image('starred-b', true), image('newest'), image('oldest')],
          starredFirst: true,
        })
      )
    ).toEqual(['starred-a', 'starred-b', 'placeholder', 'newest', 'oldest']);
  });

  it('places the placeholder last when every image is starred', () => {
    expect(
      names(buildSequence({ boardImages: [image('starred-a', true), image('starred-b', true)], starredFirst: true }))
    ).toEqual(['starred-a', 'starred-b', 'placeholder']);
  });

  it('keeps the placeholder last in ascending order even with starred-first', () => {
    expect(
      names(
        buildSequence({
          boardImages: [image('starred-a', true), image('newest')],
          imageOrderDir: 'ASC',
          starredFirst: true,
        })
      )
    ).toEqual(['starred-a', 'newest', 'placeholder']);
  });

  it('excludes the placeholder when it belongs to another board or the assets view', () => {
    expect(names(buildSequence({ boardId: 'board-2' }))).toEqual(['newest', 'middle', 'oldest']);
    expect(names(buildSequence({ galleryView: 'assets' }))).toEqual(['newest', 'middle', 'oldest']);
  });

  it('returns only images when there is no active placeholder', () => {
    expect(names(buildSequence({ activePlaceholder: null }))).toEqual(['newest', 'middle', 'oldest']);
  });
});

describe('getPreviewNavigationCursor', () => {
  const sequence = buildSequence();

  it('resolves to the placeholder while following live', () => {
    expect(getPreviewNavigationCursor(sequence, { isFollowingLive: true, selectedImageName: 'middle' })).toBe(0);
  });

  it('resolves to the selected image otherwise', () => {
    expect(getPreviewNavigationCursor(sequence, { isFollowingLive: false, selectedImageName: 'middle' })).toBe(2);
  });

  it('is unresolved when the selected image is absent from the sequence', () => {
    expect(getPreviewNavigationCursor(sequence, { isFollowingLive: false, selectedImageName: 'missing' })).toBe(-1);
    expect(getPreviewNavigationCursor(sequence, { isFollowingLive: false, selectedImageName: null })).toBe(-1);
  });
});

describe('getPreviewNavigationTarget', () => {
  const sequence = buildSequence();

  it('moves exactly one position in either direction', () => {
    expect(getPreviewNavigationTarget(sequence, 2, -1)).toEqual({ image: image('newest'), kind: 'image' });
    expect(getPreviewNavigationTarget(sequence, 2, 1)).toEqual({ image: image('oldest'), kind: 'image' });
  });

  it('steps between the newest image and the placeholder in descending order', () => {
    expect(getPreviewNavigationTarget(sequence, 1, -1)).toEqual({ kind: 'placeholder', placeholder });
    expect(getPreviewNavigationTarget(sequence, 0, 1)).toEqual({ image: image('newest'), kind: 'image' });
  });

  it('steps between the oldest image and the placeholder in ascending order', () => {
    const ascending = buildSequence({ imageOrderDir: 'ASC' });

    expect(getPreviewNavigationTarget(ascending, 2, 1)).toEqual({ kind: 'placeholder', placeholder });
    expect(getPreviewNavigationTarget(ascending, 3, -1)).toEqual({ image: image('oldest'), kind: 'image' });
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
