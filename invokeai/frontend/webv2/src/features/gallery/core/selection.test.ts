import { describe, expect, it } from 'vitest';

import type { GalleryImageItem, GalleryItem, GalleryVideoItem } from './items';
import type { GeneratedImageContract } from './types';

import * as selection from './selection';

const legacyImage: GeneratedImageContract = {
  height: 768,
  imageName: 'legacy.png',
  imageUrl: '/images/legacy.png',
  queuedAt: '2026-07-30T00:00:00.000Z',
  sourceQueueItemId: 'queue-item',
  thumbnailUrl: '/thumbnails/legacy.png',
  width: 512,
};

const imageItem: GalleryImageItem = {
  boardId: 'board-1',
  category: 'general',
  createdAt: '2026-07-30T01:00:00.000Z',
  fullUrl: '/images/shared',
  height: 768,
  isIntermediate: false,
  kind: 'image',
  name: 'shared',
  sourceQueueItemId: 'backend-gallery',
  starred: true,
  thumbnailUrl: '/thumbnails/shared',
  width: 512,
};

const videoItem: GalleryVideoItem = {
  boardId: 'board-1',
  category: 'general',
  createdAt: '2026-07-30T02:00:00.000Z',
  durationSeconds: 4,
  fullUrl: '/videos/shared',
  height: 768,
  isIntermediate: false,
  kind: 'video',
  name: 'shared',
  starred: false,
  thumbnailUrl: '/video-thumbnails/shared',
  width: 512,
};

type SelectionModule = typeof selection & {
  getPersistedSelectedGalleryItemKeys?: (values: Record<string, unknown>) => string[];
  getSelectedGalleryItemFromValues?: (values: Record<string, unknown>) => GalleryItem | null;
};

const getSelectedGalleryItemFromValues = (values: Record<string, unknown>): GalleryItem | null | undefined =>
  (selection as SelectionModule).getSelectedGalleryItemFromValues?.(values);
const getPersistedSelectedGalleryItemKeys = (values: Record<string, unknown>): string[] | undefined =>
  (selection as SelectionModule).getPersistedSelectedGalleryItemKeys?.(values);

describe('persisted Gallery selection readers', () => {
  it('returns a canonical selected image item unchanged', () => {
    expect(getSelectedGalleryItemFromValues({ selectedImage: imageItem })).toBe(imageItem);
  });

  it('returns a canonical selected video while the image-only reader rejects it', () => {
    const values = { selectedImage: videoItem, selectedImageName: 'video:shared' };

    expect(getSelectedGalleryItemFromValues(values)).toBe(videoItem);
    expect(selection.getSelectedGalleryImageFromValues(values)).toBeNull();
  });

  it('adapts a legacy generated-image object into a canonical image item', () => {
    expect(
      getSelectedGalleryItemFromValues({
        selectedBoardId: 'board-1',
        selectedImage: legacyImage,
      })
    ).toEqual({
      boardId: 'board-1',
      category: 'general',
      createdAt: legacyImage.queuedAt,
      fullUrl: legacyImage.imageUrl,
      height: legacyImage.height,
      isIntermediate: false,
      kind: 'image',
      name: legacyImage.imageName,
      sourceQueueItemId: legacyImage.sourceQueueItemId,
      starred: false,
      thumbnailUrl: legacyImage.thumbnailUrl,
      width: legacyImage.width,
    });
  });

  it.each(['legacy.png', 'image:legacy.png'])(
    'adapts the legacy recent-image fallback selected by %s',
    (selectedImageName) => {
      const values = {
        recentImages: [legacyImage],
        selectedBoardId: 'board-1',
        selectedImageName,
      };

      expect(getSelectedGalleryItemFromValues(values)).toMatchObject({
        boardId: 'board-1',
        kind: 'image',
        name: legacyImage.imageName,
      });
      expect(selection.getSelectedGalleryImageFromValues(values)).toMatchObject({
        boardId: 'board-1',
        imageName: legacyImage.imageName,
      });
    }
  );

  it('converts a canonical image item for legacy image-only consumers', () => {
    expect(selection.getSelectedGalleryImageFromValues({ selectedImage: imageItem })).toEqual({
      boardId: imageItem.boardId,
      createdAt: imageItem.createdAt,
      height: imageItem.height,
      imageCategory: imageItem.category,
      imageName: imageItem.name,
      imageUrl: imageItem.fullUrl,
      queuedAt: imageItem.createdAt,
      sourceQueueItemId: imageItem.sourceQueueItemId,
      starred: imageItem.starred,
      thumbnailUrl: imageItem.thumbnailUrl,
      width: imageItem.width,
    });
  });

  it('canonicalizes qualified and bare keys from a persisted multi-selection', () => {
    expect(
      getPersistedSelectedGalleryItemKeys({
        selectedImageNames: ['video:clip.mp4', 'legacy.png', 'image:a:b.png', 42],
        selectedImageName: 'ignored.png',
        selectedImage: videoItem,
      })
    ).toEqual(['video:clip.mp4', 'image:legacy.png', 'image:a:b.png']);
  });

  it('falls back to the legacy singular persisted selection key', () => {
    expect(getPersistedSelectedGalleryItemKeys({ selectedImageName: 'legacy.png' })).toEqual(['image:legacy.png']);
  });

  it('falls back to the canonical selected item object when persisted keys are absent', () => {
    expect(getPersistedSelectedGalleryItemKeys({ selectedImage: videoItem })).toEqual(['video:shared']);
  });
});

describe('getGalleryDeletionSuccessor', () => {
  const ref = (name: string): { kind: 'image'; name: string } => ({ kind: 'image', name });
  const keys = (...names: string[]): Set<`image:${string}`> => new Set(names.map((name) => `image:${name}` as const));

  it('picks the next item in display order — the one that slides into the deleted slot', () => {
    expect(selection.getGalleryDeletionSuccessor([ref('a'), ref('b'), ref('c')], 'image:b', keys('b'))).toEqual(
      ref('c')
    );
  });

  it('falls back to the nearest earlier item at the end of the list', () => {
    expect(selection.getGalleryDeletionSuccessor([ref('a'), ref('b'), ref('c')], 'image:c', keys('c'))).toEqual(
      ref('b')
    );
  });

  it('skips ineligible refs in both directions', () => {
    expect(
      selection.getGalleryDeletionSuccessor([ref('a'), ref('b'), ref('c'), ref('d')], 'image:b', keys('b', 'c', 'd'))
    ).toEqual(ref('a'));
  });

  it('stays in the regular block when starred-first names lead the list', () => {
    // Deleting a section's first item must not select the preceding section's last starred item.
    const orderedRefs = [ref('starred-1'), ref('starred-2'), ref('regular-1'), ref('regular-2')];

    expect(selection.getGalleryDeletionSuccessor(orderedRefs, 'image:regular-1', keys('regular-1'))).toEqual(
      ref('regular-2')
    );
  });

  it('returns null when the primary is not in the list', () => {
    expect(selection.getGalleryDeletionSuccessor([ref('a')], 'image:missing', keys('a'))).toBeNull();
    expect(selection.getGalleryDeletionSuccessor([], 'image:a', keys('a'))).toBeNull();
  });

  it('returns null when nothing survives', () => {
    expect(selection.getGalleryDeletionSuccessor([ref('a')], 'image:a', keys('a'))).toBeNull();
  });
});

describe('getGalleryNavigationStep', () => {
  const item = (name: string, starred = false): GalleryItem => ({ ...imageItem, name, starred });
  const session = (id: string, navigable = true) => ({ id, kind: 'session' as const, navigable });
  const entry = (name: string, starred = false) => ({ item: item(name, starred), kind: 'item' as const });
  // Sections have partial rows: four in-progress, five starred, then four listing items across three columns.
  const sections = [
    [session('p0'), session('p1', false), session('p2'), session('p3')],
    ['s0', 's1', 's2', 's3', 's4'].map((name) => entry(name, true)),
    ['r0', 'r1', 'r2', 'r3'].map((name) => entry(name)),
  ];
  const step = (cursorKey: string | null, direction: selection.GalleryNavigationDirection) => {
    const next = selection.getGalleryNavigationStep(sections, cursorKey, direction, 3);

    return next === null ? null : next.kind === 'session' ? next.id : next.item.name;
  };

  it('walks left and right across every seam, skipping tiles that cannot be followed', () => {
    expect(step('session:p3', 'right')).toBe('s0');
    expect(step('image:s0', 'left')).toBe('p3');
    expect(step('image:s4', 'right')).toBe('r0');
    expect(step('image:r0', 'left')).toBe('s4');
    expect(step('session:p0', 'right')).toBe('p2');
    expect(step('session:p2', 'left')).toBe('p0');
    expect(step('session:p0', 'left')).toBeNull();
    expect(step('image:r3', 'right')).toBeNull();
  });

  it('keeps the column across seams and clamps to a shorter row', () => {
    // r1 (column 1) rises onto the strip's partial row [s3, s4]; r2 clamps to its last cell.
    expect(step('image:r1', 'up')).toBe('s4');
    expect(step('image:r2', 'up')).toBe('s4');
    // s1 (column 1) rises past the partial in-progress row onto p3 (column 0).
    expect(step('image:s1', 'up')).toBe('p3');
    // s4 (column 1) descends to r1, not r0: the partial strip row does not shift the column.
    expect(step('image:s4', 'down')).toBe('r1');
    expect(step('image:s1', 'down')).toBe('s4');
    expect(step('session:p3', 'down')).toBe('s0');
    expect(step('image:r3', 'down')).toBeNull();
    expect(step('session:p0', 'up')).toBeNull();
  });

  it('lands on the nearest followable tile of a row, and skips a row with none', () => {
    // s1 (column 1) rising to the first in-progress row would land on p1, which is not running.
    expect(
      selection.getGalleryNavigationStep([sections[0]!.slice(0, 3), sections[1]!], 'image:s1', 'up', 3)
    ).toMatchObject({ id: 'p0' });
    expect(
      selection.getGalleryNavigationStep(
        [[session('p0', false), session('p1', false)], sections[1]!],
        'image:s1',
        'up',
        3
      )
    ).toBeNull();
  });

  it('steps away from a cursor that is itself not followable, such as a settling session', () => {
    expect(step('session:p1', 'right')).toBe('p2');
    expect(step('session:p1', 'left')).toBe('p0');
    // The in-progress row below holds only p3, so the column clamps to it.
    expect(step('session:p1', 'down')).toBe('p3');
  });

  it('skips a row of waiting tiles that sits between two followable rows', () => {
    const waitingRow = [session('q0', false), session('q1', false), session('q2', false)];

    expect(
      selection.getGalleryNavigationStep([[session('p0')], waitingRow, sections[1]!], 'image:s0', 'up', 3)
    ).toMatchObject({ id: 'p0' });
    expect(
      selection.getGalleryNavigationStep([[session('p0')], waitingRow, sections[1]!], 'session:p0', 'down', 3)
    ).toMatchObject({ item: { name: 's0' } });
  });

  it('lands on the first followable entry when the cursor is off the sequence', () => {
    expect(step(null, 'down')).toBe('p0');
    expect(step('image:gone', 'left')).toBe('p0');
    expect(step(null, 'right')).toBe('p0');
  });
});
