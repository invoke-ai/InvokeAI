import type { GalleryItem } from '@features/gallery/core/items';

import { describe, expect, it } from 'vitest';

import {
  getGalleryPickerDefaultIndex,
  getGalleryPickerNeighborIndex,
  getGalleryPickerRemaining,
  getGalleryPickerSelectionAfterPick,
  getGalleryPickerStatus,
  getGalleryPickerTileState,
  isGalleryPickerFull,
  isGalleryPickerNavKey,
  isGalleryPickerNavKeyForField,
  type GalleryPickerSelection,
} from './galleryPicker';

const image = (name: string): GalleryItem => ({
  boardId: 'board',
  category: 'general',
  createdAt: '2026-09-01T00:00:00.000Z',
  fullUrl: `/full/${name}`,
  height: 64,
  isIntermediate: false,
  kind: 'image',
  name,
  starred: false,
  thumbnailUrl: `/thumb/${name}`,
  width: 64,
});

const video = (name: string): GalleryItem => ({
  boardId: 'board',
  category: 'general',
  createdAt: '2026-09-01T00:00:00.000Z',
  durationSeconds: 3,
  fps: 24,
  fullUrl: `/video/${name}`,
  height: 64,
  isIntermediate: false,
  kind: 'video',
  name,
  starred: false,
  thumbnailUrl: `/video-thumb/${name}`,
  width: 64,
});

const single: GalleryPickerSelection = { mode: 'single' };

describe('getGalleryPickerTileState', () => {
  it('marks kinds the slot cannot take as unsupported before anything else', () => {
    const selection: GalleryPickerSelection = {
      addedKeys: new Set(['video:v.mp4']),
      mode: 'multiple',
      remaining: { image: 0 },
    };

    expect(getGalleryPickerTileState(video('v.mp4'), ['image'], selection)).toBe('unsupported');
    expect(getGalleryPickerTileState(video('v.mp4'), ['image'], single)).toBe('unsupported');
  });

  it('is always pickable in single mode', () => {
    expect(getGalleryPickerTileState(image('a.png'), ['image'], single)).toBe('pickable');
  });

  it('reports added, then full, then pickable in multiple mode', () => {
    const selection: GalleryPickerSelection = {
      addedKeys: new Set(['image:a.png']),
      mode: 'multiple',
      remaining: { image: 0, video: 2 },
    };

    expect(getGalleryPickerTileState(image('a.png'), ['image', 'video'], selection)).toBe('added');
    expect(getGalleryPickerTileState(image('b.png'), ['image', 'video'], selection)).toBe('full');
    expect(getGalleryPickerTileState(video('v.mp4'), ['image', 'video'], selection)).toBe('pickable');
  });

  it('treats a kind without a remaining count as unlimited', () => {
    const selection: GalleryPickerSelection = { addedKeys: new Set(), mode: 'multiple', remaining: {} };

    expect(getGalleryPickerTileState(image('a.png'), ['image'], selection)).toBe('pickable');
  });
});

describe('isGalleryPickerFull', () => {
  it('never reports single mode as full', () => {
    expect(isGalleryPickerFull(['image'], single)).toBe(false);
  });

  it('is full only when every accepted kind is exhausted', () => {
    const halfFull: GalleryPickerSelection = { addedKeys: new Set(), mode: 'multiple', remaining: { image: 0 } };

    expect(isGalleryPickerFull(['image'], halfFull)).toBe(true);
    expect(isGalleryPickerFull(['image', 'video'], halfFull)).toBe(false);
    expect(
      isGalleryPickerFull(['image', 'video'], {
        addedKeys: new Set(),
        mode: 'multiple',
        remaining: { image: 0, video: 0 },
      })
    ).toBe(true);
  });
});

describe('getGalleryPickerSelectionAfterPick', () => {
  it('returns single mode untouched', () => {
    expect(getGalleryPickerSelectionAfterPick(single, image('a.png'))).toBe(single);
  });

  it('adds the key and decrements only the picked kind, never below zero', () => {
    const before: GalleryPickerSelection = {
      addedKeys: new Set(['image:a.png']),
      mode: 'multiple',
      remaining: { image: 1, video: 0 },
    };
    const after = getGalleryPickerSelectionAfterPick(before, image('b.png'));

    expect(after).toEqual({
      addedKeys: new Set(['image:a.png', 'image:b.png']),
      mode: 'multiple',
      remaining: { image: 0, video: 0 },
    });
    expect(before.addedKeys.has('image:b.png')).toBe(false);
    expect(getGalleryPickerSelectionAfterPick(after, image('c.png'))).toMatchObject({ remaining: { image: 0 } });
  });

  it('leaves an unlimited kind unlimited', () => {
    const after = getGalleryPickerSelectionAfterPick(
      { addedKeys: new Set(), mode: 'multiple', remaining: { video: 1 } },
      image('a.png')
    );

    expect(after).toMatchObject({ remaining: { video: 1 } });
    expect(isGalleryPickerFull(['image', 'video'], after)).toBe(false);
  });
});

describe('getGalleryPickerNeighborIndex', () => {
  // 4 columns, 10 tiles: rows [0..3] [4..7] [8,9]
  const count = 10;
  const columns = 4;

  it('walks reading order across row ends and clamps at both ends', () => {
    expect(getGalleryPickerNeighborIndex(3, count, columns, 'ArrowRight')).toBe(4);
    expect(getGalleryPickerNeighborIndex(4, count, columns, 'ArrowLeft')).toBe(3);
    expect(getGalleryPickerNeighborIndex(0, count, columns, 'ArrowLeft')).toBe(0);
    expect(getGalleryPickerNeighborIndex(9, count, columns, 'ArrowRight')).toBe(9);
  });

  it('moves by a column up and down and holds still over a ragged row', () => {
    expect(getGalleryPickerNeighborIndex(1, count, columns, 'ArrowDown')).toBe(5);
    expect(getGalleryPickerNeighborIndex(5, count, columns, 'ArrowUp')).toBe(1);
    expect(getGalleryPickerNeighborIndex(6, count, columns, 'ArrowDown')).toBe(6);
    expect(getGalleryPickerNeighborIndex(2, count, columns, 'ArrowUp')).toBe(2);
  });

  it('jumps with Home and End', () => {
    expect(getGalleryPickerNeighborIndex(5, count, columns, 'Home')).toBe(0);
    expect(getGalleryPickerNeighborIndex(5, count, columns, 'End')).toBe(9);
  });

  it('enters the grid at an edge when nothing is active, and yields -1 for an empty grid', () => {
    expect(getGalleryPickerNeighborIndex(-1, count, columns, 'ArrowDown')).toBe(0);
    expect(getGalleryPickerNeighborIndex(-1, count, columns, 'End')).toBe(9);
    expect(getGalleryPickerNeighborIndex(42, count, columns, 'ArrowUp')).toBe(0);
    expect(getGalleryPickerNeighborIndex(0, 0, columns, 'ArrowDown')).toBe(-1);
  });
});

describe('isGalleryPickerNavKey', () => {
  it('recognises the six navigation keys only', () => {
    expect(isGalleryPickerNavKey('ArrowDown')).toBe(true);
    expect(isGalleryPickerNavKey('Home')).toBe(true);
    expect(isGalleryPickerNavKey('Enter')).toBe(false);
    expect(isGalleryPickerNavKey('PageDown')).toBe(false);
  });
});

describe('getGalleryPickerRemaining', () => {
  it('sums the accepted kinds and treats an unlimited kind or single mode as null', () => {
    const selection: GalleryPickerSelection = {
      addedKeys: new Set(),
      mode: 'multiple',
      remaining: { image: 2, video: 1 },
    };

    expect(getGalleryPickerRemaining(['image', 'video'], selection)).toBe(3);
    expect(getGalleryPickerRemaining(['image'], selection)).toBe(2);
    expect(getGalleryPickerRemaining(['image'], { addedKeys: new Set(), mode: 'multiple', remaining: {} })).toBeNull();
    expect(getGalleryPickerRemaining(['image'], single)).toBeNull();
  });
});

describe('getGalleryPickerDefaultIndex', () => {
  it('prefers the first pickable tile and falls back to the first tile', () => {
    const items = [video('v.mp4'), image('a.png'), image('b.png')];

    expect(getGalleryPickerDefaultIndex(items, ['image'], single)).toBe(1);
    expect(getGalleryPickerDefaultIndex(items, ['video'], single)).toBe(0);
    expect(getGalleryPickerDefaultIndex([video('v.mp4')], ['image'], single)).toBe(0);
    expect(getGalleryPickerDefaultIndex([], ['image'], single)).toBe(-1);
    expect(
      getGalleryPickerDefaultIndex(items, ['image'], {
        addedKeys: new Set(['image:a.png']),
        mode: 'multiple',
        remaining: { image: 1 },
      })
    ).toBe(2);
  });
});

describe('getGalleryPickerStatus', () => {
  const base = {
    accept: ['image'] as const,
    activeItem: image('a.png'),
    isSearching: false,
    isWindowTruncated: false,
    loadedCount: 12,
    pane: 'items' as const,
    remaining: null,
    total: 48,
    visibleBoardCount: 3,
  };

  it('counts boards on the boards pane regardless of anything else', () => {
    expect(getGalleryPickerStatus({ ...base, pane: 'boards', remaining: 0 })).toEqual([
      { count: 3, kind: 'boardCount' },
    ]);
  });

  it('explains an unsupported highlight before capacity or counts', () => {
    expect(getGalleryPickerStatus({ ...base, activeItem: video('v.mp4'), remaining: 0 })).toEqual([
      { kind: 'unsupportedVideo' },
    ]);
    expect(getGalleryPickerStatus({ ...base, accept: ['video'], activeItem: image('a.png') })).toEqual([
      { kind: 'unsupportedImage' },
    ]);
  });

  it('composes capacity with the window cap or the listing count', () => {
    expect(getGalleryPickerStatus({ ...base, remaining: 2 })).toEqual([
      { count: 2, kind: 'remainingCount' },
      { count: 48, kind: 'itemCount' },
    ]);
    expect(getGalleryPickerStatus({ ...base, isSearching: true, remaining: 0 })).toEqual([
      { kind: 'remainingNone' },
      { count: 48, kind: 'matchCount' },
    ]);
    expect(getGalleryPickerStatus({ ...base, isWindowTruncated: true })).toEqual([{ count: 12, kind: 'windowLimit' }]);
    expect(getGalleryPickerStatus({ ...base, total: null })).toEqual([]);
  });
});

describe('isGalleryPickerNavKeyForField', () => {
  it('leaves the caret keys to the field once it has text', () => {
    expect(isGalleryPickerNavKeyForField('ArrowLeft', '')).toBe(true);
    expect(isGalleryPickerNavKeyForField('ArrowLeft', 'cat')).toBe(false);
    expect(isGalleryPickerNavKeyForField('Home', 'cat')).toBe(false);
    expect(isGalleryPickerNavKeyForField('ArrowDown', 'cat')).toBe(true);
    expect(isGalleryPickerNavKeyForField('ArrowUp', 'cat')).toBe(true);
  });
});
