import { describe, expect, test } from 'vitest';

import {
  getMasonryPrefetchImageNames,
  getStaticMasonryColumns,
  getUncachedMasonryImageNames,
} from './masonryImageFetching';
import { getMasonryRenderState } from './masonryRenderState';
import { getMasonryScrollDirection } from './masonryScrollIntoView';

describe('masonry image fetching', () => {
  test('returns uncached image names in first-seen order', () => {
    expect(getUncachedMasonryImageNames(['a', 'b', 'a', 'c', 'd'], ['b', 'd'])).toEqual(['a', 'c']);
  });
});

describe('masonry render state', () => {
  test('renders loading and empty states without waiting for column measurement', () => {
    expect(getMasonryRenderState({ hasMeasuredColumnCount: false, imageCount: 0, isLoading: true })).toBe('loading');
    expect(getMasonryRenderState({ hasMeasuredColumnCount: false, imageCount: 0, isLoading: false })).toBe('empty');
  });

  test('waits for column measurement only before rendering non-empty masonry content', () => {
    expect(getMasonryRenderState({ hasMeasuredColumnCount: false, imageCount: 1, isLoading: false })).toBe('measuring');
    expect(getMasonryRenderState({ hasMeasuredColumnCount: true, imageCount: 1, isLoading: false })).toBe('ready');
  });
});

describe('masonry prefetching', () => {
  test('expands mounted range by the column-based buffer', () => {
    const imageNames = Array.from({ length: 300 }, (_, i) => `${i}`);

    expect(
      getMasonryPrefetchImageNames({
        cachedImageNames: [],
        columnCount: 3,
        imageNames,
        mountedRange: { endIndex: 150, startIndex: 150 },
      })
    ).toEqual(imageNames.slice(54, 247));
  });

  test('clamps expanded mounted range to image list bounds', () => {
    const imageNames = Array.from({ length: 20 }, (_, i) => `${i}`);

    expect(
      getMasonryPrefetchImageNames({
        cachedImageNames: [],
        columnCount: 3,
        imageNames,
        mountedRange: { endIndex: 2, startIndex: 1 },
      })
    ).toEqual(imageNames);
  });

  test('filters cached names and deduplicates remaining names', () => {
    expect(
      getMasonryPrefetchImageNames({
        cachedImageNames: ['b'],
        columnCount: 1,
        imageNames: ['a', 'b', 'a', 'c'],
        mountedRange: { endIndex: 3, startIndex: 0 },
      })
    ).toEqual(['a', 'c']);
  });

  test('returns no names when mounted range is unavailable', () => {
    expect(
      getMasonryPrefetchImageNames({
        cachedImageNames: [],
        columnCount: 1,
        imageNames: ['a', 'b', 'c'],
        mountedRange: null,
      })
    ).toEqual([]);
  });
});

describe('masonry scroll direction', () => {
  test('does not request a viewport scroll for a mounted target', () => {
    expect(
      getMasonryScrollDirection({
        mountedRange: { endIndex: 20, startIndex: 10 },
        previousIndex: 19,
        targetIndex: 15,
      })
    ).toBeNull();
  });

  test('scrolls down when the target is below the mounted range', () => {
    expect(
      getMasonryScrollDirection({
        mountedRange: { endIndex: 20, startIndex: 10 },
        previousIndex: 19,
        targetIndex: 24,
      })
    ).toBe('down');
  });

  test('scrolls up when the target is above the mounted range', () => {
    expect(
      getMasonryScrollDirection({
        mountedRange: { endIndex: 20, startIndex: 10 },
        previousIndex: 11,
        targetIndex: 6,
      })
    ).toBe('up');
  });

  test('falls back to previous index when the mounted range is unavailable', () => {
    expect(
      getMasonryScrollDirection({
        mountedRange: null,
        previousIndex: 11,
        targetIndex: 15,
      })
    ).toBe('down');
    expect(
      getMasonryScrollDirection({
        mountedRange: null,
        previousIndex: 11,
        targetIndex: 7,
      })
    ).toBe('up');
  });
});

describe('static masonry column packing', () => {
  test('places each image into the shortest current column by aspect height', () => {
    expect(
      getStaticMasonryColumns({
        columnCount: 2,
        imageDimensionsByName: new Map([
          ['tall', { height: 300, width: 100 }],
          ['wide-1', { height: 100, width: 300 }],
          ['wide-2', { height: 100, width: 300 }],
          ['wide-3', { height: 100, width: 300 }],
        ]),
        imageNames: ['tall', 'wide-1', 'wide-2', 'wide-3'],
      })
    ).toEqual([
      [{ imageName: 'tall', index: 0 }],
      [
        { imageName: 'wide-1', index: 1 },
        { imageName: 'wide-2', index: 2 },
        { imageName: 'wide-3', index: 3 },
      ],
    ]);
  });

  test('uses square fallback dimensions until image dimensions are cached', () => {
    expect(
      getStaticMasonryColumns({
        columnCount: 2,
        imageDimensionsByName: new Map(),
        imageNames: ['a', 'b', 'c'],
      })
    ).toEqual([
      [
        { imageName: 'a', index: 0 },
        { imageName: 'c', index: 2 },
      ],
      [{ imageName: 'b', index: 1 }],
    ]);
  });
});
