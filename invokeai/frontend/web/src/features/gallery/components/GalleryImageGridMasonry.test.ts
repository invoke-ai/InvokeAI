import { describe, expect, test } from 'vitest';

import {
  deleteMasonryBackgroundInFlightImageNames,
  getMasonryInFlightImageNames,
  getMasonryInitialItemCount,
  getMasonryPrefetchImageNames,
  getMasonrySkippedImageNames,
  getMasonryWarmupImageNames,
  getShouldScheduleNextMasonryWarmupBatch,
  getStaticMasonryColumns,
  getUncachedMasonryImageNames,
  setMasonryBackgroundInFlightImageNames,
} from './masonryImageFetching';
import { getMasonryRenderState } from './masonryRenderState';
import { getMasonryScrollDirection } from './masonryScrollIntoView';

describe('masonry image fetching', () => {
  test('returns uncached image names in first-seen order', () => {
    expect(getUncachedMasonryImageNames(['a', 'b', 'a', 'c', 'd'], ['b', 'd'])).toEqual(['a', 'c']);
  });

  test('visible fetch ignores background in-flight image names', () => {
    const visibleInFlightImageNames = ['c'];
    const backgroundInFlightImageNames = ['b'];

    expect(getUncachedMasonryImageNames(['a', 'b', 'c', 'd'], [], visibleInFlightImageNames)).toEqual(['a', 'b', 'd']);
    expect(backgroundInFlightImageNames).toEqual(['b']);
  });

  test('visible fetch skips cached and visible in-flight image names', () => {
    expect(getUncachedMasonryImageNames(['a', 'b', 'c', 'd'], ['b'], ['c'])).toEqual(['a', 'd']);
  });

  test('combines visible in-flight names with background-owned names', () => {
    expect(
      getMasonryInFlightImageNames({
        backgroundInFlightImageNames: new Map([
          ['b', 1],
          ['c', 2],
        ]),
        visibleInFlightImageNames: ['a'],
      })
    ).toEqual(['a', 'b', 'c']);
  });

  test('clears background in-flight names owned by the same request', () => {
    const backgroundInFlightImageNames = new Map([
      ['a', 1],
      ['b', 2],
    ]);

    deleteMasonryBackgroundInFlightImageNames({
      backgroundInFlightImageNames,
      imageNames: ['a', 'b'],
      requestId: 1,
    });

    expect(backgroundInFlightImageNames).toEqual(new Map([['b', 2]]));
  });

  test('preserves background in-flight names owned by a newer request', () => {
    const backgroundInFlightImageNames = new Map([['a', 2]]);

    deleteMasonryBackgroundInFlightImageNames({
      backgroundInFlightImageNames,
      imageNames: ['a'],
      requestId: 1,
    });

    expect(backgroundInFlightImageNames).toEqual(new Map([['a', 2]]));
  });

  test('sets background in-flight names to the current request owner', () => {
    const backgroundInFlightImageNames = new Map([['a', 1]]);

    setMasonryBackgroundInFlightImageNames({
      backgroundInFlightImageNames,
      imageNames: ['a', 'b'],
      requestId: 2,
    });

    expect(backgroundInFlightImageNames).toEqual(
      new Map([
        ['a', 2],
        ['b', 2],
      ])
    );
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
  test('expands mounted range by the column-based buffer and prioritizes the scroll direction', () => {
    const imageNames = Array.from({ length: 300 }, (_, i) => `${i}`);

    expect(
      getMasonryPrefetchImageNames({
        cachedImageNames: [],
        columnCount: 3,
        imageNames,
        mountedRange: { endIndex: 150, startIndex: 150 },
        scrollDirection: 'down',
      })
    ).toEqual([...imageNames.slice(150, 300), ...imageNames.slice(0, 150).reverse()]);
  });

  test('prioritizes names above the mounted range when scrolling up', () => {
    const imageNames = Array.from({ length: 20 }, (_, i) => `${i}`);

    expect(
      getMasonryPrefetchImageNames({
        batchSize: 6,
        cachedImageNames: [],
        columnCount: 1,
        imageNames,
        mountedRange: { endIndex: 10, startIndex: 10 },
        scrollDirection: 'up',
      })
    ).toEqual(['10', '9', '8', '7', '6', '5']);
  });

  test('caps prefetch batches', () => {
    const imageNames = Array.from({ length: 20 }, (_, i) => `${i}`);

    expect(
      getMasonryPrefetchImageNames({
        batchSize: 4,
        cachedImageNames: [],
        columnCount: 1,
        imageNames,
        mountedRange: { endIndex: 10, startIndex: 10 },
      })
    ).toEqual(['10', '11', '12', '13']);
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
    ).toEqual([...imageNames.slice(1), '0']);
  });

  test('filters cached names and deduplicates remaining names', () => {
    expect(
      getMasonryPrefetchImageNames({
        cachedImageNames: ['b'],
        columnCount: 1,
        imageNames: ['a', 'b', 'a', 'c'],
        inFlightImageNames: ['c'],
        mountedRange: { endIndex: 3, startIndex: 0 },
      })
    ).toEqual(['a']);
  });

  test('skips cached, visible in-flight, and background in-flight names', () => {
    expect(
      getMasonryPrefetchImageNames({
        cachedImageNames: ['b'],
        columnCount: 1,
        imageNames: ['a', 'b', 'c', 'd'],
        inFlightImageNames: ['c', 'd'],
        mountedRange: { endIndex: 3, startIndex: 0 },
      })
    ).toEqual(['a']);
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

describe('masonry warmup', () => {
  test('caps warmup batches', () => {
    expect(
      getMasonryWarmupImageNames({
        batchSize: 3,
        cachedImageNames: [],
        imageNames: ['a', 'b', 'c', 'd'],
        maxImageCount: 10,
      })
    ).toEqual(['a', 'b', 'c']);
  });

  test('preserves image order while skipping cached and in-flight names', () => {
    expect(
      getMasonryWarmupImageNames({
        batchSize: 10,
        cachedImageNames: ['b'],
        imageNames: ['a', 'b', 'c', 'd'],
        inFlightImageNames: ['c'],
        maxImageCount: 10,
      })
    ).toEqual(['a', 'd']);
  });

  test('skips cached, visible in-flight, and background in-flight names', () => {
    expect(
      getMasonryWarmupImageNames({
        batchSize: 10,
        cachedImageNames: ['b'],
        imageNames: ['a', 'b', 'c', 'd'],
        inFlightImageNames: ['c', 'd'],
        maxImageCount: 10,
      })
    ).toEqual(['a']);
  });

  test('skips names omitted by a previous successful warmup response', () => {
    expect(
      getMasonryWarmupImageNames({
        batchSize: 10,
        cachedImageNames: [],
        imageNames: ['a', 'b', 'c', 'd'],
        maxImageCount: 10,
        skippedImageNames: ['b', 'c'],
      })
    ).toEqual(['a', 'd']);
  });

  test('uses the current image list as the warmup reset boundary', () => {
    expect(
      getMasonryWarmupImageNames({
        batchSize: 10,
        cachedImageNames: ['a', 'b'],
        imageNames: ['x', 'y', 'z'],
        maxImageCount: 2,
      })
    ).toEqual(['x', 'y']);
  });

  test('resets skipped warmup names with a fresh skipped-name set', () => {
    expect(
      getMasonryWarmupImageNames({
        batchSize: 10,
        cachedImageNames: [],
        imageNames: ['a', 'b'],
        maxImageCount: 10,
        skippedImageNames: [],
      })
    ).toEqual(['a', 'b']);
  });

  test('returns no skipped names when warmup response covers all requests', () => {
    expect(
      getMasonrySkippedImageNames({
        requestedImageNames: ['a', 'b'],
        returnedImageNames: ['a', 'b'],
      })
    ).toEqual([]);
  });

  test('returns only response-omitted warmup names in requested order', () => {
    expect(
      getMasonrySkippedImageNames({
        requestedImageNames: ['a', 'b', 'c', 'd'],
        returnedImageNames: ['b', 'd'],
      })
    ).toEqual(['a', 'c']);
  });

  test('schedules the next batch only after a successful warmup fetch', () => {
    expect(
      getShouldScheduleNextMasonryWarmupBatch({
        didFetchBatch: true,
        imageNamesToFetchCount: 3,
        isCancelled: false,
      })
    ).toBe(true);
  });

  test('stops without scheduling after a failed warmup fetch', () => {
    expect(
      getShouldScheduleNextMasonryWarmupBatch({
        didFetchBatch: false,
        imageNamesToFetchCount: 3,
        isCancelled: false,
      })
    ).toBe(false);
  });

  test('stops without scheduling when the warmup batch is empty', () => {
    expect(
      getShouldScheduleNextMasonryWarmupBatch({
        didFetchBatch: true,
        imageNamesToFetchCount: 0,
        isCancelled: false,
      })
    ).toBe(false);
  });
});

describe('masonry initial item count', () => {
  test('clamps the initial render to the configured limit', () => {
    expect(
      getMasonryInitialItemCount({
        columnCount: 20,
        imageCount: 1000,
        initialItemCountLimit: 256,
        itemsPerColumn: 48,
        minimumInitialItemCount: 128,
      })
    ).toBe(256);
  });

  test('uses the minimum initial item count when the gallery is large enough', () => {
    expect(
      getMasonryInitialItemCount({
        columnCount: 1,
        imageCount: 500,
        initialItemCountLimit: 256,
        itemsPerColumn: 48,
        minimumInitialItemCount: 128,
      })
    ).toBe(128);
  });

  test('scales the initial render by column count', () => {
    expect(
      getMasonryInitialItemCount({
        columnCount: 4,
        imageCount: 500,
        initialItemCountLimit: 256,
        itemsPerColumn: 48,
        minimumInitialItemCount: 128,
      })
    ).toBe(192);
  });

  test('clamps the initial render to small gallery size', () => {
    expect(
      getMasonryInitialItemCount({
        columnCount: 4,
        imageCount: 20,
        initialItemCountLimit: 256,
        itemsPerColumn: 48,
        minimumInitialItemCount: 128,
      })
    ).toBe(20);
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
