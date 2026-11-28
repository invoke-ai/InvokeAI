import type { GalleryImageItem } from '@features/gallery/core/items';

import { GALLERY_PAGE_SIZE } from '@features/gallery/core/paging';
import { describe, expect, it } from 'vitest';

import {
  buildGalleryGridRows,
  buildSparseGalleryNavigationEntries,
  GALLERY_GRID_GAP_PX,
  GALLERY_STARRED_HEADER_HEIGHT_PX,
  getGalleryCellSizePx,
  getGalleryColumnCount,
  getGalleryColumnCountForCell,
  getGalleryGridRowIndexForItemKey,
  getGallerySparseRowIndexForItemKey,
  getGallerySparseRowKey,
  getGallerySparseSelectionPages,
  getGallerySparseSlotKey,
  getGalleryStarredLayout,
  getGalleryStarredStripItems,
  planGalleryPageOffsets,
} from './galleryGridLayout';

const createImageItem = (name: string, starred = false): GalleryImageItem => ({
  boardId: 'none',
  category: 'general',
  createdAt: '2026-06-09T00:00:00.000Z',
  fullUrl: `/api/v1/images/i/${name}/full`,
  height: 768,
  isIntermediate: false,
  kind: 'image',
  name,
  colorLabel: null,
  starred,
  rating: 0,
  rejected: false,
  thumbnailUrl: `/api/v1/images/i/${name}/thumbnail`,
  width: 512,
});

const buildRows = (items: GalleryImageItem[]) => buildGalleryGridRows(items, 2);

describe('planGalleryPageOffsets', () => {
  it.each([
    { endIndexExclusive: 0, expected: [], startIndex: 0, total: 0 },
    { endIndexExclusive: 59, expected: [0], startIndex: 2, total: 120 },
    { endIndexExclusive: GALLERY_PAGE_SIZE, expected: [0], startIndex: 0, total: 120 },
    { endIndexExclusive: GALLERY_PAGE_SIZE + 1, expected: [0, GALLERY_PAGE_SIZE], startIndex: 59, total: 120 },
    { endIndexExclusive: 1_000, expected: [0, GALLERY_PAGE_SIZE], startIndex: 58, total: 61 },
    { endIndexExclusive: 100, expected: [], startIndex: 90, total: 61 },
    {
      endIndexExclusive: 721,
      expected: Array.from({ length: 13 }, (_, index) => index * GALLERY_PAGE_SIZE),
      startIndex: 0,
      total: null,
    },
  ])(
    'plans offsets for range [$startIndex, $endIndexExclusive) with total $total',
    ({ endIndexExclusive, expected, startIndex, total }) => {
      expect(planGalleryPageOffsets({ endIndexExclusive, startIndex, total })).toEqual(expected);
    }
  );
});

describe('getGalleryColumnCountForCell', () => {
  it('rounds to the nearest whole cell and clamps to the caller bounds', () => {
    const bounds = { max: 8, min: 3, targetCellPx: 72 };

    expect(getGalleryColumnCountForCell({ ...bounds, widthPx: 320 })).toBe(4);
    expect(getGalleryColumnCountForCell({ ...bounds, widthPx: 120 })).toBe(3);
    expect(getGalleryColumnCountForCell({ ...bounds, widthPx: 2000 })).toBe(8);
  });
});

describe('getGalleryColumnCount', () => {
  it('fits more columns as the viewport grows', () => {
    const narrow = getGalleryColumnCount({ imageDensityPercent: 50, widthPx: 360 });
    const roomy = getGalleryColumnCount({ imageDensityPercent: 50, widthPx: 1200 });

    expect(roomy).toBeGreaterThan(narrow);
  });
});

describe('getGalleryCellSizePx', () => {
  it('divides the width evenly after removing the inter-column gaps', () => {
    expect(getGalleryCellSizePx({ columnCount: 4, widthPx: 400 + GALLERY_GRID_GAP_PX * 3 })).toBe(100);
  });
});

describe('buildGalleryGridRows', () => {
  it('chunks the listing into rows of the column count, keyed by their leading cell', () => {
    const rows = buildRows(['a', 'b', 'c'].map((name) => createImageItem(name)));

    expect(rows.map((row) => row.cells.map((cell) => cell.name))).toEqual([['a', 'b'], ['c']]);
    expect(rows.map((row) => row.key)).toEqual(['regular:image:a', 'regular:image:c']);
  });

  it('caps the strip at three rows of the current column count', () => {
    const starredItems = Array.from({ length: 7 }, (_, index) => createImageItem(`starred-${index}`, true));

    expect(getGalleryStarredStripItems(starredItems, 2).map((item) => item.name)).not.toContain('starred-6');
    expect(getGalleryStarredStripItems(starredItems, 2)).toHaveLength(6);
  });
});

describe('pinned sections layout', () => {
  it('sizes the starred strip by its disclosure row plus its open rows', () => {
    const open = getGalleryStarredLayout({ collapsed: false, columns: 2, shownCount: 3, tileSize: 96 });
    const collapsed = getGalleryStarredLayout({ collapsed: true, columns: 2, shownCount: 3, tileSize: 96 });

    expect(open.rowCount).toBe(2);
    expect(open.height).toBe(GALLERY_STARRED_HEADER_HEIGHT_PX + 2 * (96 + GALLERY_GRID_GAP_PX));
    expect(collapsed.height).toBe(GALLERY_STARRED_HEADER_HEIGHT_PX);
    expect(getGalleryStarredLayout({ collapsed: false, columns: 2, shownCount: 0, tileSize: 96 }).height).toBe(0);
  });
});

describe('getGalleryGridRowIndexForItemKey', () => {
  it('finds the listing row holding an item; a strip item has no row', () => {
    const items = [createImageItem('regular-1'), createImageItem('regular-2'), createImageItem('regular-3')];

    expect(getGalleryGridRowIndexForItemKey(items, 'image:starred-1', 2)).toBe(-1);
    expect(getGalleryGridRowIndexForItemKey(items, 'image:regular-1', 2)).toBe(0);
    expect(getGalleryGridRowIndexForItemKey(items, 'image:regular-3', 2)).toBe(1);
  });
});

describe('sparse gallery geometry', () => {
  it('maps loaded sparse items to their absolute selection pages', () => {
    const infiniteItem = createImageItem('infinite');
    const paginatedItem = createImageItem('paginated');

    expect(
      getGallerySparseSelectionPages({
        itemSlots: new Map([[180, infiniteItem]]),
        pageOffset: 0,
      }).get('image:infinite')
    ).toBe(3);
    expect(
      getGallerySparseSelectionPages({
        itemSlots: new Map([[5, paginatedItem]]),
        pageOffset: GALLERY_PAGE_SIZE * 2,
      }).get('image:paginated')
    ).toBe(2);
  });

  it('maps distant item positions to absolute rows and uses position-stable identities', () => {
    const first = createImageItem('first');
    const third = createImageItem('third');
    const slots = new Map([
      [120, first],
      [122, third],
    ]);

    expect(getGallerySparseRowIndexForItemKey(slots, 'image:first', 2, 3)).toBe(63);
    expect(getGallerySparseRowIndexForItemKey(slots, 'image:third', 2, 3)).toBe(64);
    expect(getGallerySparseRowIndexForItemKey(slots, 'image:missing', 2, 3)).toBe(-1);
    expect(getGallerySparseRowKey(60)).toBe('listing-row:60');
    expect(getGallerySparseSlotKey(122)).toBe('listing-slot:122');
  });

  it('keeps keyboard row positions across page-local hydration holes', () => {
    const first = createImageItem('first');
    const third = createImageItem('third');
    const entries = buildSparseGalleryNavigationEntries({
      itemSlots: new Map([
        [120, first],
        [122, third],
      ]),
      pageOffsets: [120],
      total: 130,
    });

    expect(entries).toHaveLength(11);
    expect(entries[0]).toEqual({ id: 'gallery-unloaded-slot:119', kind: 'slot', navigable: true });
    expect(entries[1]).toEqual({ item: first, kind: 'item' });
    expect(entries[2]).toEqual({ id: 'gallery-gap-slot:121', kind: 'slot', navigable: false });
    expect(entries[3]).toEqual({ item: third, kind: 'item' });
  });

  it('covers only the runs of subscribed pages, not the positions between them', () => {
    const first = createImageItem('first');
    const revealed = createImageItem('revealed');
    const entries = buildSparseGalleryNavigationEntries({
      columnCount: 2,
      itemSlots: new Map([
        [0, first],
        [600_000, revealed],
      ]),
      pageOffsets: [0, 600_000],
      total: 700_000,
    });

    // Each page plus one boundary row on each side of the distant run, and one after the first.
    expect(entries).toHaveLength(60 + 2 + 2 + 60 + 2);
    expect(entries[0]).toEqual({ item: first, kind: 'item' });
    expect(entries[62]).toEqual({ id: 'gallery-unloaded-slot:599998', kind: 'slot', navigable: true });
    expect(entries[64]).toEqual({ item: revealed, kind: 'item' });
  });
});
