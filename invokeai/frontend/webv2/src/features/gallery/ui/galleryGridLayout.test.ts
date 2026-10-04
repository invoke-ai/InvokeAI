import type { GalleryImageItem } from '@features/gallery/core/items';

import { describe, expect, it } from 'vitest';

import {
  buildGalleryGridRows,
  GALLERY_GRID_GAP_PX,
  GALLERY_PINNED_FOOTER_PX,
  GALLERY_STARRED_HEADER_HEIGHT_PX,
  getGalleryCellSizePx,
  getGalleryColumnCount,
  getGalleryColumnCountForCell,
  getGalleryGridRowIndexForItemKey,
  getGalleryPinnedHeightPx,
  getGalleryStarredLayout,
  getGalleryStarredStripItems,
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
  starred,
  thumbnailUrl: `/api/v1/images/i/${name}/thumbnail`,
  width: 512,
});

const buildRows = (items: GalleryImageItem[]) => buildGalleryGridRows(items, 2);

describe('getGalleryColumnCountForCell', () => {
  it('rounds to the nearest whole cell and clamps to the caller bounds', () => {
    const bounds = { max: 8, min: 3, targetCellPx: 72 };

    expect(getGalleryColumnCountForCell({ ...bounds, widthPx: 320 })).toBe(4);
    expect(getGalleryColumnCountForCell({ ...bounds, widthPx: 120 })).toBe(3);
    expect(getGalleryColumnCountForCell({ ...bounds, widthPx: 2000 })).toBe(8);
  });

  it('falls back to the minimum before the width is measured', () => {
    expect(getGalleryColumnCountForCell({ max: 8, min: 3, targetCellPx: 72, widthPx: 0 })).toBe(3);
  });
});

describe('getGalleryColumnCount', () => {
  it('gives the same answer at the same width regardless of placement', () => {
    const stacked = getGalleryColumnCount({ imageDensityPercent: 50, widthPx: 600 });
    const wide = getGalleryColumnCount({ imageDensityPercent: 50, widthPx: 600 });

    expect(stacked).toBe(wide);
  });

  it('fits more columns as the viewport grows', () => {
    const narrow = getGalleryColumnCount({ imageDensityPercent: 50, widthPx: 360 });
    const roomy = getGalleryColumnCount({ imageDensityPercent: 50, widthPx: 1200 });

    expect(roomy).toBeGreaterThan(narrow);
  });

  it('fits more columns as density rises', () => {
    const sparse = getGalleryColumnCount({ imageDensityPercent: 0, widthPx: 800 });
    const dense = getGalleryColumnCount({ imageDensityPercent: 100, widthPx: 800 });

    expect(dense).toBeGreaterThan(sparse);
  });

  it('clamps to a usable range at both extremes', () => {
    expect(getGalleryColumnCount({ imageDensityPercent: 100, widthPx: 40 })).toBe(2);
    expect(getGalleryColumnCount({ imageDensityPercent: 100, widthPx: 8000 })).toBe(48);
    expect(getGalleryColumnCount({ imageDensityPercent: 0, widthPx: 8000 })).toBe(42);
  });

  it('keeps density effective in a wide placement instead of pinning it at a column cap', () => {
    // A bottom panel around 1100px wide: every density step must still change the count.
    const counts = [0, 25, 50, 75, 100].map((percent) =>
      getGalleryColumnCount({ imageDensityPercent: percent, widthPx: 1100 })
    );

    expect(counts).toEqual([...counts].sort((a, b) => a - b));
    expect(new Set(counts).size).toBe(counts.length);
  });

  it('ignores out-of-range density instead of producing a nonsense count', () => {
    expect(getGalleryColumnCount({ imageDensityPercent: -50, widthPx: 800 })).toBe(
      getGalleryColumnCount({ imageDensityPercent: 0, widthPx: 800 })
    );
    expect(getGalleryColumnCount({ imageDensityPercent: 500, widthPx: 800 })).toBe(
      getGalleryColumnCount({ imageDensityPercent: 100, widthPx: 800 })
    );
  });

  it('falls back to the minimum before the viewport has been measured', () => {
    expect(getGalleryColumnCount({ imageDensityPercent: 50, widthPx: 0 })).toBe(2);
  });
});

describe('getGalleryCellSizePx', () => {
  it('divides the width evenly after removing the inter-column gaps', () => {
    expect(getGalleryCellSizePx({ columnCount: 4, widthPx: 400 + GALLERY_GRID_GAP_PX * 3 })).toBe(100);
  });

  it('uses a plausible square before measurement so the first paint is not zero-height', () => {
    expect(getGalleryCellSizePx({ columnCount: 4, widthPx: 0 })).toBe(96);
  });

  it('never returns a non-positive size when the width is smaller than the gaps', () => {
    expect(getGalleryCellSizePx({ columnCount: 12, widthPx: 4 })).toBeGreaterThan(0);
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

  it('closes the pinned block with its footer only when something is pinned', () => {
    expect(getGalleryPinnedHeightPx(0, 0)).toBe(0);
    expect(getGalleryPinnedHeightPx(40, 0)).toBe(40 + GALLERY_PINNED_FOOTER_PX);
    expect(getGalleryPinnedHeightPx(40, 124)).toBe(164 + GALLERY_PINNED_FOOTER_PX);
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
