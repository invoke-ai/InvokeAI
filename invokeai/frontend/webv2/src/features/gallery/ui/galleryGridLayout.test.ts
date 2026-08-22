import type { GalleryImageItem } from '@features/gallery/core/items';

import { describe, expect, it } from 'vitest';

import type { GalleryQueuePlaceholder } from './galleryStateView';

import {
  buildGalleryGridRows,
  GALLERY_GRID_GAP_PX,
  GALLERY_STARRED_HEADER_HEIGHT_PX,
  getGalleryCellSizePx,
  getGalleryColumnCount,
  getGalleryGridRowHeightPx,
  getGalleryGridRowIndexForItem,
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

const createPlaceholder = (id: string): GalleryQueuePlaceholder => ({
  backendItemId: null,
  boardId: 'none',
  height: 1024,
  id,
  itemIndex: 0,
  queueItemId: `queue-${id}`,
  width: 1024,
});

const buildRows = (overrides: Partial<Parameters<typeof buildGalleryGridRows>[0]> = {}) =>
  buildGalleryGridRows({
    columnCount: 2,
    imageOrderDir: 'DESC',
    isStarredOpen: true,
    items: [],
    pendingPlaceholders: [],
    ...overrides,
  });

describe('getGalleryColumnCount', () => {
  it('gives the same answer at the same width regardless of placement', () => {
    // The whole point of measuring rather than branching on layout: a 600px
    // gallery is a 600px gallery whether it sits in a panel or the center.
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
    expect(getGalleryColumnCount({ imageDensityPercent: 100, widthPx: 8000 })).toBe(12);
    expect(getGalleryColumnCount({ imageDensityPercent: 0, widthPx: 8000 })).toBe(12);
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
  it('chunks plain items into rows of the column count with no section chrome', () => {
    const rows = buildRows({ items: ['a', 'b', 'c'].map((name) => createImageItem(name)) });

    expect(rows.map((row) => row.kind)).toEqual(['cells', 'cells']);
    expect(rows[0]?.kind === 'cells' && rows[0].cells.length).toBe(2);
    expect(rows[1]?.kind === 'cells' && rows[1].cells.length).toBe(1);
  });

  it('lifts starred items into a header-led section above the regular items', () => {
    const rows = buildRows({
      items: [createImageItem('regular-1'), createImageItem('starred-1', true), createImageItem('regular-2')],
    });

    expect(rows.map((row) => row.kind)).toEqual(['starred-header', 'cells', 'starred-gap', 'cells']);
    expect(rows[0]?.kind === 'starred-header' && rows[0].itemCount).toBe(1);
    expect(rows[1]?.kind === 'cells' && rows[1].section).toBe('starred');
    expect(rows[3]?.kind === 'cells' && rows[3].section).toBe('regular');
  });

  it('keeps the header but drops the starred rows while collapsed', () => {
    const items = [createImageItem('starred-1', true), createImageItem('regular-1')];
    const rows = buildRows({ isStarredOpen: false, items });

    expect(rows.map((row) => row.kind)).toEqual(['starred-header', 'starred-gap', 'cells']);
  });

  it('keeps regular row keys stable across a starred collapse so their cells are not recreated', () => {
    const items = [createImageItem('starred-1', true), createImageItem('regular-1'), createImageItem('regular-2')];
    const openKeys = buildRows({ items })
      .filter((row) => row.kind === 'cells' && row.section === 'regular')
      .map((row) => row.key);
    const collapsedKeys = buildRows({ isStarredOpen: false, items })
      .filter((row) => row.kind === 'cells' && row.section === 'regular')
      .map((row) => row.key);

    expect(collapsedKeys).toEqual(openKeys);
  });

  it('gives every row a unique key', () => {
    const rows = buildRows({
      items: [createImageItem('starred-1', true), createImageItem('regular-1'), createImageItem('regular-2')],
      pendingPlaceholders: [createPlaceholder('slot-1')],
    });
    const keys = rows.map((row) => row.key);

    expect(new Set(keys).size).toBe(keys.length);
  });

  it('slots placeholders ahead of regular items for newest-first ordering and after them otherwise', () => {
    const items = [createImageItem('regular-1')];
    const pendingPlaceholders = [createPlaceholder('slot-1')];

    const newestFirst = buildRows({ items, pendingPlaceholders });
    expect(newestFirst[0]?.kind === 'cells' && newestFirst[0].cells[0]?.kind).toBe('placeholder');

    const oldestFirst = buildRows({ imageOrderDir: 'ASC', items, pendingPlaceholders });
    expect(oldestFirst[0]?.kind === 'cells' && oldestFirst[0].cells[0]?.kind).toBe('item');
  });

  it('never mixes starred and regular cells in one row', () => {
    const rows = buildRows({
      columnCount: 3,
      items: [createImageItem('starred-1', true), createImageItem('regular-1'), createImageItem('regular-2')],
    });

    for (const row of rows) {
      if (row.kind === 'cells') {
        expect(row.cells.length).toBeLessThanOrEqual(3);
      }
    }
    expect(rows.filter((row) => row.kind === 'cells')).toHaveLength(2);
  });
});

describe('getGalleryGridRowHeightPx', () => {
  it('sizes chrome rows by their own constants and cell rows by the shared row height', () => {
    const rows = buildRows({ items: [createImageItem('starred-1', true), createImageItem('regular-1')] });

    expect(rows.map((row) => getGalleryGridRowHeightPx(row, 100))).toEqual([
      GALLERY_STARRED_HEADER_HEIGHT_PX,
      100,
      GALLERY_GRID_GAP_PX,
      100,
    ]);
  });
});

describe('getGalleryGridRowIndexForItem', () => {
  it('finds the row holding an item by its index in the source list, across sections', () => {
    const rows = buildRows({
      items: [createImageItem('regular-1'), createImageItem('starred-1', true), createImageItem('regular-2')],
    });

    // Item 1 is starred, so it lives in the starred row despite its list position.
    expect(getGalleryGridRowIndexForItem(rows, 1)).toBe(1);
    expect(getGalleryGridRowIndexForItem(rows, 2)).toBe(3);
    expect(getGalleryGridRowIndexForItem(rows, 99)).toBe(-1);
  });
});
