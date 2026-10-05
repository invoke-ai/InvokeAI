import type { GalleryItem, GalleryItemKey } from '@features/gallery/core/items';

import { toGalleryItemKey } from '@features/gallery/core/items';

export const GALLERY_GRID_GAP_PX = 4;
/** The disclosure row of a pinned section (in progress, starred). */
export const GALLERY_STARRED_HEADER_HEIGHT_PX = 24;
/** The pinned block's hairline rule plus the margin that separates it from the listing. */
export const GALLERY_PINNED_FOOTER_PX = 9;

const GALLERY_MIN_COLUMN_COUNT = 2;
/** Keep the column cap high enough that minimum cell size governs wide layouts at every density. */
const GALLERY_MAX_COLUMN_COUNT = 48;

/** Cell size the density slider interpolates between: 0% is largest, 100% smallest. */
const GALLERY_MAX_CELL_PX = 192;
const GALLERY_MIN_CELL_PX = 48;

/** Density selects target cell size; measured width determines columns consistently across placements. */
export const getGalleryTargetCellPx = (imageDensityPercent: number): number => {
  const percent = Math.min(100, Math.max(0, imageDensityPercent));

  return GALLERY_MAX_CELL_PX - ((GALLERY_MAX_CELL_PX - GALLERY_MIN_CELL_PX) * percent) / 100;
};

/** How many `targetCellPx` cells fit in `widthPx`, clamped; an unmeasured width yields `min`. */
export const getGalleryColumnCountForCell = ({
  max,
  min,
  targetCellPx,
  widthPx,
}: {
  max: number;
  min: number;
  targetCellPx: number;
  widthPx: number;
}): number => (widthPx <= 0 ? min : Math.min(max, Math.max(min, Math.round(widthPx / targetCellPx))));

export const getGalleryColumnCount = ({
  imageDensityPercent,
  widthPx,
}: {
  imageDensityPercent: number;
  widthPx: number;
}): number =>
  getGalleryColumnCountForCell({
    max: GALLERY_MAX_COLUMN_COUNT,
    min: GALLERY_MIN_COLUMN_COUNT,
    targetCellPx: getGalleryTargetCellPx(imageDensityPercent),
    widthPx,
  });

/** Falls back to a plausible square before the viewport has been measured. */
export const getGalleryCellSizePx = ({ columnCount, widthPx }: { columnCount: number; widthPx: number }): number =>
  widthPx > 0 ? Math.max(1, (widthPx - GALLERY_GRID_GAP_PX * (columnCount - 1)) / columnCount) : 96;

export type GalleryGridSection = 'regular' | 'starred';

export type GalleryGridRow = { cells: GalleryItem[]; key: string; kind: 'cells'; section: GalleryGridSection };

/** The starred strip shows at most this many rows at the current column count. */
const GALLERY_STARRED_STRIP_MAX_ROWS = 3;

export const getGalleryStarredStripItems = (starredItems: readonly GalleryItem[], columnCount: number): GalleryItem[] =>
  starredItems.slice(0, GALLERY_STARRED_STRIP_MAX_ROWS * columnCount);

/**
 * Every item the grid has on hand — strip first, then the listing. The two
 * refetch independently, so an item can sit on both sides for a moment.
 */
export const mergeGalleryLoadedItems = (
  starredItems: readonly GalleryItem[],
  items: readonly GalleryItem[]
): GalleryItem[] => {
  if (starredItems.length === 0) {
    return items as GalleryItem[];
  }

  const seen = new Set(starredItems.map(toGalleryItemKey));

  return [...starredItems, ...items.filter((item) => !seen.has(toGalleryItemKey(item)))];
};

/** Key rows by their first cell so changes above them preserve thumbnail DOM identity. */
export const chunkGalleryCellsIntoRows = (
  cells: readonly GalleryItem[],
  columnCount: number,
  section: GalleryGridSection
): GalleryGridRow[] => {
  const rows: GalleryGridRow[] = [];

  for (let index = 0; index < cells.length; index += columnCount) {
    const rowCells = cells.slice(index, index + columnCount);

    rows.push({
      cells: rowCells,
      key: `${section}:${toGalleryItemKey(rowCells[0]!)}`,
      kind: 'cells',
      section,
    });
  }

  return rows;
};

/** Only listing items form virtual rows; the starred strip remains pinned above them. */
export const buildGalleryGridRows = (items: readonly GalleryItem[], columnCount: number): GalleryGridRow[] =>
  chunkGalleryCellsIntoRows(items, columnCount, 'regular');

/** The listing row holding `itemKey`; -1 for an item the listing does not hold (a strip item). */
export const getGalleryGridRowIndexForItemKey = (
  items: readonly GalleryItem[],
  itemKey: GalleryItemKey,
  columnCount: number
): number => {
  const index = items.findIndex((item) => toGalleryItemKey(item) === itemKey);

  return index < 0 ? -1 : Math.floor(index / columnCount);
};

export const getGalleryProgressLayout = ({
  columns,
  tileSize,
  sessionCount,
  visible,
  collapsed,
}: {
  columns: number;
  tileSize: number;
  sessionCount: number;
  visible: boolean;
  collapsed: boolean;
}) => {
  const headerHeight = GALLERY_STARRED_HEADER_HEIGHT_PX;
  const paddingBottom = 8;
  const rowCount = Math.ceil(sessionCount / columns);
  const rowHeight = tileSize + GALLERY_GRID_GAP_PX;
  return {
    columns,
    tileSize,
    headerHeight,
    paddingBottom,
    rowCount,
    rowHeight,
    height: visible && sessionCount > 0 ? headerHeight + (collapsed ? 0 : rowCount * rowHeight + paddingBottom) : 0,
  };
};
export type GalleryProgressLayout = ReturnType<typeof getGalleryProgressLayout>;

/** The pinned starred strip: its disclosure row plus, while open, its bounded rows. */
export const getGalleryStarredLayout = ({
  columns,
  tileSize,
  shownCount,
  collapsed,
}: {
  columns: number;
  tileSize: number;
  shownCount: number;
  collapsed: boolean;
}) => {
  const rowCount = Math.ceil(shownCount / columns);
  const rowHeight = tileSize + GALLERY_GRID_GAP_PX;

  return {
    rowCount,
    rowHeight,
    height: shownCount > 0 ? GALLERY_STARRED_HEADER_HEIGHT_PX + (collapsed ? 0 : rowCount * rowHeight) : 0,
  };
};

/** The shared in-progress/starred block height defines the virtualizer's scroll margin. */
export const getGalleryPinnedHeightPx = (progressHeight: number, starredHeight: number): number =>
  progressHeight + starredHeight > 0 ? progressHeight + starredHeight + GALLERY_PINNED_FOOTER_PX : 0;
