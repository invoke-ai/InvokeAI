import type { GalleryItem } from '@features/gallery/core/items';
import type { GalleryOrderDir } from '@features/gallery/core/types';

import { toGalleryItemKey } from '@features/gallery/core/items';

import type { GalleryQueuePlaceholder } from './galleryStateView';

import { getGalleryPlaceholderInsertionIndex } from './galleryStateView';

export const GALLERY_GRID_GAP_PX = 4;
export const GALLERY_STARRED_HEADER_HEIGHT_PX = 24;

const GALLERY_MIN_COLUMN_COUNT = 2;
const GALLERY_MAX_COLUMN_COUNT = 12;

/** Cell size the density slider interpolates between: 0% is largest, 100% smallest. */
const GALLERY_MAX_CELL_PX = 192;
const GALLERY_MIN_CELL_PX = 48;

/**
 * Density picks a target thumbnail size, and the available width decides how
 * many of those fit.
 *
 * Keying off width rather than placement is what lets the two layouts share
 * one grid: at equal pixel width the same density produces the same columns
 * whether the gallery sits in a side panel or the center. The old
 * layout-keyed maximum meant 50% density meant ~90px cells stacked and ~200px
 * cells wide.
 */
export const getGalleryTargetCellPx = (imageDensityPercent: number): number => {
  const percent = Math.min(100, Math.max(0, imageDensityPercent));

  return GALLERY_MAX_CELL_PX - ((GALLERY_MAX_CELL_PX - GALLERY_MIN_CELL_PX) * percent) / 100;
};

export const getGalleryColumnCount = ({
  imageDensityPercent,
  widthPx,
}: {
  imageDensityPercent: number;
  widthPx: number;
}): number => {
  if (widthPx <= 0) {
    return GALLERY_MIN_COLUMN_COUNT;
  }

  const columnCount = Math.round(widthPx / getGalleryTargetCellPx(imageDensityPercent));

  return Math.min(GALLERY_MAX_COLUMN_COUNT, Math.max(GALLERY_MIN_COLUMN_COUNT, columnCount));
};

/** Falls back to a plausible square before the viewport has been measured. */
export const getGalleryCellSizePx = ({ columnCount, widthPx }: { columnCount: number; widthPx: number }): number =>
  widthPx > 0 ? Math.max(1, (widthPx - GALLERY_GRID_GAP_PX * (columnCount - 1)) / columnCount) : 96;

export type GalleryGridCell =
  | { kind: 'item'; item: GalleryItem; itemIndex: number }
  | { kind: 'placeholder'; placeholder: GalleryQueuePlaceholder };

export type GalleryGridSection = 'regular' | 'starred';

export type GalleryGridRow =
  | { cells: GalleryGridCell[]; key: string; kind: 'cells'; section: GalleryGridSection }
  | { key: string; kind: 'starred-gap' }
  | { itemCount: number; key: string; kind: 'starred-header' };

const getGalleryGridCellKey = (cell: GalleryGridCell): string =>
  cell.kind === 'placeholder' ? `placeholder:${cell.placeholder.id}` : toGalleryItemKey(cell.item);

/**
 * Rows are keyed by their leading cell rather than their index so that
 * structural changes above a row (collapsing the starred section, a
 * placeholder resolving) move the row without recreating it: the virtualizer
 * and React both track the row by key, so its thumbnails keep their DOM.
 */
const chunkGalleryCellsIntoRows = (
  cells: GalleryGridCell[],
  columnCount: number,
  section: GalleryGridSection
): GalleryGridRow[] => {
  const rows: GalleryGridRow[] = [];

  for (let index = 0; index < cells.length; index += columnCount) {
    const rowCells = cells.slice(index, index + columnCount);

    rows.push({
      cells: rowCells,
      key: `${section}:${getGalleryGridCellKey(rowCells[0]!)}`,
      kind: 'cells',
      section,
    });
  }

  return rows;
};

/**
 * The grid's whole row model in one pure pass: starred items become a
 * disclosure section above the regular items, and pending queue placeholders
 * slot into the regular section at the position their images will land.
 */
export const buildGalleryGridRows = ({
  columnCount,
  imageOrderDir,
  isStarredOpen,
  items,
  pendingPlaceholders,
}: {
  columnCount: number;
  imageOrderDir: GalleryOrderDir;
  isStarredOpen: boolean;
  items: GalleryItem[];
  pendingPlaceholders: GalleryQueuePlaceholder[];
}): GalleryGridRow[] => {
  const regularItemCells: GalleryGridCell[] = [];
  const regularItems: GalleryItem[] = [];
  const starredItemCells: GalleryGridCell[] = [];

  items.forEach((item, itemIndex) => {
    const cell: GalleryGridCell = { item, itemIndex, kind: 'item' };

    if (item.starred) {
      starredItemCells.push(cell);
    } else {
      regularItemCells.push(cell);
      regularItems.push(item);
    }
  });

  const placeholderCells: GalleryGridCell[] = pendingPlaceholders.map((placeholder) => ({
    kind: 'placeholder',
    placeholder,
  }));
  const placeholderInsertionIndex = getGalleryPlaceholderInsertionIndex(regularItems, imageOrderDir, false);
  const regularCells = [
    ...regularItemCells.slice(0, placeholderInsertionIndex),
    ...placeholderCells,
    ...regularItemCells.slice(placeholderInsertionIndex),
  ];
  const rows: GalleryGridRow[] = [];

  if (starredItemCells.length > 0) {
    rows.push({ itemCount: starredItemCells.length, key: 'starred-header', kind: 'starred-header' });

    if (isStarredOpen) {
      rows.push(...chunkGalleryCellsIntoRows(starredItemCells, columnCount, 'starred'));
    }

    if (regularCells.length > 0) {
      rows.push({ key: 'starred-gap', kind: 'starred-gap' });
    }
  }

  rows.push(...chunkGalleryCellsIntoRows(regularCells, columnCount, 'regular'));

  return rows;
};

/** `cellRowHeightPx` is the thumbnail row height including its trailing gap. */
export const getGalleryGridRowHeightPx = (row: GalleryGridRow, cellRowHeightPx: number): number => {
  if (row.kind === 'starred-header') {
    return GALLERY_STARRED_HEADER_HEIGHT_PX;
  }

  if (row.kind === 'starred-gap') {
    return GALLERY_GRID_GAP_PX;
  }

  return cellRowHeightPx;
};

export const getGalleryGridRowIndexForItem = (rows: GalleryGridRow[], itemIndex: number): number =>
  rows.findIndex(
    (row) => row.kind === 'cells' && row.cells.some((cell) => cell.kind === 'item' && cell.itemIndex === itemIndex)
  );
