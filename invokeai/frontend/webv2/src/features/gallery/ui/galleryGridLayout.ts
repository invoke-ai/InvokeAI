import type { GalleryItem, GalleryItemKey } from '@features/gallery/core/items';
import type { GalleryNavigationEntry } from '@features/gallery/core/selection';

import { toGalleryItemKey } from '@features/gallery/core/items';
import { GALLERY_PAGE_SIZE } from '@features/gallery/core/paging';

/** A null total means the listing size is not known yet. */
export const planGalleryPageOffsets = ({
  endIndexExclusive,
  startIndex,
  total,
}: {
  endIndexExclusive: number;
  startIndex: number;
  total: number | null;
}): number[] => {
  if (!Number.isFinite(startIndex) || !Number.isFinite(endIndexExclusive)) {
    return [];
  }

  let firstIndex = Math.max(0, Math.floor(startIndex));
  let afterLastIndex = Math.max(0, Math.ceil(endIndexExclusive));

  if (total !== null) {
    const boundedTotal = Math.max(0, Math.floor(total));

    firstIndex = Math.min(firstIndex, boundedTotal);
    afterLastIndex = Math.min(afterLastIndex, boundedTotal);
  }

  if (afterLastIndex <= firstIndex) {
    return [];
  }

  const firstOffset = Math.floor(firstIndex / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE;
  const lastOffset = Math.floor((afterLastIndex - 1) / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE;
  const offsets: number[] = [];

  for (let offset = firstOffset; offset <= lastOffset; offset += GALLERY_PAGE_SIZE) {
    offsets.push(offset);
  }

  return offsets;
};

export const GALLERY_GRID_GAP_PX = 4;
export const GALLERY_STARRED_HEADER_HEIGHT_PX = 24;
export const GALLERY_PINNED_FOOTER_PX = 9;

const GALLERY_MIN_COLUMN_COUNT = 2;
/** High enough that minimum cell size governs wide layouts at every density. */
const GALLERY_MAX_COLUMN_COUNT = 48;

/** 0% density is largest, 100% smallest. */
const GALLERY_MAX_CELL_PX = 192;
const GALLERY_MIN_CELL_PX = 48;

export const getGalleryTargetCellPx = (imageDensityPercent: number): number => {
  const percent = Math.min(100, Math.max(0, imageDensityPercent));

  return GALLERY_MAX_CELL_PX - ((GALLERY_MAX_CELL_PX - GALLERY_MIN_CELL_PX) * percent) / 100;
};

/** An unmeasured width yields `min`. */
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

export const getGalleryCellSizePx = ({ columnCount, widthPx }: { columnCount: number; widthPx: number }): number =>
  widthPx > 0 ? Math.max(1, (widthPx - GALLERY_GRID_GAP_PX * (columnCount - 1)) / columnCount) : 96;

export type GalleryGridSection = 'regular' | 'starred';

export type GalleryGridRow = { cells: GalleryItem[]; key: string; kind: 'cells'; section: GalleryGridSection };

const GALLERY_STARRED_STRIP_MAX_ROWS = 3;

export const getGalleryStarredStripItems = (starredItems: readonly GalleryItem[], columnCount: number): GalleryItem[] =>
  starredItems.slice(0, GALLERY_STARRED_STRIP_MAX_ROWS * columnCount);

/** Strip and listing refetch independently, so an item may briefly sit in both. */
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

/** Keyed by first cell so changes above preserve thumbnail DOM identity. */
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

export const buildGalleryGridRows = (items: readonly GalleryItem[], columnCount: number): GalleryGridRow[] =>
  chunkGalleryCellsIntoRows(items, columnCount, 'regular');

/** -1 for an item the listing does not hold (a strip item). */
export const getGalleryGridRowIndexForItemKey = (
  items: readonly GalleryItem[],
  itemKey: GalleryItemKey,
  columnCount: number
): number => {
  const index = items.findIndex((item) => toGalleryItemKey(item) === itemKey);

  return index < 0 ? -1 : Math.floor(index / columnCount);
};

/** Recent pseudo-rows are a prefix only for descending order. */
export const getGallerySparseRowIndexForItemKey = (
  itemSlots: ReadonlyMap<number, GalleryItem>,
  itemKey: GalleryItemKey,
  columnCount: number,
  leadingRows: number
): number => {
  for (const [index, item] of itemSlots) {
    if (toGalleryItemKey(item) === itemKey) {
      return leadingRows + Math.floor(index / columnCount);
    }
  }

  return -1;
};

export const getGallerySparseSelectionPages = ({
  itemSlots,
  pageOffset,
}: {
  itemSlots: ReadonlyMap<number, GalleryItem>;
  pageOffset: number;
}): ReadonlyMap<GalleryItemKey, number> => {
  const pages = new Map<GalleryItemKey, number>();

  for (const [itemIndex, item] of itemSlots) {
    pages.set(toGalleryItemKey(item), Math.floor((pageOffset + itemIndex) / GALLERY_PAGE_SIZE));
  }

  return pages;
};

export const getGallerySparseRowKey = (rowIndex: number): string => `listing-row:${rowIndex}`;
export const getGallerySparseSlotKey = (itemIndex: number): string => `listing-slot:${itemIndex}`;

/** Empty positions keep arrow-navigation geometry across hydration gaps; each subscribed run adds its own entries. */
export const buildSparseGalleryNavigationEntries = ({
  columnCount = 1,
  includeUnloadedBoundaries = true,
  itemSlots,
  pageOffsets,
  pendingPageOffsets,
  total,
}: {
  columnCount?: number;
  includeUnloadedBoundaries?: boolean;
  itemSlots: ReadonlyMap<number, GalleryItem>;
  pageOffsets: readonly number[];
  /** Their empty positions wait for data instead of being skipped as gaps. */
  pendingPageOffsets?: ReadonlySet<number>;
  total: number | null;
}): GalleryNavigationEntry[] => {
  if (pageOffsets.length === 0) {
    return [];
  }

  const sortedOffsets = [...new Set(pageOffsets)].sort((left, right) => left - right);
  const runs: { first: number; last: number }[] = [];

  for (const offset of sortedOffsets) {
    const run = runs.at(-1);

    if (run && offset === run.last + GALLERY_PAGE_SIZE) {
      run.last = offset;
    } else {
      runs.push({ first: offset, last: offset });
    }
  }

  const activePages = new Set(sortedOffsets);
  const entries: GalleryNavigationEntry[] = [];
  let nextIndex = 0;

  for (const [runIndex, run] of runs.entries()) {
    const activeStartIndex = Math.floor(run.first / columnCount) * columnCount;
    const startIndex =
      includeUnloadedBoundaries && run.first > 0 ? Math.max(0, activeStartIndex - columnCount) : activeStartIndex;
    const activeEndIndex = Math.min(total ?? Number.POSITIVE_INFINITY, run.last + GALLERY_PAGE_SIZE);
    const isFollowedByRun = runIndex < runs.length - 1;
    const endIndex =
      includeUnloadedBoundaries && total !== null
        ? Math.min(total, Math.ceil(activeEndIndex / columnCount) * columnCount + columnCount)
        : isFollowedByRun
          ? Math.ceil(activeEndIndex / columnCount) * columnCount
          : activeEndIndex;

    for (let index = Math.max(startIndex, nextIndex); index < endIndex; index += 1) {
      const item = itemSlots.get(index);
      const pageOffset = Math.floor(index / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE;
      const isSettledActivePage = activePages.has(pageOffset) && !pendingPageOffsets?.has(pageOffset);

      entries.push(
        item
          ? { item, kind: 'item' }
          : isSettledActivePage
            ? { id: `gallery-gap-slot:${index}`, kind: 'slot', navigable: false }
            : { id: `gallery-unloaded-slot:${index}`, kind: 'slot', navigable: true }
      );
    }

    nextIndex = Math.max(nextIndex, endIndex);
  }

  return entries;
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

/** A strip that failed with nothing to show keeps its header row to report the failure. */
export const getGalleryStarredLayout = ({
  columns,
  tileSize,
  shownCount,
  collapsed,
  failed = false,
}: {
  columns: number;
  tileSize: number;
  shownCount: number;
  collapsed: boolean;
  failed?: boolean;
}) => {
  const rowCount = Math.ceil(shownCount / columns);
  const rowHeight = tileSize + GALLERY_GRID_GAP_PX;

  return {
    rowCount,
    rowHeight,
    height:
      shownCount > 0
        ? GALLERY_STARRED_HEADER_HEIGHT_PX + (collapsed ? 0 : rowCount * rowHeight)
        : failed
          ? GALLERY_STARRED_HEADER_HEIGHT_PX
          : 0,
  };
};

/** Defines the virtualizer's scroll margin. */
export const getGalleryPinnedHeightPx = (progressHeight: number, starredHeight: number): number =>
  progressHeight + starredHeight > 0 ? progressHeight + starredHeight + GALLERY_PINNED_FOOTER_PX : 0;
