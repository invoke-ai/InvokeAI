import {
  WORKFLOW_LIBRARY_ENRICHMENT_DEMAND,
  WORKFLOW_LIBRARY_METADATA_WINDOW,
} from '@features/workflow/data/libraryBrowseStore';

import type { WorkflowLibraryCardSize } from './workflowThumbnail';

/**
 * Column count and virtualizer bounds for the library grid. The browse store retains at most 60 consecutive metadata
 * positions and enriches at most 48 demanded identities, visible cards first (`libraryBrowseStore.ts`); every
 * column count must keep the rows a capped viewport can show inside both.
 */

/** The tallest viewport, in estimated rows, at one to three columns. */
const MAX_VIEWPORT_ROWS = 15;

export const MAX_GRID_COLUMNS = 6;
/** Space between cards, horizontally and below each row (Chakra `3`). */
export const GRID_GAP_PX = 12;
/** The grid's end padding beside the scrollbar (Chakra `2`). */
export const GRID_END_PADDING_PX = 8;

const MIN_CARD_WIDTH: Record<WorkflowLibraryCardSize, number> = { large: 220, small: 150 };

/** As many columns as fit at the size's minimum card width, from one up to six. */
export const getWorkflowLibraryGridColumns = (viewportWidth: number, size: WorkflowLibraryCardSize): number =>
  Math.max(
    1,
    Math.min(
      MAX_GRID_COLUMNS,
      Math.floor((viewportWidth - GRID_END_PADDING_PX + GRID_GAP_PX) / (MIN_CARD_WIDTH[size] + GRID_GAP_PX))
    )
  );

export const getWorkflowLibraryCardWidth = (viewportWidth: number, columns: number): number =>
  Math.max(1, (viewportWidth - GRID_END_PADDING_PX - GRID_GAP_PX * (columns - 1)) / columns);

export interface WorkflowLibraryGridBounds {
  /** Rows mounted and demanded on either side of the viewport. */
  overscanRows: number;
  /** The viewport's height cap, in estimated rows; it can span one more, partially visible, row. */
  maxViewportRows: number;
}

/**
 * Up to four columns keep two overscan rows; five and six keep one so the viewport still spans several rows. With
 * `maxViewportRows` R and overscan O at C columns, (R + 1) × C visible cards fit the demand bound and
 * (R + 1 + 2O) × C positions fit the metadata window: six columns allow 7 rows (48 visible, 60 resident).
 */
export const getWorkflowLibraryGridBounds = (columns: number): WorkflowLibraryGridBounds => {
  const overscanRows = columns >= 5 ? 1 : 2;
  const maxViewportRows = Math.min(
    MAX_VIEWPORT_ROWS,
    Math.floor(WORKFLOW_LIBRARY_ENRICHMENT_DEMAND / columns) - 1,
    Math.floor(WORKFLOW_LIBRARY_METADATA_WINDOW / columns) - 1 - 2 * overscanRows
  );

  return { maxViewportRows, overscanRows };
};
