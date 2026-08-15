import { useCallback, useRef, useState } from 'react';

import type { ProjectsViewId } from './projectLibraryView';

/**
 * The grid's measurements, as numbers rather than CSS breakpoints.
 *
 * The virtualizer needs both before anything is laid out: how many cards share
 * a row, and how tall a row is. Measuring the container rather than the window
 * keeps both honest when the panel is resized instead of the browser.
 */

const MIN_CARD_WIDTH_PX = 240;
const MAX_COLUMNS = 4;

/** `gap="4"` between cards and rows. */
export const PROJECTS_GRID_GAP_PX = 16;

/** `p="3"` around a name line and a timestamp line. */
export const PROJECT_CARD_META_HEIGHT_PX = 60;

/** List rows are contiguous — each fills its whole 44px slot: a 35px cover (56px / 16:10) plus breathing room. Dividers were dropped with the softened row hover. */
export const PROJECT_ROW_HEIGHT_PX = 44;

export const PROJECT_GROUP_HEADER_HEIGHT_PX = 44;

export const getProjectsColumnCount = (widthPx: number): number => {
  if (widthPx <= 0) {
    return 1;
  }

  return Math.max(1, Math.min(MAX_COLUMNS, Math.floor(widthPx / MIN_CARD_WIDTH_PX)));
};

/**
 * A grid row is one card tall: a 16:10 cover over a fixed meta block, plus the
 * gap below it. Derived from the measured width because the cover's height
 * follows the column width.
 */
export const getProjectsGridRowHeight = ({
  aspectRatio,
  columnCount,
  widthPx,
}: {
  aspectRatio: number;
  columnCount: number;
  widthPx: number;
}): number => {
  const columns = Math.max(1, columnCount);
  const columnWidth = (widthPx - PROJECTS_GRID_GAP_PX * (columns - 1)) / columns;

  if (!Number.isFinite(columnWidth) || columnWidth <= 0) {
    return PROJECT_CARD_META_HEIGHT_PX + PROJECTS_GRID_GAP_PX;
  }

  return Math.round(columnWidth / aspectRatio + PROJECT_CARD_META_HEIGHT_PX + PROJECTS_GRID_GAP_PX);
};

export const useProjectsMetrics = (
  view: ProjectsViewId
): { columnCount: number; measureRef: (node: HTMLDivElement | null) => void; widthPx: number } => {
  const [widthPx, setWidthPx] = useState(0);
  const observerRef = useRef<ResizeObserver | null>(null);

  const measureRef = useCallback((node: HTMLDivElement | null) => {
    observerRef.current?.disconnect();
    observerRef.current = null;

    if (!node) {
      return;
    }

    setWidthPx(node.clientWidth);

    if (typeof ResizeObserver === 'undefined') {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];

      if (entry) {
        setWidthPx(entry.contentRect.width);
      }
    });

    observer.observe(node);
    observerRef.current = observer;
  }, []);

  return { columnCount: view === 'list' ? 1 : getProjectsColumnCount(widthPx), measureRef, widthPx };
};
