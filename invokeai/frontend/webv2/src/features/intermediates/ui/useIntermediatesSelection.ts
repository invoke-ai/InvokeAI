import type { IntermediatesRow, IntermediatesSummary } from '@features/intermediates/core/types';

import {
  EMPTY_SELECTION,
  selectAllMatching,
  summarizeSelection,
  toggleRowSelection,
  withRowSelected,
  type IntermediatesSelection,
} from '@features/intermediates/core/selection';
import { useCallback, useMemo, useState } from 'react';

const EMPTY_TOTALS = { reclaimableBytes: 0, rows: 0, safeImages: 0, safeVideos: 0, unknownSizeCount: 0 };

/**
 * Row selection across pages. An entry point's project stays selected until the user changes the selection; "all
 * matching" is estimated from the server's totals minus the excluded rows, and resolved by the server on preview.
 */
export const useIntermediatesSelection = ({
  initialProjectId,
  rows,
  summary,
}: {
  initialProjectId: string | null;
  rows: readonly IntermediatesRow[];
  summary: IntermediatesSummary | undefined;
}) => {
  const [selection, setSelection] = useState<IntermediatesSelection>(EMPTY_SELECTION);
  // The manager filters to the focused project, so pagination cannot hide the entry point's selection.
  const [pendingProjectId, setPendingProjectId] = useState<string | null>(initialProjectId);
  const totals = summary?.totals ?? EMPTY_TOTALS;
  const focusedRow = pendingProjectId ? rows.find((row) => row.projectId === pendingProjectId) : undefined;
  const effectiveSelection = focusedRow ? withRowSelected(selection, focusedRow) : selection;
  const selectionSummary = useMemo(
    () => summarizeSelection(effectiveSelection, totals, rows),
    [effectiveSelection, rows, totals]
  );
  const hasSelection = selectionSummary.rows > 0;
  const isComplete = hasSelection && selectionSummary.rows === totals.rows;

  const reset = useCallback(() => {
    setPendingProjectId(null);
    setSelection(EMPTY_SELECTION);
  }, []);
  const toggleRow = useCallback(
    (row: IntermediatesRow) => {
      setPendingProjectId(null);
      setSelection(toggleRowSelection(effectiveSelection, row));
    },
    [effectiveSelection]
  );
  // Select all selects every matching row, including pages not loaded; only a complete selection clears.
  const toggleAll = useCallback(() => {
    setPendingProjectId(null);
    setSelection(isComplete ? EMPTY_SELECTION : selectAllMatching());
  }, [isComplete]);

  return {
    effectiveSelection,
    hasSelection,
    reset,
    selectionState: hasSelection ? (isComplete ? ('all' as const) : ('some' as const)) : ('none' as const),
    selectionSummary,
    toggleAll,
    toggleRow,
  };
};
