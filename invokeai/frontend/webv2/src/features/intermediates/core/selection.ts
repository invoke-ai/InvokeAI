import type { IntermediatesRow, IntermediatesScope, IntermediatesScopeTarget } from './types';

import { getIntermediatesRowKey } from './types';

/**
 * Row selection for the manager. `rows` keeps a snapshot of every picked row, so picks survive paging and can be
 * summarized and targeted without the page that showed them; `all-matching` stands for every row the current
 * filters match, including rows never loaded, minus the excluded rows, whose snapshots keep the estimate honest. The
 * server resolves what "all matching" is when a preview is requested.
 */
export type IntermediatesSelection =
  | { mode: 'rows'; rows: ReadonlyMap<string, IntermediatesRow> }
  | { mode: 'all-matching'; excluded: ReadonlyMap<string, IntermediatesRow> };

export const EMPTY_SELECTION: IntermediatesSelection = { mode: 'rows', rows: new Map() };

export const isRowSelected = (selection: IntermediatesSelection, row: IntermediatesRow): boolean =>
  selection.mode === 'all-matching'
    ? !selection.excluded.has(getIntermediatesRowKey(row))
    : selection.rows.has(getIntermediatesRowKey(row));

export const withRowSelected = (selection: IntermediatesSelection, row: IntermediatesRow): IntermediatesSelection => {
  return isRowSelected(selection, row) ? selection : toggleRowSelection(selection, row);
};

export const toggleRowSelection = (
  selection: IntermediatesSelection,
  row: IntermediatesRow
): IntermediatesSelection => {
  const key = getIntermediatesRowKey(row);
  if (selection.mode === 'all-matching') {
    const excluded = new Map(selection.excluded);
    if (excluded.has(key)) {
      excluded.delete(key);
    } else {
      excluded.set(key, row);
    }
    return { mode: 'all-matching', excluded };
  }
  const rows = new Map(selection.rows);
  if (rows.has(key)) {
    rows.delete(key);
  } else {
    rows.set(key, row);
  }
  return { mode: 'rows', rows };
};

export const selectAllMatching = (): IntermediatesSelection => ({ mode: 'all-matching', excluded: new Map() });

export interface SelectionSummary {
  rows: number;
  safeImages: number;
  safeVideos: number;
  reclaimableBytes: number;
  unknownSizeCount: number;
}

export type SelectionTotals = SelectionSummary;

const sumRows = (rows: Iterable<IntermediatesRow>): SelectionSummary => {
  const summary: SelectionSummary = { reclaimableBytes: 0, rows: 0, safeImages: 0, safeVideos: 0, unknownSizeCount: 0 };
  for (const row of rows) {
    summary.rows += 1;
    summary.safeImages += row.images.safe;
    summary.safeVideos += row.videos.safe;
    summary.reclaimableBytes += row.reclaimableBytes;
    summary.unknownSizeCount += row.unknownSizeCount;
  }
  return summary;
};

/**
 * Sums the selection. Explicit picks are summed from their snapshots, whichever page showed them; "all matching"
 * starts from the server's totals and subtracts the excluded rows, never going below what is visibly selected (an
 * excluded row may have vanished off-page since its snapshot). A row on the current page is read fresh, since its
 * snapshot may predate a cleanup; the server preview stays the authority for what a delete does.
 */
export const summarizeSelection = (
  selection: IntermediatesSelection,
  totals: SelectionTotals,
  visibleRows: readonly IntermediatesRow[] = []
): SelectionSummary => {
  const freshRows = new Map(visibleRows.map((row) => [getIntermediatesRowKey(row), row]));
  const fresh = (key: string, snapshot: IntermediatesRow): IntermediatesRow => freshRows.get(key) ?? snapshot;
  if (selection.mode === 'rows') {
    return sumRows([...selection.rows].map(([key, snapshot]) => fresh(key, snapshot)));
  }
  const excluded = sumRows([...selection.excluded].map(([key, snapshot]) => fresh(key, snapshot)));
  const visible = sumRows(visibleRows.filter((row) => isRowSelected(selection, row)));
  const remaining = (field: keyof SelectionSummary): number =>
    Math.max(totals[field] - excluded[field], visible[field]);
  return {
    reclaimableBytes: remaining('reclaimableBytes'),
    rows: remaining('rows'),
    safeImages: remaining('safeImages'),
    safeVideos: remaining('safeVideos'),
    unknownSizeCount: remaining('unknownSizeCount'),
  };
};

const toTarget = ({ projectId, userId }: IntermediatesScopeTarget): IntermediatesScopeTarget => ({ projectId, userId });

/** The summary filters a selection was made under; a `matching` scope carries them to the server. */
export interface IntermediatesScopeFilters {
  ownerId: string | null;
  projectId: string | null;
  search: string;
}

/**
 * What a confirmation acts on. Explicit picks become their targets. Without exclusions, a project filter or a
 * search, "all matching" is exactly the owner filter (or everyone); otherwise the server resolves the filters.
 */
export const resolveScope = ({
  ownerId,
  projectId,
  search,
  selection,
}: IntermediatesScopeFilters & { selection: IntermediatesSelection }): IntermediatesScope => {
  if (selection.mode === 'rows') {
    return { kind: 'selection', targets: [...selection.rows.values()].map(toTarget) };
  }
  const trimmed = search.trim();
  if (selection.excluded.size === 0 && projectId === null && !trimmed) {
    return ownerId === null ? { kind: 'everyone' } : { kind: 'owner', userId: ownerId };
  }
  return {
    excluded: [...selection.excluded.values()].map(toTarget),
    kind: 'matching',
    projectId,
    search: trimmed || null,
    userId: ownerId,
  };
};
