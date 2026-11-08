/**
 * Snapshots selection-changing calls into individual engine-history steps. Unchanged selections record nothing;
 * replay restores snapshots directly without recording again. A change is refused while a replay runs, and one
 * whose step history could never keep is rolled back to the exact prior snapshot.
 */

import type { History } from '@workbench/canvas-engine/history/history';
import type { Rect, SelectionOp } from '@workbench/canvas-engine/types';

import { NO_HELD_ASSET_REFS } from '@workbench/canvas-engine/history/history';
import { isEmpty } from '@workbench/canvas-engine/math/rect';

import type { SelectionCommit, SelectionSnapshot, SelectionState } from './selectionState';

const COMMIT_LABELS: Record<SelectionOp, string> = {
  add: 'Add to selection',
  intersect: 'Intersect selection',
  replace: 'Select',
  subtract: 'Subtract from selection',
};

const sameRect = (left: Rect | null, right: Rect | null): boolean =>
  left === right ||
  (left !== null &&
    right !== null &&
    left.x === right.x &&
    left.y === right.y &&
    left.width === right.width &&
    left.height === right.height);

const sameAlpha = (left: Uint8ClampedArray | null, right: Uint8ClampedArray | null): boolean => {
  if (left === null || right === null) {
    return left === right;
  }
  if (left.length !== right.length) {
    return false;
  }
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) {
      return false;
    }
  }
  return true;
};

const sameSnapshot = (left: SelectionSnapshot, right: SelectionSnapshot): boolean =>
  left.selected === right.selected &&
  left.commits.length === right.commits.length &&
  sameRect(left.rect, right.rect) &&
  sameRect(left.bounds, right.bounds) &&
  sameAlpha(left.alpha, right.alpha);

/** A degenerate replace (a lasso too small to close) deselects rather than selects. */
const commitLabel = (commit: SelectionCommit): string =>
  commit.op === 'replace' && isEmpty(commit.bounds) ? 'Deselect' : COMMIT_LABELS[commit.op];

/** Wraps `selection` so its mutations record on `history`; reads and replay pass through untouched. */
export const withSelectionHistory = (
  selection: SelectionState,
  history: Pick<History, 'admit' | 'isReplaying'>
): SelectionState => {
  // Consecutive steps share their boundary capture; charge a plane once.
  let lastAfter: SelectionSnapshot | null = null;
  const record = (label: string, mutate: () => void): void => {
    if (history.isReplaying()) {
      return;
    }
    const before = selection.snapshot();
    mutate();
    const after = selection.snapshot();
    if (sameSnapshot(before, after)) {
      return;
    }
    const beforeBytes = before === lastAfter ? 0 : (before.alpha?.byteLength ?? 0);
    // The result's size is known only once computed; restoring the snapshot is exact.
    const admission = history.admit(beforeBytes + (after.alpha?.byteLength ?? 0));
    if (!admission) {
      selection.restore(before);
      return;
    }
    lastAfter = after;
    admission.publish({
      bytes: admission.bytes,
      heldAssetRefs: NO_HELD_ASSET_REFS,
      label,
      redo: () => selection.restore(after),
      undo: () => selection.restore(before),
    });
  };

  return {
    ...selection,
    clear: () => record('Deselect', () => selection.clear()),
    commit: (commit) => record(commitLabel(commit), () => selection.commit(commit)),
    invert: (domain) => record('Invert selection', () => selection.invert(domain)),
    replaceMask: (mask) => record('Select object', () => selection.replaceMask(mask)),
    selectAll: (domain) => record('Select all', () => selection.selectAll(domain)),
  };
};
