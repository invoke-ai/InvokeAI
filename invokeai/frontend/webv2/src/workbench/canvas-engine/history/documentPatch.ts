/**
 * Structural history stores forward/inverse reducer actions without bitmaps, sharing the pixel undo stack at
 * nominal byte cost.
 */

import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';

import type { HistoryEntry } from './history';

import { collectHistoryMediaRefs, HISTORY_ENTRY_OVERHEAD_BYTES } from './history';

/** Options for {@link createDocumentPatchEntry}. */
export interface CreateDocumentPatchEntryOptions {
  label: string;
  /** The action that performs the change (dispatched on redo). */
  forward: CanvasProjectMutation;
  /** The action that reverses the change (dispatched on undo). */
  inverse: CanvasProjectMutation;
  /** Applies one side of the patch; throws without applying to leave the entry in place. */
  dispatch(action: CanvasProjectMutation): void;
  /** Approximate retained size (default {@link HISTORY_ENTRY_OVERHEAD_BYTES}). */
  bytes?: number;
}

/** Creates a reversible structural entry that dispatches inverse on undo, forward on redo. */
export const createDocumentPatchEntry = (opts: CreateDocumentPatchEntryOptions): HistoryEntry => {
  const { dispatch, forward, inverse, label } = opts;
  const bytes = opts.bytes ?? HISTORY_ENTRY_OVERHEAD_BYTES;

  return {
    bytes,
    heldAssetRefs: collectHistoryMediaRefs(forward, inverse),
    label,
    redo: () => dispatch(forward),
    undo: () => dispatch(inverse),
  };
};
