/**
 * A structural history entry: a pair of reducer actions (forward + inverse)
 * dispatched to undo/redo a document-shape change.
 *
 * Unlike a pixel {@link createImagePatchEntry | image patch}, a document patch
 * carries no bitmaps — just two {@link CanvasProjectMutation}s. Undo dispatches the
 * `inverse`, redo dispatches the `forward`. It exists so structural canvas edits
 * (add/remove/reorder layer, rename, …) can live on the same engine-owned undo
 * stack as paint edits. Phase 3's layers-panel operations lean on this; P2.1 uses
 * it lightly (if at all) for composing an auto-created layer into a stroke's undo.
 *
 * Byte cost is tiny (actions are small plain objects); the default keeps it off
 * the byte budget's radar while still counting as one of the entry-count slots.
 *
 * Zero React, zero DOM, zero import-time side effects.
 */

import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import type { HistoryEntry } from './history';

/** Nominal byte cost for a structural entry (small; actions are plain objects). */
export const DOCUMENT_PATCH_DEFAULT_BYTES = 256;

/** Options for {@link createDocumentPatchEntry}. */
export interface CreateDocumentPatchEntryOptions {
  label: string;
  /** The action that performs the change (dispatched on redo). */
  forward: CanvasProjectMutation;
  /** The action that reverses the change (dispatched on undo). */
  inverse: CanvasProjectMutation;
  /** The reducer bridge. */
  dispatch(action: CanvasProjectMutation): void;
  /** Approximate retained size (default {@link DOCUMENT_PATCH_DEFAULT_BYTES}). */
  bytes?: number;
}

/** Creates a reversible structural entry that dispatches inverse on undo, forward on redo. */
export const createDocumentPatchEntry = (opts: CreateDocumentPatchEntryOptions): HistoryEntry => {
  const { dispatch, forward, inverse, label } = opts;
  const bytes = opts.bytes ?? DOCUMENT_PATCH_DEFAULT_BYTES;

  return {
    bytes,
    label,
    redo: () => dispatch(forward),
    undo: () => dispatch(inverse),
  };
};
