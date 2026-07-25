/**
 * How a selection gesture's boolean op is resolved.
 *
 * Every selection tool (lasso, marquee) offers the same choice: a persistent op
 * mode from its options bar, transiently overridden by modifiers held at commit
 * time. Keeping the resolution here means the tools agree by construction rather
 * than by convention, and neither has to import the other.
 *
 * Zero React, zero import-time side effects.
 */

import type { PointerModifiers, SelectionOp } from '@workbench/canvas-engine/types';

/**
 * Resolves the boolean op for a commit: held modifiers win over the persistent
 * `mode`. shift = add, alt = subtract, shift+alt = intersect.
 */
export const selectionOpFor = (modifiers: Pick<PointerModifiers, 'shift' | 'alt'>, mode: SelectionOp): SelectionOp => {
  if (modifiers.shift && modifiers.alt) {
    return 'intersect';
  }
  if (modifiers.shift) {
    return 'add';
  }
  if (modifiers.alt) {
    return 'subtract';
  }
  return mode;
};
