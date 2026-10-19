/**
 * Shared selection-op resolution: commit-time modifiers override persistent mode consistently across lasso and
 * marquee.
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
