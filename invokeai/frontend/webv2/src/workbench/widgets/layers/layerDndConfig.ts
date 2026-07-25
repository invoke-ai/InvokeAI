import type { KeyboardSensorOptions } from '@dnd-kit/core';

import { sortableKeyboardCoordinates } from '@dnd-kit/sortable';

export const LAYER_KEYBOARD_SENSOR_OPTIONS = {
  coordinateGetter: sortableKeyboardCoordinates,
  keyboardCodes: {
    cancel: ['Escape'],
    end: ['Enter'],
    start: ['Enter'],
  },
} satisfies KeyboardSensorOptions;

/** The shape of a keyboard event this guard inspects (DOM or React synthetic). */
export interface LayerDragKeyEvent {
  target: EventTarget | null;
  altKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  shiftKey: boolean;
}

/**
 * Whether a key event may start a keyboard drag of a layer row.
 *
 * dnd-kit claims `Enter` to start a drag and inspects only `event.code`, and the
 * whole row is the drag handle (no separate activator node, so dnd-kit's own
 * `event.target !== activator` guard never engages). That let two unrelated
 * Enter presses start a drag nobody asked for:
 *
 * - **A MODIFIED Enter.** `mod+Enter` is Invoke. Pressing it with a row focused
 *   also began an invisible keyboard drag, leaving the row stuck at drag opacity
 *   — which reads as the layer having been disabled, though nothing about the
 *   layer changed.
 * - **An Enter from a PORTALLED descendant.** React events bubble through the
 *   component tree, not the DOM, so keystrokes in the layer-properties popover
 *   (portalled to the body, and holding multi-line prompt textareas where Enter
 *   is a newline) reached the row. The popover defended itself by stopping
 *   propagation wholesale, which also stopped the event reaching the window —
 *   killing every global hotkey inside it, including the prompt-attention keys.
 *
 * Guarding the activator fixes both at the source, so no portalled panel has to
 * remember to stop propagation — and none of them has to break global hotkeys to
 * do it.
 */
export const shouldStartLayerKeyboardDrag = (event: LayerDragKeyEvent, rowNode: HTMLElement | null): boolean => {
  // An unmodified Enter is the drag gesture; every modified one belongs to a
  // command (Invoke, and anything a user rebinds onto mod+Enter).
  if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
    return false;
  }
  if (typeof HTMLElement === 'undefined' || !(event.target instanceof HTMLElement)) {
    return true;
  }
  // Typing Enter in a field is never a request to reorder layers.
  if (event.target.closest('input, textarea, select, [contenteditable="true"]')) {
    return false;
  }
  // Portal bleed: the event reached this row through the React tree, but its
  // DOM node lives elsewhere.
  return !rowNode || rowNode.contains(event.target);
};
