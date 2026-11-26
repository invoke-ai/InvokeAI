import type { MouseEvent } from 'react';

/**
 * A context menu opened with the keyboard (the Context Menu key, Shift+F10) has no pointer position of its own: Chromium
 * reports button -1 at a point of its own choosing, other engines no pointer type or 0,0. A heuristic: engines vary.
 */
export const isKeyboardContextMenu = (event: MouseEvent<Element>): boolean => {
  const native = event.nativeEvent as Partial<PointerEvent>;

  return event.button === -1 || native.pointerType === '' || (event.clientX === 0 && event.clientY === 0);
};

const clamp = (value: number, max: number) => Math.min(Math.max(value, 0), max);

/** Below the left edge of `element`, clamped to the viewport, as a tall element can run off it. */
export const getContextMenuPointBelow = (element: Element): { x: number; y: number } => {
  const rect = element.getBoundingClientRect();

  return { x: clamp(rect.left, window.innerWidth), y: clamp(rect.bottom, window.innerHeight) };
};

/**
 * Where a context menu opens: at the pointer, or from the keyboard below the focused element inside the handler's own,
 * else below the handler's.
 */
export const getContextMenuPoint = (event: MouseEvent<Element>): { x: number; y: number } => {
  if (!isKeyboardContextMenu(event)) {
    return { x: event.clientX, y: event.clientY };
  }

  const focused = document.activeElement;

  return getContextMenuPointBelow(focused && event.currentTarget.contains(focused) ? focused : event.currentTarget);
};
