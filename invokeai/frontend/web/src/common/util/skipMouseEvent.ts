import type { MouseEvent } from 'react';

/**
 * Prevents the default behavior of the event.
 */
export const skipMouseEvent = (e: MouseEvent) => {
  e.preventDefault();
};
