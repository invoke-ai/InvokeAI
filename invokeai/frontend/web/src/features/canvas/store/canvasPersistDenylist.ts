import { CanvasState } from './canvasTypes';

/**
 * Canvas slice persist denylist
 */
const itemsToDenylist: (keyof CanvasState)[] = [
  'cursorPosition',
  'isCanvasInitialized',
  'doesCanvasNeedScaling',
];

export const canvasDenylist = itemsToDenylist.map(
  (denylistItem) => `canvas.${denylistItem}`
);
