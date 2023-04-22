import { CanvasState } from './canvasTypes';

/**
 * Canvas slice persist blacklist
 */
const itemsToBlacklist: (keyof CanvasState)[] = [
  'cursorPosition',
  'isCanvasInitialized',
  'doesCanvasNeedScaling',
];

export const canvasBlacklist = itemsToBlacklist.map(
  (blacklistItem) => `canvas.${blacklistItem}`
);
