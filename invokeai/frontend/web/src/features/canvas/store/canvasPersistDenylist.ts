import { CanvasState } from './canvasTypes';

/**
 * Canvas slice persist denylist
 */
export const canvasPersistDenylist: (keyof CanvasState)[] = ['cursorPosition'];
