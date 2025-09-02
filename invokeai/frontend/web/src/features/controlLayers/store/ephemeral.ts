import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { atom } from 'nanostores';

// Ephemeral state for canvas - not persisted across sessions.

/**
 * Registry of canvas manager instances keyed by canvasId.
 */
export const $canvasManagers = atom<Map<string, CanvasManager>>(new Map());
