import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { atom } from 'nanostores';

// Ephemeral state for canvas - not persisted across sessions.

/**
 * The global canvas manager instance.
 */
export const $canvasManager = atom<CanvasManager | null>(null);
