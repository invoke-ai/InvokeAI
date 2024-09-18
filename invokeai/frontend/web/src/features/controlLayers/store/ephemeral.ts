import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { atom, computed } from 'nanostores';

// Ephemeral state for canvas - not persisted across sessions.

/**
 * The global canvas manager instance.
 */
export const $canvasManager = atom<CanvasManager | null>(null);

/**
 * The index of the active tab in the canvas right panel.
 */
export const $canvasRightPanelTabIndex = atom(0);
/**
 * The name of the active tab in the canvas right panel.
 */
export const $canvasRightPanelTab = computed($canvasRightPanelTabIndex, (index) =>
  index === 0 ? 'layers' : 'gallery'
);
export const selectCanvasRightPanelLayersTab = () => $canvasRightPanelTabIndex.set(0);
export const selectCanvasRightPanelGalleryTab = () => $canvasRightPanelTabIndex.set(1);
