import { useCanvasContext, useCanvasContextSafe } from 'features/controlLayers/contexts/CanvasInstanceContext';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';

/**
 * Hook to get the canvas manager from the canvas instance context.
 * Throws if not within a CanvasInstanceProvider.
 */
export const useCanvasManager = (): CanvasManager => {
  const { manager } = useCanvasContext();
  if (!manager) {
    throw new Error('Canvas manager not initialized');
  }
  return manager;
};

/**
 * Hook to get the canvas manager from the canvas instance context.
 * Returns null if not within a CanvasInstanceProvider or if manager not initialized.
 */
export const useCanvasManagerSafe = (): CanvasManager | null => {
  const context = useCanvasContextSafe();
  return context?.manager || null;
};