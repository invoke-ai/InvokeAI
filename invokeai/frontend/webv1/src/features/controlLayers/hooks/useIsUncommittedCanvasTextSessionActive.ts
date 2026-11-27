import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCallback } from 'react';

/**
 * Returns a stable checker for whether a canvas text session exists but has not been committed/canceled.
 */
export const useIsUncommittedCanvasTextSessionActive = () => {
  const canvasManager = useCanvasManagerSafe();
  return useCallback(
    () => canvasManager !== null && canvasManager.tool.tools.text.$session.get() !== null,
    [canvasManager]
  );
};
