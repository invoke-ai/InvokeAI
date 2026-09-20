import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCallback } from 'react';

/** Returns a stable checker for whether a vector path edit session is active. */
export const useIsCanvasPathEditSessionActive = () => {
  const canvasManager = useCanvasManagerSafe();
  return useCallback(
    () => canvasManager !== null && canvasManager.tool.tools.path.$editSession.get() !== null,
    [canvasManager]
  );
};
