import { useStore } from '@nanostores/react';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { $canvasManager } from 'features/controlLayers/store/ephemeral';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext } from 'react';
import { assert } from 'tsafe';

const CanvasManagerContext = createContext<CanvasManager | null>(null);

export const CanvasManagerProviderGate = memo(({ children }: PropsWithChildren) => {
  const canvasManager = useStore($canvasManager);

  if (!canvasManager) {
    return null;
  }

  return <CanvasManagerContext.Provider value={canvasManager}>{children}</CanvasManagerContext.Provider>;
});

CanvasManagerProviderGate.displayName = 'CanvasManagerProviderGate';

/**
 * Consumes the CanvasManager from the context. This hook must be used within the CanvasManagerProviderGate, otherwise
 * it will throw an error.
 */
export const useCanvasManager = (): CanvasManager => {
  const canvasManager = useContext(CanvasManagerContext);
  assert(canvasManager, 'useCanvasManagerContext must be used within a CanvasManagerProviderGate');
  return canvasManager;
};

/**
 * Consumes the CanvasManager from the context. If the CanvasManager is not available, it will return null.
 */
export const useCanvasManagerSafe = (): CanvasManager | null => {
  const canvasManager = useStore($canvasManager);
  return canvasManager;
};
