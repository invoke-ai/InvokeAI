import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { $canvasManagers } from 'features/controlLayers/store/ephemeral';
import { selectActiveCanvasId } from 'features/controlLayers/store/selectors';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext } from 'react';
import { assert } from 'tsafe';

const CanvasManagerContext = createContext<CanvasManager | null>(null);

export const CanvasManagerProviderGate = memo(({ children }: PropsWithChildren) => {
  const activeCanvasId = useAppSelector(selectActiveCanvasId);
  const canvasManagers = useStore($canvasManagers);
  const canvasManager = activeCanvasId ? canvasManagers.get(activeCanvasId) || null : null;

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
  const activeCanvasId = useAppSelector(selectActiveCanvasId);
  const canvasManagers = useStore($canvasManagers);
  return activeCanvasId ? canvasManagers.get(activeCanvasId) || null : null;
};
