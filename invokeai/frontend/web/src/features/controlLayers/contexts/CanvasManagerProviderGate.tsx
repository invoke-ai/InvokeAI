import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { $canvasManagers } from 'features/controlLayers/store/ephemeral';
import { selectActiveCanvasId } from 'features/controlLayers/store/selectors';
import type { PropsWithChildren } from 'react';
import { createContext, memo } from 'react';
import { assert } from 'tsafe';

const CanvasManagerContext = createContext<{ [canvasId: string]: CanvasManager } | null>(null);

export const CanvasManagerProviderGate = memo(({ children }: PropsWithChildren) => {
  const canvasManagers = useStore($canvasManagers);
  const selectedCanvasId = useAppSelector(selectActiveCanvasId);

  if (Object.keys(canvasManagers).length === 0 || !canvasManagers[selectedCanvasId]) {
    return null;
  }

  return <CanvasManagerContext.Provider value={canvasManagers}>{children}</CanvasManagerContext.Provider>;
});

CanvasManagerProviderGate.displayName = 'CanvasManagerProviderGate';

/**
 * Consumes the CanvasManager from the context. If the CanvasManager is not available, it will throw an error.
 */
export const useCanvasManager = (): CanvasManager => {
  const canvasManager = useCanvasManagerSafe();
  assert(canvasManager, 'Canvas manager does not exist');

  return canvasManager;
};

/**
 * Consumes the CanvasManager from the context. If the CanvasManager is not available, it will return null.
 */
export const useCanvasManagerSafe = (): CanvasManager | null => {
  const canvasManagers = useStore($canvasManagers);
  const canvasId = useAppSelector(selectActiveCanvasId);

  return canvasManagers[canvasId] ?? null;
};
