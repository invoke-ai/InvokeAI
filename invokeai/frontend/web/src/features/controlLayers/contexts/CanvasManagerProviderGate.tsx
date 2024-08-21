import { useStore } from '@nanostores/react';
import { $canvasManager, type CanvasManager } from 'features/controlLayers/konva/CanvasManager';
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

export const useCanvasManager = (): CanvasManager => {
  const canvasManager = useContext(CanvasManagerContext);
  assert(canvasManager, 'useCanvasManagerContext must be used within a CanvasManagerProviderGate');
  return canvasManager;
};
