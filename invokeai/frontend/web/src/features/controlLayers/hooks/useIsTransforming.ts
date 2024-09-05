import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';

export const useIsTransforming = () => {
  const canvasManager = useCanvasManager();
  const isTransforming = useStore(canvasManager.stateApi.$isTranforming);
  return isTransforming;
};
