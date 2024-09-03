import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';

export const useCanvasIsBusy = () => {
  const canvasManager = useCanvasManager();
  const isBusy = useStore(canvasManager.$isBusy);

  return isBusy;
};
