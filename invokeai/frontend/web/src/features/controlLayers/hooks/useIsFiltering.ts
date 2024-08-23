import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';

export const useIsFiltering = () => {
  const canvasManager = useCanvasManager();
  const isFiltering = useStore(canvasManager.filter.$isFiltering);
  return isFiltering;
};
