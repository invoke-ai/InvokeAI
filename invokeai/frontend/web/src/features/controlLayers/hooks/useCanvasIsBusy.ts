import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';

export const useCanvasIsBusy = () => {
  const canvasManager = useCanvasManager();
  /**
   * Whether the canvas is busy:
   * - While staging
   * - While an entity is transforming
   * - While an entity is filtering
   * - While the canvas is doing some other long-running operation, like rasterizing a layer
   */
  const isBusy = useStore(canvasManager.$isBusy);

  return isBusy;
};
