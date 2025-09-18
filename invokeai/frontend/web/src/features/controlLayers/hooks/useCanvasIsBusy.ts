import { useStore } from '@nanostores/react';
import { $false } from 'app/store/nanostores/util';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';

/**
 * Returns a boolena indicating whether the canvas is busy:
 * - While staging
 * - While an entity is transforming
 * - While an entity is filtering
 * - While the canvas is doing some other long-running operation, like rasterizing a layer
 *
 * This hook will fall back to false if the canvas manager is not initialized.
 */
export const useCanvasIsBusy = () => {
  const canvasManager = useCanvasManagerSafe();
  const isBusy = useStore(canvasManager?.$isBusy ?? $false);

  return isBusy;
};
