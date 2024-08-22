import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useMemo } from 'react';

export const useIsTransforming = () => {
  const canvasManager = useCanvasManager();
  const transformingEntity = useStore(canvasManager.stateApi.$transformingEntity);
  const isTransforming = useMemo(() => {
    return Boolean(transformingEntity);
  }, [transformingEntity]);
  return isTransforming;
};
